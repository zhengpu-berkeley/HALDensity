"""Interval-censored data hyperparameter tuners.

Provides:
- IntervalCensoredInitTuner: Stage 1 (Init) tuner using Optuna CV
- IntervalCensoredEMTuner: Stage 2 (EM) tuner with oversmooth or direct no-oversmooth mode
- IntervalCensoredJointTuner: Convenience wrapper running Init -> EM
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold

from haldensity.censoring._defaults import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.interval.estimators import (
    IntervalCensoredInitEstimator,
    IntervalCensoredEMStage,
    IntervalCensoredFISTAEstimator,
    IntervalCensoredProjectedGDEstimator,
)
from haldensity.censoring.interval.metrics import incomplete_loglik_interval

from ._base import (
    BaseCensoredInitTuner,
    BaseCensoredEMTuner,
    BaseCVOversmoothEMTuner,
    TuningResult,
)


class IntervalCensoredInitTuner(BaseCensoredInitTuner):
    """Stage 1 hyperparameter tuner for interval-censored data.
    
    Tunes `norm_constraint` and `basis_order` using Optuna with CV.
    Uses interval log-likelihood: sum log(F(R) - F(L)) as the metric.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with columns 'L' and 'R' (interval endpoints).
    cv_folds : int
        Number of CV folds.
    random_state : int
        Random seed.
    n_grid_points : int
        Number of grid points for density evaluation.
    param_overrides : dict | None
        Override default parameter ranges.
    use_conservative_adjustment : bool
        Whether to apply conservative 1-SE adjustment.
    silent : bool
        Whether to suppress output.
    L_col : str
        Name of left interval column.
    R_col : str
        Name of right interval column.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[Dict[str, Any]] = None,
        use_conservative_adjustment: bool = TUNER_DEFAULTS.use_conservative_adjustment,
        conservative_k_percent: float = TUNER_DEFAULTS.conservative_k_percent,
        conservative_max_steps: int = TUNER_DEFAULTS.conservative_max_steps,
        conservative_step_pct: float = TUNER_DEFAULTS.conservative_step_pct,
        init_knot_strategy: str = "midpoint",
        init_turnbull_tol: float = 1e-5,
        silent: bool = True,
        L_col: str = "L",
        R_col: str = "R",
    ):
        super().__init__(
            data=data,
            cv_folds=cv_folds,
            random_state=random_state,
            n_grid_points=n_grid_points,
            param_overrides=param_overrides,
            use_conservative_adjustment=use_conservative_adjustment,
            conservative_k_percent=conservative_k_percent,
            conservative_max_steps=conservative_max_steps,
            conservative_step_pct=conservative_step_pct,
            silent=silent,
        )
        self.L_col = str(L_col)
        self.R_col = str(R_col)
        self.init_knot_strategy = str(init_knot_strategy)
        self.init_turnbull_tol = float(init_turnbull_tol)
        self._init_grid_points_override: Optional[np.ndarray] = None

        # If using Turnbull knots, compute them once on the full dataset and reuse
        # across CV folds/trials. This avoids repeatedly running Turnbull's NPMLE
        # inside each fold evaluation.
        #
        # Note: This makes the knot grid fixed during CV, analogous to supplying
        # grid_points_override, while still tuning norm_constraint/basis_order.
        strat = self.init_knot_strategy.strip().lower()
        if strat in {"turnbull", "npmle", "turnbull_npmle", "turnbull-npmle"}:
            try:
                from lifelines import KaplanMeierFitter  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "init_knot_strategy='turnbull' requires the optional dependency 'lifelines'."
                ) from exc

            L = np.asarray(self.data[self.L_col].values, dtype=float).ravel()
            R = np.asarray(self.data[self.R_col].values, dtype=float).ravel()
            kmf = KaplanMeierFitter().fit_interval_censoring(L, R, tol=float(self.init_turnbull_tol))
            cdf_df = kmf.cumulative_density_
            if cdf_df is None or cdf_df.shape[1] == 0:
                raise RuntimeError("Turnbull fit failed to produce cumulative density.")
            timeline = cdf_df.index.to_numpy(dtype=float)
            cdf = cdf_df.iloc[:, 0].to_numpy(dtype=float)
            dcdf = np.diff(cdf, prepend=0.0)
            jump_times = timeline[dcdf > 0]
            self._init_grid_points_override = np.unique(
                np.concatenate(([0.0], jump_times.astype(float), [1.0]))
            )
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest tunable parameters."""
        ovr = self.param_overrides
        
        # Basis order
        basis_order_spec = ovr.get("basis_order", [0, 1, 2])
        if isinstance(basis_order_spec, (list, tuple)):
            basis_order = trial.suggest_categorical("basis_order", list(basis_order_spec))
        else:
            basis_order = basis_order_spec
        
        # Norm constraint
        norm_spec = ovr.get("norm_constraint", {"low": 1.0, "high": 1000.0, "log": True})
        if isinstance(norm_spec, dict):
            norm_constraint = trial.suggest_float(
                "norm_constraint",
                norm_spec["low"],
                norm_spec["high"],
                log=norm_spec.get("log", True),
            )
        else:
            norm_constraint = norm_spec
        
        return {"basis_order": basis_order, "norm_constraint": norm_constraint}
    
    def _evaluate_fold(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> float:
        """Evaluate parameters on a single CV fold."""
        est = IntervalCensoredInitEstimator(
            tol=float(self._defaults["tol"]),
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=str(self._defaults["solver"]),
            use_secondary_solver=bool(self._defaults["use_secondary_solver"]),
            include_intercept_in_constraint=False,
        )
        est.fit(
            train_df,
            L_col=self.L_col,
            R_col=self.R_col,
            grid_points_override=self._init_grid_points_override,
            knot_strategy=self.init_knot_strategy,
            turnbull_tol=self.init_turnbull_tol,
        )
        
        return incomplete_loglik_interval(est, val_df, L_col=self.L_col, R_col=self.R_col)
    
    def _fit_final_estimator(self, params: Dict[str, Any]) -> IntervalCensoredInitEstimator:
        """Fit final estimator on full data."""
        est = IntervalCensoredInitEstimator(
            tol=float(self._defaults["tol"]),
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=str(self._defaults["solver"]),
            use_secondary_solver=bool(self._defaults["use_secondary_solver"]),
            include_intercept_in_constraint=False,
        )
        est.fit(
            self.data,
            L_col=self.L_col,
            R_col=self.R_col,
            grid_points_override=self._init_grid_points_override,
            knot_strategy=self.init_knot_strategy,
            turnbull_tol=self.init_turnbull_tol,
        )
        return est


class IntervalCensoredEMTuner(BaseCensoredEMTuner):
    """Stage 2 hyperparameter tuner for interval-censored data.
    
    Supports two modes via `do_over_smooth`:
    - True (default): Grid search over oversmooth factors + EM refinement
    - False: Direct no-oversmooth EM refinement from Stage 1
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with columns 'L' and 'R'.
    stage1_estimator : Any
        Fitted Stage 1 estimator.
    random_state : int
        Random seed.
    n_grid_points : int
        Number of grid points.
    do_over_smooth : bool
        If True, use oversmooth grid. If False, run direct no-oversmooth EM.
    oversmooth_factors : list[float] | None
        Factors for oversmooth grid.
    em_m_imputations : int
        Imputations for EM.
    em_max_em_iter : int
        Max EM iterations.
    em_tol : float
        EM tolerance.
    em_norm_factor : float
        Factor for EM norm constraint.
    selection : str
        Selection criterion ('em_ll' or 'll_gain').
    silent : bool
        Suppress output.
    L_col : str
        Left interval column name.
    R_col : str
        Right interval column name.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        stage1_estimator: Any,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        do_over_smooth: bool = True,
        oversmooth_factors: Optional[list[float]] = None,
        em_m_imputations: int = EM_DEFAULTS.m_imputations,
        em_max_em_iter: int = 20,
        em_tol: float = EM_DEFAULTS.em_tol,
        em_norm_factor: float = 1.0,
        em_e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        selection: str = "em_ll",
        silent: bool = True,
        solver: Optional[str] = None,
        use_secondary_solver: Optional[bool] = None,
        L_col: str = "L",
        R_col: str = "R",
    ):
        # Note: em_use_sc_adjustment is not used for interval censored
        super().__init__(
            data=data,
            stage1_estimator=stage1_estimator,
            random_state=random_state,
            n_grid_points=n_grid_points,
            do_over_smooth=do_over_smooth,
            oversmooth_factors=oversmooth_factors,
            em_m_imputations=em_m_imputations,
            em_max_em_iter=em_max_em_iter,
            em_tol=em_tol,
            em_norm_factor=em_norm_factor,
            em_e_step_n_grid=em_e_step_n_grid,
            em_use_sc_adjustment=False,  # Not applicable for IC
            selection=selection,
            silent=silent,
            solver=solver,
            use_secondary_solver=use_secondary_solver,
        )
        self.L_col = str(L_col)
        self.R_col = str(R_col)
    
    def _fit_init_estimator(self, norm_constraint: float) -> IntervalCensoredInitEstimator:
        """Fit an initialization estimator with given norm_constraint."""
        return IntervalCensoredInitEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            include_intercept_in_constraint=False,
        ).fit(self.data, L_col=self.L_col, R_col=self.R_col)
    
    def _compute_loglik(self, estimator: Any) -> float:
        """Compute log-likelihood for the given estimator."""
        return incomplete_loglik_interval(estimator, self.data, L_col=self.L_col, R_col=self.R_col)
    
    def _run_em_stage(
        self, 
        initial_estimator: Any, 
        m_step_norm_constraint: float
    ) -> Any:
        """Run EM stage and return the EM result."""
        em_stage = IntervalCensoredEMStage(
            m_imputations=self.em_m_imputations,
            max_em_iter=self.em_max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            e_step_n_grid=self.em_e_step_n_grid,
            tol=EM_DEFAULTS.tol,
            m_step_solver=self.solver,
            verbose=not self.silent,
            rng_seed=self.random_state,
            L_col=self.L_col,
            R_col=self.R_col,
        )
        
        return em_stage.run(
            initial_estimator=initial_estimator,
            data=self.data,
        )


class IntervalCensoredJointTuner:
    """Convenience wrapper that runs Stage 1 -> Stage 2 tuning.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with columns 'L' and 'R'.
    cv_folds : int
        CV folds for Stage 1.
    random_state : int
        Random seed.
    n_grid_points : int
        Grid points.
    stage1_n_trials : int
        Optuna trials for Stage 1.
    do_over_smooth : bool
        Mode for Stage 2.
    oversmooth_factors : list[float] | None
        Factors for Stage 2 oversmooth.
    em_m_imputations : int
        EM imputations.
    em_max_em_iter : int
        EM max iterations.
    silent : bool
        Suppress output.
    L_col : str
        Left interval column.
    R_col : str
        Right interval column.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        stage1_n_trials: int = 50,
        do_over_smooth: bool = True,
        oversmooth_factors: Optional[list[float]] = None,
        em_m_imputations: int = EM_DEFAULTS.m_imputations,
        em_max_em_iter: int = 20,
        silent: bool = True,
        use_conservative_adjustment: bool = TUNER_DEFAULTS.use_conservative_adjustment,
        init_knot_strategy: str = "midpoint",
        init_turnbull_tol: float = 1e-5,
        L_col: str = "L",
        R_col: str = "R",
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_grid_points = n_grid_points
        self.stage1_n_trials = stage1_n_trials
        self.do_over_smooth = do_over_smooth
        self.oversmooth_factors = oversmooth_factors
        self.em_m_imputations = em_m_imputations
        self.em_max_em_iter = em_max_em_iter
        self.silent = silent
        self.use_conservative_adjustment = use_conservative_adjustment
        self.init_knot_strategy = str(init_knot_strategy)
        self.init_turnbull_tol = float(init_turnbull_tol)
        self.L_col = L_col
        self.R_col = R_col
        
        # Results
        self.stage1_result: Optional[TuningResult] = None
        self.stage2_result: Optional[TuningResult] = None
    
    def optimize(self) -> Tuple[TuningResult, TuningResult]:
        """Run both stages and return results.
        
        Returns
        -------
        Tuple[TuningResult, TuningResult]
            (stage1_result, stage2_result)
        """
        # Stage 1: Init tuning
        init_tuner = IntervalCensoredInitTuner(
            data=self.data,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            n_grid_points=self.n_grid_points,
            use_conservative_adjustment=self.use_conservative_adjustment,
            init_knot_strategy=self.init_knot_strategy,
            init_turnbull_tol=self.init_turnbull_tol,
            silent=self.silent,
            L_col=self.L_col,
            R_col=self.R_col,
        )
        self.stage1_result = init_tuner.optimize(n_trials=self.stage1_n_trials)
        
        # Stage 2: EM tuning
        em_tuner = IntervalCensoredEMTuner(
            data=self.data,
            stage1_estimator=self.stage1_result.estimator,
            random_state=self.random_state,
            n_grid_points=self.n_grid_points,
            do_over_smooth=self.do_over_smooth,
            oversmooth_factors=self.oversmooth_factors,
            em_m_imputations=self.em_m_imputations,
            em_max_em_iter=self.em_max_em_iter,
            silent=self.silent,
            L_col=self.L_col,
            R_col=self.R_col,
        )
        self.stage2_result = em_tuner.optimize()
        
        return (self.stage1_result, self.stage2_result)


class IntervalCensoredCVOversmoothEMTuner(BaseCVOversmoothEMTuner):
    """CV-based oversmooth EM tuner for interval-censored data.

    Performs K-fold cross-validation with Optuna over
    ``(oversmooth_factor, em_norm_factor)`` to select the combination that
    maximises held-out incomplete-data log-likelihood, then refits with the
    winner on the full dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Data with columns ``L_col`` and ``R_col``.
    stage1_estimator : Any
        Fitted Stage 1 estimator.
    cv_folds : int
        Number of CV folds.
    random_state : int
        Random seed.
    n_grid_points : int
        Number of grid points.
    oversmooth_factors : list[float] | None
        Range specification for oversmooth factor. If one value, fixed; if
        multiple, min/max define search range. Default [0.1, 1.0].
    em_norm_factors : list[float] | None
        Range specification for EM norm factor. If one value, fixed; if
        multiple, min/max define search range. Default [1.0, 5.0].
    em_m_imputations : int
        Imputations for EM E-step.
    em_max_em_iter : int
        Max EM iterations.
    em_tol : float
        EM convergence tolerance.
    em_e_step_n_grid : int
        Grid size for E-step inverse CDF.
    silent : bool
        Suppress output.
    solver : str | None
        Solver to use.
    use_secondary_solver : bool | None
        Whether to use secondary solver.
    L_col : str
        Left interval column name.
    R_col : str
        Right interval column name.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        stage1_estimator: Any,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        oversmooth_factors: Optional[list[float]] = None,
        em_norm_factors: Optional[list[float]] = None,
        em_m_imputations: int = EM_DEFAULTS.m_imputations,
        em_max_em_iter: int = 20,
        em_tol: float = EM_DEFAULTS.em_tol,
        em_e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        silent: bool = True,
        solver: Optional[str] = None,
        use_secondary_solver: Optional[bool] = None,
        L_col: str = "L",
        R_col: str = "R",
    ):
        super().__init__(
            data=data,
            stage1_estimator=stage1_estimator,
            cv_folds=cv_folds,
            random_state=random_state,
            n_grid_points=n_grid_points,
            oversmooth_factors=oversmooth_factors,
            em_norm_factors=em_norm_factors,
            em_m_imputations=em_m_imputations,
            em_max_em_iter=em_max_em_iter,
            em_tol=em_tol,
            em_e_step_n_grid=em_e_step_n_grid,
            silent=silent,
            solver=solver,
            use_secondary_solver=use_secondary_solver,
        )
        self.L_col = str(L_col)
        self.R_col = str(R_col)

    def _fit_init_estimator_on_data(
        self, data: pd.DataFrame, norm_constraint: float,
    ) -> IntervalCensoredInitEstimator:
        # Important: keep the Stage-1 knot structure fixed during CV-oversmooth.
        #
        # Otherwise, if the provided stage1_estimator used a non-default knot strategy
        # (e.g. Turnbull mass-point knots), refits inside CV (and oversmooth_factor != 1)
        # would silently fall back to the default midpoint knot grid, which can create
        # a much larger basis and make EM appear to "gain knots" compared to the
        # original initializer.
        grid_override = getattr(self.stage1_estimator, "_grid_points_hal", None)
        return IntervalCensoredInitEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            include_intercept_in_constraint=False,
        ).fit(
            data,
            L_col=self.L_col,
            R_col=self.R_col,
            grid_points_override=np.asarray(grid_override, dtype=float)
            if grid_override is not None and len(grid_override) > 0
            else None,
        )

    def _run_em_stage_on_data(
        self,
        initial_estimator: Any,
        data: pd.DataFrame,
        m_step_norm_constraint: float,
    ) -> Any:
        em_stage = IntervalCensoredEMStage(
            m_imputations=self.em_m_imputations,
            max_em_iter=self.em_max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            e_step_n_grid=self.em_e_step_n_grid,
            tol=EM_DEFAULTS.tol,
            m_step_solver=self.solver,
            verbose=False,
            rng_seed=self.random_state,
            L_col=self.L_col,
            R_col=self.R_col,
        )
        return em_stage.run(initial_estimator=initial_estimator, data=data)

    def _compute_loglik_on_data(
        self, estimator: Any, data: pd.DataFrame,
    ) -> float:
        return incomplete_loglik_interval(
            estimator, data, L_col=self.L_col, R_col=self.R_col,
        )


class IntervalCensoredFISTATuner:
    """Optuna CV tuner for direct interval-censored FISTA HAL-MLE."""

    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[Dict[str, Any]] = None,
        silent: bool = True,
        L_col: str = "L",
        R_col: str = "R",
        knot_strategy: str = "midpoint",
        turnbull_tol: float = 1e-5,
        fista_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.param_overrides = param_overrides or {}
        self.silent = bool(silent)
        self.L_col = str(L_col)
        self.R_col = str(R_col)
        self.knot_strategy = str(knot_strategy)
        self.turnbull_tol = float(turnbull_tol)
        self.fista_kwargs = dict(fista_kwargs or {})

        self.kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        ovr = self.param_overrides

        basis_order_spec = ovr.get("basis_order", [0, 1])
        if isinstance(basis_order_spec, (list, tuple)):
            basis_order = trial.suggest_categorical("basis_order", list(basis_order_spec))
        else:
            basis_order = int(basis_order_spec)

        lam_spec = ovr.get("lam", {"low": 1e-4, "high": 1.0, "log": True})
        if isinstance(lam_spec, dict):
            lam = trial.suggest_float(
                "lam",
                float(lam_spec["low"]),
                float(lam_spec["high"]),
                log=bool(lam_spec.get("log", True)),
            )
        else:
            lam = float(lam_spec)

        return {"basis_order": int(basis_order), "lam": float(lam)}

    def _evaluate_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame, params: Dict[str, Any]) -> float:
        est = IntervalCensoredFISTAEstimator(
            lam=float(params["lam"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(params["basis_order"]),
            **self.fista_kwargs,
        )
        est.fit(
            train_df,
            L_col=self.L_col,
            R_col=self.R_col,
            knot_strategy=self.knot_strategy,
            turnbull_tol=self.turnbull_tol,
        )
        return incomplete_loglik_interval(est, val_df, L_col=self.L_col, R_col=self.R_col)

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        scores: list[float] = []
        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)
            try:
                score = self._evaluate_fold(train_df, val_df, params)
                scores.append(score)
            except Exception:
                scores.append(float("-inf"))
        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def optimize(self, n_trials: int = 30) -> TuningResult:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self._objective, n_trials=int(n_trials), show_progress_bar=(not self.silent))

        self.best_params = {
            "basis_order": int(self.study.best_params["basis_order"]),
            "lam": float(self.study.best_params["lam"]),
        }
        self.best_metric_value = -float(self.study.best_value)

        estimator = IntervalCensoredFISTAEstimator(
            lam=float(self.best_params["lam"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(self.best_params["basis_order"]),
            **self.fista_kwargs,
        )
        estimator.fit(
            self.data,
            L_col=self.L_col,
            R_col=self.R_col,
            knot_strategy=self.knot_strategy,
            turnbull_tol=self.turnbull_tol,
        )

        metadata = {
            "best_metric_value": self.best_metric_value,
            "study": self.study,
            "cv_folds": self.cv_folds,
        }
        return TuningResult(
            estimator=estimator,
            best_params=self.best_params,
            metadata=metadata,
        )


class IntervalCensoredProjectedGDTuner:
    """Optuna CV tuner for direct interval-censored projected-GD HAL-MLE."""

    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[Dict[str, Any]] = None,
        silent: bool = True,
        L_col: str = "L",
        R_col: str = "R",
        knot_strategy: str = "midpoint",
        turnbull_tol: float = 1e-5,
        pgd_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.param_overrides = param_overrides or {}
        self.silent = bool(silent)
        self.L_col = str(L_col)
        self.R_col = str(R_col)
        self.knot_strategy = str(knot_strategy)
        self.turnbull_tol = float(turnbull_tol)
        self.pgd_kwargs = dict(pgd_kwargs or {})

        self.kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        ovr = self.param_overrides

        basis_order_spec = ovr.get("basis_order", [0, 1])
        if isinstance(basis_order_spec, (list, tuple)):
            basis_order = trial.suggest_categorical("basis_order", list(basis_order_spec))
        else:
            basis_order = int(basis_order_spec)

        nc_spec = ovr.get("norm_constraint", {"low": 1.0, "high": 100.0, "log": True})
        if isinstance(nc_spec, dict):
            norm_constraint = trial.suggest_float(
                "norm_constraint",
                float(nc_spec["low"]),
                float(nc_spec["high"]),
                log=bool(nc_spec.get("log", True)),
            )
        else:
            norm_constraint = float(nc_spec)

        return {"basis_order": int(basis_order), "norm_constraint": float(norm_constraint)}

    def _evaluate_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame, params: Dict[str, Any]) -> float:
        est = IntervalCensoredProjectedGDEstimator(
            norm_constraint=float(params["norm_constraint"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(params["basis_order"]),
            **self.pgd_kwargs,
        )
        est.fit(
            train_df,
            L_col=self.L_col,
            R_col=self.R_col,
            knot_strategy=self.knot_strategy,
            turnbull_tol=self.turnbull_tol,
        )
        return incomplete_loglik_interval(est, val_df, L_col=self.L_col, R_col=self.R_col)

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        scores: list[float] = []
        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)
            try:
                score = self._evaluate_fold(train_df, val_df, params)
                scores.append(score)
            except Exception:
                scores.append(float("-inf"))
        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def optimize(self, n_trials: int = 30) -> TuningResult:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self._objective, n_trials=int(n_trials), show_progress_bar=(not self.silent))

        self.best_params = {
            "basis_order": int(self.study.best_params["basis_order"]),
            "norm_constraint": float(self.study.best_params["norm_constraint"]),
        }
        self.best_metric_value = -float(self.study.best_value)

        estimator = IntervalCensoredProjectedGDEstimator(
            norm_constraint=float(self.best_params["norm_constraint"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(self.best_params["basis_order"]),
            **self.pgd_kwargs,
        )
        estimator.fit(
            self.data,
            L_col=self.L_col,
            R_col=self.R_col,
            knot_strategy=self.knot_strategy,
            turnbull_tol=self.turnbull_tol,
        )

        metadata = {
            "best_metric_value": self.best_metric_value,
            "study": self.study,
            "cv_folds": self.cv_folds,
        }
        return TuningResult(
            estimator=estimator,
            best_params=self.best_params,
            metadata=metadata,
        )
