"""Right-censored data hyperparameter tuners.

Provides:
- RightCensoredInitTuner: Stage 1 (Init) tuner using Optuna CV
- RightCensoredEMTuner: Stage 2 (EM) tuner with oversmooth or direct no-oversmooth mode
- RightCensoredJointTuner: Convenience wrapper running Init -> EM
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold

from haldensity.censoring._defaults import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.right.estimators import (
    RightCensoredInitEstimator,
    RightCensoredEMStage,
)
from haldensity.censoring.right.observed_mle import (
    RightCensoredObservedFISTAEstimator,
    RightCensoredObservedFPGDEstimator,
    RightCensoredObservedL1MLE,
)
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.metrics import incomplete_loglik, ipcw_loglik

from ._base import (
    BaseCensoredInitTuner,
    BaseCensoredEMTuner,
    BaseCVOversmoothEMTuner,
    TuningResult,
)


def _compress_right_theta_to_support(
    estimator: Any,
    support: np.ndarray,
) -> np.ndarray:
    """Project a fitted right-censored theta onto a target support."""
    support = np.asarray(support, dtype=float)
    all_knots = np.asarray(estimator._grid_points_hal, dtype=float)
    theta_full = np.asarray(estimator.theta_hat, dtype=float)
    basis_order = int(estimator.basis_order)
    poly_cols = basis_order if basis_order > 0 else 0
    knot_start = 1 + poly_cols

    theta_selected = np.zeros(knot_start + support.size, dtype=float)
    theta_selected[:knot_start] = theta_full[:knot_start]
    for i, knot in enumerate(support):
        idx = np.where(np.isclose(all_knots, knot, atol=1e-10, rtol=0.0))[0]
        if idx.size > 0:
            theta_selected[knot_start + i] = theta_full[knot_start + int(idx[0])]
    return theta_selected


def _compress_right_theta_to_selected_support(estimator: Any) -> np.ndarray:
    """Project a fitted Stage-1 theta onto its selected fixed support."""
    return _compress_right_theta_to_support(
        estimator,
        np.asarray(estimator.grid_points_hal_selected, dtype=float),
    )


class RightCensoredInitTuner(BaseCensoredInitTuner):
    """Stage 1 hyperparameter tuner for right-censored data.
    
    Tunes `norm_constraint` and `basis_order` using Optuna with CV.
    Uses incomplete-data log-likelihood as the metric.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with columns 'T' (time) and 'Delta' (event indicator).
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
    conservative_k_percent : float
        Threshold for conservative adjustment.
    conservative_max_steps : int
        Maximum conservative search steps.
    conservative_step_pct : float
        Step size for conservative search.
    silent : bool
        Whether to suppress output.
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
        silent: bool = True,
        validation_metric: str = "observed_loglik",
        clip: float = 1e-6,
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
        metric = str(validation_metric).strip().lower()
        if metric not in {"observed_loglik", "ipcw_loglik"}:
            raise ValueError("validation_metric must be 'observed_loglik' or 'ipcw_loglik'")
        self.validation_metric = metric
        self.clip = float(clip)

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
    
    def _score_validation_fold(
        self,
        estimator: RightCensoredInitEstimator,
        val_df: pd.DataFrame,
        km: KaplanMeier,
    ) -> float:
        if self.validation_metric == "ipcw_loglik":
            return ipcw_loglik(
                estimator,
                val_df,
                time_col="T",
                delta_col="Delta",
                km=km,
                clip=self.clip,
            )
        return incomplete_loglik(estimator, val_df, time_col="T", delta_col="Delta")

    def _evaluate_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        params: Dict[str, Any]
    ) -> float:
        """Evaluate parameters on a single CV fold."""
        # Fit KM on training data
        km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
        
        T_vals = np.asarray(train_df["T"].values, dtype=float)
        Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        est = RightCensoredInitEstimator(
            tol=float(self._defaults["tol"]),
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=str(self._defaults["solver"]),
            use_secondary_solver=bool(self._defaults["use_secondary_solver"]),
        )
        est.fit(df_unc, sample_weights=w_unc)

        return self._score_validation_fold(est, val_df, km)
    
    def _fit_final_estimator(self, params: Dict[str, Any]) -> RightCensoredInitEstimator:
        """Fit final estimator on full data."""
        km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
        
        T_vals = np.asarray(self.data["T"].values, dtype=float)
        Delta_vals = np.asarray(self.data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        est = RightCensoredInitEstimator(
            tol=float(self._defaults["tol"]),
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=str(self._defaults["solver"]),
            use_secondary_solver=bool(self._defaults["use_secondary_solver"]),
        )
        est.fit(df_unc, sample_weights=w_unc)
        return est

    def optimize(self, n_trials: int = 50) -> TuningResult:
        result = super().optimize(n_trials=n_trials)
        result.metadata["validation_metric"] = self.validation_metric
        result.metadata["validation_metric_clip"] = self.clip
        return result


class RightCensoredEMTuner(BaseCensoredEMTuner):
    """Stage 2 hyperparameter tuner for right-censored data.
    
    Supports two modes via `do_over_smooth`:
    - True (default): Grid search over oversmooth factors + EM refinement
    - False: Direct no-oversmooth EM refinement from Stage 1
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with columns 'T' and 'Delta'.
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
    solver : str | None
        Solver override.
    use_secondary_solver : bool | None
        Secondary solver override.
    """
    
    def _fit_init_estimator(self, norm_constraint: float) -> RightCensoredInitEstimator:
        """Fit an initialization estimator with given norm_constraint."""
        km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
        
        T_vals = np.asarray(self.data["T"].values, dtype=float)
        Delta_vals = np.asarray(self.data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        est = RightCensoredInitEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
        )
        est.fit(df_unc, sample_weights=w_unc)
        return est
    
    def _compute_loglik(self, estimator: Any) -> float:
        """Compute log-likelihood for the given estimator."""
        return incomplete_loglik(estimator, self.data, time_col="T", delta_col="Delta")
    
    def _run_em_stage(
        self, 
        initial_estimator: Any, 
        m_step_norm_constraint: float
    ) -> Any:
        """Run EM stage and return the EM result."""
        km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
        
        def s_c_predict(t: np.ndarray) -> np.ndarray:
            return np.atleast_1d(km.predict(t))
        
        em_stage = RightCensoredEMStage(
            m_imputations=self.em_m_imputations,
            max_em_iter=self.em_max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            use_sc_adjustment=self.em_use_sc_adjustment,
            e_step_n_grid=self.em_e_step_n_grid,
            tol=EM_DEFAULTS.tol,
            m_step_solver=self.solver,
            verbose=not self.silent,
            rng_seed=self.random_state,
        )
        
        return em_stage.run(
            initial_estimator=initial_estimator,
            data=self.data,
            S_c_predict=s_c_predict,
        )


class RightCensoredJointTuner:
    """Convenience wrapper that runs Stage 1 -> Stage 2 tuning.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with columns 'T' and 'Delta'.
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
        init_tuner = RightCensoredInitTuner(
            data=self.data,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            n_grid_points=self.n_grid_points,
            use_conservative_adjustment=self.use_conservative_adjustment,
            silent=self.silent,
        )
        self.stage1_result = init_tuner.optimize(n_trials=self.stage1_n_trials)
        
        # Stage 2: EM tuning
        em_tuner = RightCensoredEMTuner(
            data=self.data,
            stage1_estimator=self.stage1_result.estimator,
            random_state=self.random_state,
            n_grid_points=self.n_grid_points,
            do_over_smooth=self.do_over_smooth,
            oversmooth_factors=self.oversmooth_factors,
            em_m_imputations=self.em_m_imputations,
            em_max_em_iter=self.em_max_em_iter,
            silent=self.silent,
        )
        self.stage2_result = em_tuner.optimize()
        
        return (self.stage1_result, self.stage2_result)


class RightCensoredCVOversmoothEMTuner(BaseCVOversmoothEMTuner):
    """CV-based oversmooth EM tuner for right-censored data.

    Performs K-fold cross-validation with Optuna over
    ``(oversmooth_factor, em_norm_factor)`` to select the combination that
    maximizes held-out incomplete-data log-likelihood, then refits with the
    winner on the full dataset.
    """

    def _fit_init_estimator_on_data(
        self, data: pd.DataFrame, norm_constraint: float,
    ) -> RightCensoredInitEstimator:
        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")

        t_vals = np.asarray(data["T"].values, dtype=float)
        delta_vals = np.asarray(data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(t_vals, delta_vals, lambda t: np.atleast_1d(km.predict(t)))

        unc_mask = delta_vals == 1
        df_unc = pd.DataFrame({"W1": t_vals[unc_mask]})
        w_unc = weights[unc_mask]

        return RightCensoredInitEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
        ).fit(df_unc, sample_weights=w_unc)

    def _run_em_stage_on_data(
        self,
        initial_estimator: Any,
        data: pd.DataFrame,
        m_step_norm_constraint: float,
    ) -> Any:
        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")

        def s_c_predict(t: np.ndarray) -> np.ndarray:
            return np.atleast_1d(km.predict(t))

        em_stage = RightCensoredEMStage(
            m_imputations=self.em_m_imputations,
            max_em_iter=self.em_max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            use_sc_adjustment=self.em_use_sc_adjustment,
            e_step_n_grid=self.em_e_step_n_grid,
            tol=EM_DEFAULTS.tol,
            m_step_solver=self.solver,
            verbose=False,
            rng_seed=self.random_state,
        )
        return em_stage.run(
            initial_estimator=initial_estimator,
            data=data,
            S_c_predict=s_c_predict,
        )

    def _compute_loglik_on_data(
        self, estimator: Any, data: pd.DataFrame,
    ) -> float:
        return incomplete_loglik(estimator, data, time_col="T", delta_col="Delta")


class RightCensoredObservedFISTATuner:
    """Optuna CV tuner for direct right-censored observed-data FISTA HAL-MLE."""

    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[Dict[str, Any]] = None,
        silent: bool = True,
        time_col: str = "T",
        delta_col: str = "Delta",
        fista_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.param_overrides = param_overrides or {}
        self.silent = bool(silent)
        self.time_col = str(time_col)
        self.delta_col = str(delta_col)
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
        est = RightCensoredObservedFISTAEstimator(
            lam=float(params["lam"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(params["basis_order"]),
            **self.fista_kwargs,
        )
        est.fit(train_df, time_col=self.time_col, delta_col=self.delta_col)
        return incomplete_loglik(est, val_df, time_col=self.time_col, delta_col=self.delta_col)

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

        estimator = RightCensoredObservedFISTAEstimator(
            lam=float(self.best_params["lam"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(self.best_params["basis_order"]),
            **self.fista_kwargs,
        )
        estimator.fit(self.data, time_col=self.time_col, delta_col=self.delta_col)

        metadata = {
            "best_metric_value": self.best_metric_value,
            "study": self.study,
            "cv_folds": self.cv_folds,
            "validation_metric": "observed_loglik",
        }
        return TuningResult(
            estimator=estimator,
            best_params=self.best_params,
            metadata=metadata,
        )


class RightCensoredObservedFPGDTuner:
    """Optuna CV tuner for direct right-censored observed-data FPGD HAL-MLE."""

    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[Dict[str, Any]] = None,
        silent: bool = True,
        time_col: str = "T",
        delta_col: str = "Delta",
        fpgd_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.param_overrides = param_overrides or {}
        self.silent = bool(silent)
        self.time_col = str(time_col)
        self.delta_col = str(delta_col)
        self.fpgd_kwargs = dict(fpgd_kwargs or {})

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
        est = RightCensoredObservedFPGDEstimator(
            norm_constraint=float(params["norm_constraint"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(params["basis_order"]),
            **self.fpgd_kwargs,
        )
        est.fit(train_df, time_col=self.time_col, delta_col=self.delta_col)
        return incomplete_loglik(est, val_df, time_col=self.time_col, delta_col=self.delta_col)

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

        estimator = RightCensoredObservedFPGDEstimator(
            norm_constraint=float(self.best_params["norm_constraint"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(self.best_params["basis_order"]),
            **self.fpgd_kwargs,
        )
        estimator.fit(self.data, time_col=self.time_col, delta_col=self.delta_col)

        metadata = {
            "best_metric_value": self.best_metric_value,
            "study": self.study,
            "cv_folds": self.cv_folds,
            "validation_metric": "observed_loglik",
        }
        return TuningResult(
            estimator=estimator,
            best_params=self.best_params,
            metadata=metadata,
        )


class RightCensoredObservedL1Tuner:
    """Deterministic pathwise CV tuner for fixed-support observed-data L1 HAL-MLE.

    The working model is fixed to the selected Stage-1 support provided by
    ``stage1_estimator``. Cross-validation tunes a scalar multiplier applied to
    the Stage-1 norm constraint over a deterministic path, using the Stage-1
    fit as the natural warm start near ``1.0 * M1`` and then continuing along
    the path with observed-data warm starts.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        stage1_estimator: Any,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        norm_constraint_factors: Optional[list[float]] = None,
        silent: bool = True,
        time_col: str = "T",
        delta_col: str = "Delta",
        warm_start_final: bool = True,
        warm_start_cv: bool = True,
        l1_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.stage1_estimator = stage1_estimator
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.silent = bool(silent)
        self.time_col = str(time_col)
        self.delta_col = str(delta_col)
        self.warm_start_final = bool(warm_start_final)
        self.warm_start_cv = bool(warm_start_cv)
        self.l1_kwargs = dict(l1_kwargs or {})

        forbidden = {"working_grid_points", "norm_constraint", "basis_order", "warm_start_theta"}
        overlap = forbidden.intersection(self.l1_kwargs)
        if overlap:
            bad = ", ".join(sorted(overlap))
            raise ValueError(
                "l1_kwargs must not override fixed-support tuner arguments; "
                f"remove: {bad}"
            )

        self.base_norm_constraint = float(getattr(
            stage1_estimator, "norm_constraint",
            getattr(stage1_estimator, "_norm_constraint", 100.0),
        ))
        self.basis_order = int(stage1_estimator.basis_order)
        self.working_grid_points = np.asarray(
            stage1_estimator.grid_points_hal_selected,
            dtype=float,
        ).copy()
        self.init_tol = float(getattr(stage1_estimator, "tol", EM_DEFAULTS.tol))
        self.init_n_grid_points = int(
            getattr(stage1_estimator, "n_grid_points", TUNER_DEFAULTS.n_grid_points)
        )
        self.init_solver = str(getattr(stage1_estimator, "solver", TUNER_DEFAULTS.solver))
        self.init_use_secondary_solver = bool(
            getattr(stage1_estimator, "use_secondary_solver", TUNER_DEFAULTS.use_secondary_solver)
        )
        self.init_include_intercept_in_constraint = bool(
            getattr(stage1_estimator, "include_intercept_in_constraint", False)
        )
        self.final_warm_start_theta = (
            _compress_right_theta_to_selected_support(stage1_estimator)
            if self.warm_start_final
            else None
        )
        self.norm_constraint_factors = self._normalize_requested_factors(norm_constraint_factors)

        self.kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        self._fold_splits = [
            (train_idx.copy(), val_idx.copy())
            for train_idx, val_idx in self.kfold.split(self.data)
        ]
        self._cv_fold_warm_start_cache: Dict[int, Optional[np.ndarray]] = {}
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        self.path_factors_: Optional[list[float]] = None
        self.path_anchor_factor_: Optional[float] = None

    @staticmethod
    def _default_factor_grid() -> list[float]:
        return [float(x) for x in np.round(np.linspace(1.0, 2.0, 11), 2)]

    @classmethod
    def _normalize_requested_factors(
        cls,
        values: Optional[list[float]],
    ) -> list[float]:
        factors = cls._default_factor_grid() if values is None else [float(x) for x in values]
        if len(factors) == 0:
            raise ValueError("norm_constraint_factors must contain at least one value")
        if any(x <= 0.0 for x in factors):
            raise ValueError("norm_constraint_factors must be positive")
        return sorted({float(x) for x in factors})

    @staticmethod
    def _ensure_anchor_factor(values: list[float]) -> list[float]:
        factors = sorted({float(x) for x in values})
        if len(factors) == 0:
            raise ValueError("norm_constraint_factors must contain at least one value")
        if factors[0] < 1.0 < factors[-1] and not any(np.isclose(x, 1.0) for x in factors):
            factors.append(1.0)
            factors = sorted({float(x) for x in factors})
        return factors

    def _resolve_path_factors(self, n_trials: Optional[int]) -> list[float]:
        factors = list(self.norm_constraint_factors)
        if len(factors) == 2 and n_trials is not None and int(n_trials) >= 2:
            lo = float(min(factors))
            hi = float(max(factors))
            factors = [float(x) for x in np.linspace(lo, hi, int(n_trials))]
        return self._ensure_anchor_factor(factors)

    @staticmethod
    def _build_path_sequence(
        factors: list[float],
    ) -> Tuple[float, list[float], list[float]]:
        ordered = sorted({float(x) for x in factors})
        if len(ordered) == 0:
            raise ValueError("At least one norm-constraint factor is required")
        exact_anchor = [x for x in ordered if np.isclose(x, 1.0)]
        if exact_anchor:
            anchor = float(exact_anchor[0])
        else:
            anchor = float(min(ordered, key=lambda x: (abs(x - 1.0), x)))
        upward = [float(x) for x in ordered if x > anchor and not np.isclose(x, anchor)]
        downward = [float(x) for x in ordered if x < anchor and not np.isclose(x, anchor)]
        downward.reverse()
        return anchor, upward, downward

    @staticmethod
    def _summarize_path_records(path_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for factor, group in path_df.groupby("norm_constraint_factor", sort=True):
            scores = group["validation_score"].to_numpy(dtype=float)
            valid_scores = scores[np.isfinite(scores)]
            runtimes = group["runtime_seconds"].to_numpy(dtype=float)
            runtimes = runtimes[np.isfinite(runtimes)]
            fit_iterations = group["n_iterations_run"].to_numpy(dtype=float)
            fit_iterations = fit_iterations[np.isfinite(fit_iterations)]
            converged = group["converged"].to_numpy(dtype=float)
            converged = converged[np.isfinite(converged)]
            learning_rates = group["final_learning_rate"].to_numpy(dtype=float)
            learning_rates = learning_rates[np.isfinite(learning_rates)]
            recovery = group["recovery_count"].to_numpy(dtype=float)
            recovery = recovery[np.isfinite(recovery)]
            rows.append({
                "norm_constraint_factor": float(factor),
                "norm_constraint": float(group["norm_constraint"].iloc[0]),
                "mean_cv_loglik": float(np.mean(valid_scores)) if valid_scores.size else float("-inf"),
                "sd_cv_loglik": (
                    float(np.std(valid_scores, ddof=1)) if valid_scores.size >= 2 else 0.0
                ),
                "n_valid_folds": int(valid_scores.size),
                "mean_runtime_seconds": float(np.mean(runtimes)) if runtimes.size else float("nan"),
                "mean_fit_iterations": (
                    float(np.mean(fit_iterations)) if fit_iterations.size else float("nan")
                ),
                "convergence_rate": float(np.mean(converged)) if converged.size else float("nan"),
                "mean_final_learning_rate": (
                    float(np.mean(learning_rates)) if learning_rates.size else float("nan")
                ),
                "mean_recovery_count": float(np.mean(recovery)) if recovery.size else float("nan"),
            })
        return pd.DataFrame(rows).sort_values("norm_constraint_factor").reset_index(drop=True)

    @staticmethod
    def _select_best_path_row(path_df: pd.DataFrame) -> pd.Series:
        valid = path_df[np.isfinite(path_df["mean_cv_loglik"])].copy()
        if valid.empty:
            raise RuntimeError("No finite CVL1 path values were produced")
        return valid.sort_values(
            ["mean_cv_loglik", "norm_constraint_factor"],
            ascending=[False, True],
        ).iloc[0]

    def _build_fold_warm_start_theta(self, train_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Fit a fold-local IPCW initializer and project it to the fixed support."""
        t_vals = np.asarray(train_df[self.time_col].values, dtype=float)
        delta_vals = np.asarray(train_df[self.delta_col].values, dtype=int)
        unc_mask = delta_vals == 1
        if not np.any(unc_mask):
            return None

        km = KaplanMeier().fit(train_df, time_col=self.time_col, delta_col=self.delta_col)
        weights = compute_ipcw_weights(
            t_vals,
            delta_vals,
            lambda t: np.atleast_1d(km.predict(t)),
        )

        init_est = RightCensoredInitEstimator(
            tol=self.init_tol,
            norm_constraint=self.base_norm_constraint,
            n_grid_points=self.init_n_grid_points,
            basis_order=self.basis_order,
            solver=self.init_solver,
            use_secondary_solver=self.init_use_secondary_solver,
            include_intercept_in_constraint=self.init_include_intercept_in_constraint,
        )
        init_est.fit(
            pd.DataFrame({"W1": t_vals[unc_mask]}),
            sample_weights=weights[unc_mask],
            grid_points_override=self.working_grid_points,
            skip_coefficient_pruning=True,
        )
        if init_est.theta_hat is None:
            return None
        return _compress_right_theta_to_support(init_est, self.working_grid_points)

    def _get_fold_warm_start_theta(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Return a cached fold-local warm start, falling back to cold if needed."""
        if not self.warm_start_cv:
            return None
        if fold_idx not in self._cv_fold_warm_start_cache:
            try:
                self._cv_fold_warm_start_cache[fold_idx] = self._build_fold_warm_start_theta(train_df)
            except Exception:
                self._cv_fold_warm_start_cache[fold_idx] = None
        warm_start = self._cv_fold_warm_start_cache[fold_idx]
        return None if warm_start is None else np.asarray(warm_start, dtype=float).copy()

    def _fit_factor(
        self,
        fit_df: pd.DataFrame,
        factor: float,
        *,
        warm_start_theta: Optional[np.ndarray],
    ) -> Tuple[RightCensoredObservedL1MLE, dict[str, Any], float]:
        est = RightCensoredObservedL1MLE(
            working_grid_points=self.working_grid_points,
            norm_constraint=float(factor) * self.base_norm_constraint,
            basis_order=self.basis_order,
            warm_start_theta=warm_start_theta,
            **self.l1_kwargs,
        )
        tic = time.perf_counter()
        est.fit(fit_df, time_col=self.time_col, delta_col=self.delta_col)
        runtime_seconds = float(time.perf_counter() - tic)
        return est, est.get_results(), runtime_seconds

    def _evaluate_factor_path(
        self,
        fit_df: pd.DataFrame,
        *,
        warm_start_theta: Optional[np.ndarray],
        score_df: Optional[pd.DataFrame] = None,
        fold_idx: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[float, RightCensoredObservedL1MLE]]:
        if self.path_factors_ is None or self.path_anchor_factor_ is None:
            raise RuntimeError("Path factors must be resolved before evaluation")

        anchor_factor, upward_factors, downward_factors = self._build_path_sequence(self.path_factors_)
        rows: list[dict[str, Any]] = []
        estimators: Dict[float, RightCensoredObservedL1MLE] = {}

        def run_one_factor(
            factor: float,
            branch: str,
            step_index: int,
            start_theta: Optional[np.ndarray],
        ) -> Optional[np.ndarray]:
            norm_constraint = float(factor) * self.base_norm_constraint
            try:
                estimator, results, runtime_seconds = self._fit_factor(
                    fit_df,
                    factor,
                    warm_start_theta=start_theta,
                )
                estimators[float(factor)] = estimator
                validation_score = float("nan")
                if score_df is not None:
                    validation_score = float(
                        incomplete_loglik(
                            estimator,
                            score_df,
                            time_col=self.time_col,
                            delta_col=self.delta_col,
                        )
                    )
                rows.append({
                    "fold_idx": int(fold_idx) if fold_idx is not None else -1,
                    "norm_constraint_factor": float(factor),
                    "norm_constraint": float(norm_constraint),
                    "path_branch": str(branch),
                    "path_step": int(step_index),
                    "validation_score": validation_score,
                    "runtime_seconds": float(runtime_seconds),
                    "n_iterations_run": float(results["n_iterations_run"]),
                    "converged": float(results["converged"]),
                    "final_learning_rate": float(results["final_learning_rate"]),
                    "recovery_count": float(results["recovery_count"]),
                })
                return np.asarray(estimator.theta_hat, dtype=float).copy()
            except Exception:
                rows.append({
                    "fold_idx": int(fold_idx) if fold_idx is not None else -1,
                    "norm_constraint_factor": float(factor),
                    "norm_constraint": float(norm_constraint),
                    "path_branch": str(branch),
                    "path_step": int(step_index),
                    "validation_score": float("-inf") if score_df is not None else float("nan"),
                    "runtime_seconds": float("nan"),
                    "n_iterations_run": float("nan"),
                    "converged": float("nan"),
                    "final_learning_rate": float("nan"),
                    "recovery_count": float("nan"),
                })
                return None

        anchor_theta = run_one_factor(
            anchor_factor,
            "anchor",
            0,
            None if warm_start_theta is None else np.asarray(warm_start_theta, dtype=float).copy(),
        )

        prev_theta = None if anchor_theta is None else anchor_theta.copy()
        for step_index, factor in enumerate(upward_factors, start=1):
            next_theta = run_one_factor(factor, "up", step_index, prev_theta)
            if next_theta is not None:
                prev_theta = next_theta

        prev_theta = None if anchor_theta is None else anchor_theta.copy()
        for step_index, factor in enumerate(downward_factors, start=1):
            next_theta = run_one_factor(factor, "down", step_index, prev_theta)
            if next_theta is not None:
                prev_theta = next_theta

        path_df = pd.DataFrame(rows)
        if not path_df.empty:
            branch_order = {"anchor": 0, "up": 1, "down": 2}
            path_df["_branch_order"] = path_df["path_branch"].map(branch_order).astype(int)
            path_df = path_df.sort_values(
                ["path_step", "_branch_order", "norm_constraint_factor"],
                ascending=[True, True, True],
            ).drop(columns="_branch_order").reset_index(drop=True)
        return path_df, estimators

    def optimize(self, n_trials: Optional[int] = None) -> TuningResult:
        self.path_factors_ = self._resolve_path_factors(n_trials)
        self.path_anchor_factor_, _, _ = self._build_path_sequence(self.path_factors_)

        fold_path_frames: list[pd.DataFrame] = []
        for fold_idx, (train_idx, val_idx) in enumerate(self._fold_splits):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)
            warm_start = self._get_fold_warm_start_theta(fold_idx, train_df)
            fold_path_df, _ = self._evaluate_factor_path(
                train_df,
                warm_start_theta=warm_start,
                score_df=val_df,
                fold_idx=fold_idx,
            )
            fold_path_frames.append(fold_path_df)

        cv_fold_df = (
            pd.concat(fold_path_frames, ignore_index=True)
            if fold_path_frames
            else pd.DataFrame()
        )
        cv_path_df = self._summarize_path_records(cv_fold_df)
        best_row = self._select_best_path_row(cv_path_df)
        best_factor = float(best_row["norm_constraint_factor"])
        best_norm_constraint = best_factor * self.base_norm_constraint
        self.best_params = {
            "norm_constraint_factor": best_factor,
            "norm_constraint": float(best_norm_constraint),
        }
        self.best_metric_value = float(best_row["mean_cv_loglik"])

        full_path_df, full_estimators = self._evaluate_factor_path(
            self.data,
            warm_start_theta=self.final_warm_start_theta,
        )
        if best_factor not in full_estimators:
            raise RuntimeError("Failed to recover the selected full-data CVL1MLE estimator")
        estimator = full_estimators[best_factor]

        metadata = {
            "best_metric_value": self.best_metric_value,
            "cv_folds": self.cv_folds,
            "validation_metric": "observed_loglik",
            "working_support_size": int(self.working_grid_points.size),
            "base_norm_constraint": float(self.base_norm_constraint),
            "warm_start_final": bool(self.warm_start_final),
            "warm_start_cv": bool(self.warm_start_cv),
            "n_cv_fold_warm_starts": int(
                sum(theta is not None for theta in self._cv_fold_warm_start_cache.values())
            ),
            "path_anchor_factor": float(self.path_anchor_factor_),
            "requested_norm_constraint_factors": list(self.norm_constraint_factors),
            "resolved_norm_constraint_factors": list(self.path_factors_),
            "n_path_points": int(len(self.path_factors_)),
            "optimize_n_trials_argument": None if n_trials is None else int(n_trials),
            "cv_fold_path": cv_fold_df,
            "cv_path_summary": cv_path_df,
            "full_path": full_path_df,
        }
        return TuningResult(
            estimator=estimator,
            best_params=self.best_params,
            metadata=metadata,
        )
