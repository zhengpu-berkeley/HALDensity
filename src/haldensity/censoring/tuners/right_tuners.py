"""Right-censored data hyperparameter tuners.

Provides:
- RightCensoredInitTuner: Stage 1 (Init) tuner using Optuna CV
- RightCensoredEMTuner: Stage 2 (EM) tuner with oversmooth or direct no-oversmooth mode
- RightCensoredJointTuner: Convenience wrapper running Init -> EM
"""

from __future__ import annotations

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
