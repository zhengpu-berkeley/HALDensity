"""Right-censored data hyperparameter tuners.

Provides:
- RightCensoredInitTuner: Stage 1 (Init) tuner using Optuna CV
- RightCensoredEMTuner: Stage 2 (EM) tuner with oversmooth or CV mode
- RightCensoredJointTuner: Convenience wrapper running Init -> EM
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from haldensity.censoring._defaults import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.right.estimators import (
    RightCensoredInitEstimator,
    RightCensoredEMStage,
)
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.metrics import incomplete_loglik

from ._base import (
    BaseCensoredInitTuner,
    BaseCensoredEMTuner,
    TuningResult,
    OverSmoothInitRecord,
    OverSmoothEMRecord,
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
        # Fit KM on training data
        km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
        
        T_vals = np.asarray(train_df["T"].values, dtype=float)
        Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        est = RightCensoredInitEstimator(
            tol=self._defaults["tol"],
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=self._defaults["solver"],
            use_secondary_solver=self._defaults["use_secondary_solver"],
        ).fit(df_unc, sample_weights=w_unc)
        
        return incomplete_loglik(est, val_df, time_col="T", delta_col="Delta")
    
    def _fit_final_estimator(self, params: Dict[str, Any]) -> RightCensoredInitEstimator:
        """Fit final estimator on full data."""
        km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
        
        T_vals = np.asarray(self.data["T"].values, dtype=float)
        Delta_vals = np.asarray(self.data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        return RightCensoredInitEstimator(
            tol=self._defaults["tol"],
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=self._defaults["solver"],
            use_secondary_solver=self._defaults["use_secondary_solver"],
        ).fit(df_unc, sample_weights=w_unc)


class RightCensoredEMTuner(BaseCensoredEMTuner):
    """Stage 2 hyperparameter tuner for right-censored data.
    
    Supports two modes via `do_over_smooth`:
    - True (default): Grid search over oversmooth factors + EM refinement
    - False: CV-based tuning of m_step_norm_multiplier
    
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
        If True, use oversmooth grid. If False, use CV.
    oversmooth_factors : list[float] | None
        Factors for oversmooth grid.
    cv_folds : int
        CV folds for CV mode.
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
        
        return RightCensoredInitEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
        ).fit(df_unc, sample_weights=w_unc)
    
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
        s_c_predict = lambda t: np.atleast_1d(km.predict(t))
        
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
    
    def _evaluate_cv_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        m_step_norm_multiplier: float,
    ) -> float:
        """Evaluate m_step_norm_multiplier on a CV fold."""
        # Fit Stage 1 on training data
        km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
        
        T_vals = np.asarray(train_df["T"].values, dtype=float)
        Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        init_est = RightCensoredInitEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=self.base_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
        ).fit(df_unc, sample_weights=w_unc)
        
        # Run EM
        m_step_norm_constraint = self.base_norm_constraint * m_step_norm_multiplier
        s_c_predict = lambda t: np.atleast_1d(km.predict(t))
        
        em_stage = RightCensoredEMStage(
            m_imputations=self.em_m_imputations,
            max_em_iter=min(10, self.em_max_em_iter),  # Fewer iterations for CV
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
        
        em_result = em_stage.run(
            initial_estimator=init_est,
            data=train_df,
            S_c_predict=s_c_predict,
        )
        
        return incomplete_loglik(em_result.final_estimator, val_df, time_col="T", delta_col="Delta")


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
            cv_folds=self.cv_folds,
            em_m_imputations=self.em_m_imputations,
            em_max_em_iter=self.em_max_em_iter,
            silent=self.silent,
        )
        self.stage2_result = em_tuner.optimize()
        
        return (self.stage1_result, self.stage2_result)
