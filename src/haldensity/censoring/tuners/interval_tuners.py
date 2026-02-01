"""Interval-censored data hyperparameter tuners.

Provides:
- IntervalCensoredInitTuner: Stage 1 (Init) tuner using Optuna CV
- IntervalCensoredEMTuner: Stage 2 (EM) tuner with oversmooth or CV mode
- IntervalCensoredJointTuner: Convenience wrapper running Init -> EM
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from haldensity.censoring.core.models import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.interval.estimators import (
    IntervalCensoredInitEstimator,
    IntervalCensoredEMStage,
)
from haldensity.censoring.interval.metrics import incomplete_loglik_interval

from ._base import (
    BaseCensoredInitTuner,
    BaseCensoredEMTuner,
    TuningResult,
    OverSmoothInitRecord,
    OverSmoothEMRecord,
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
            tol=self._defaults["tol"],
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=self._defaults["solver"],
            use_secondary_solver=self._defaults["use_secondary_solver"],
            include_intercept_in_constraint=False,
        ).fit(train_df, L_col=self.L_col, R_col=self.R_col)
        
        return incomplete_loglik_interval(est, val_df, L_col=self.L_col, R_col=self.R_col)
    
    def _fit_final_estimator(self, params: Dict[str, Any]) -> IntervalCensoredInitEstimator:
        """Fit final estimator on full data."""
        return IntervalCensoredInitEstimator(
            tol=self._defaults["tol"],
            norm_constraint=params["norm_constraint"],
            n_grid_points=self.n_grid_points,
            basis_order=params["basis_order"],
            solver=self._defaults["solver"],
            use_secondary_solver=self._defaults["use_secondary_solver"],
            include_intercept_in_constraint=False,
        ).fit(self.data, L_col=self.L_col, R_col=self.R_col)


class IntervalCensoredEMTuner(BaseCensoredEMTuner):
    """Stage 2 hyperparameter tuner for interval-censored data.
    
    Supports two modes via `do_over_smooth`:
    - True (default): Grid search over oversmooth factors + EM refinement
    - False: CV-based tuning of m_step_norm_multiplier
    
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
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
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
            cv_folds=cv_folds,
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
    
    def _evaluate_cv_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        m_step_norm_multiplier: float,
    ) -> float:
        """Evaluate m_step_norm_multiplier on a CV fold."""
        # Fit Stage 1 on training data
        init_est = IntervalCensoredInitEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=self.base_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            include_intercept_in_constraint=False,
        ).fit(train_df, L_col=self.L_col, R_col=self.R_col)
        
        # Run EM
        m_step_norm_constraint = self.base_norm_constraint * m_step_norm_multiplier
        
        em_stage = IntervalCensoredEMStage(
            m_imputations=self.em_m_imputations,
            max_em_iter=min(10, self.em_max_em_iter),  # Fewer iterations for CV
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
        
        em_result = em_stage.run(
            initial_estimator=init_est,
            data=train_df,
        )
        
        return incomplete_loglik_interval(em_result.final_estimator, val_df, L_col=self.L_col, R_col=self.R_col)


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
            cv_folds=self.cv_folds,
            em_m_imputations=self.em_m_imputations,
            em_max_em_iter=self.em_max_em_iter,
            silent=self.silent,
            L_col=self.L_col,
            R_col=self.R_col,
        )
        self.stage2_result = em_tuner.optimize()
        
        return (self.stage1_result, self.stage2_result)


# Backward compatibility aliases
IntervalCensoredOptunaHyperparameterTuner = IntervalCensoredInitTuner
IntervalCensoredEMStageOverSmoothTuner = IntervalCensoredEMTuner
