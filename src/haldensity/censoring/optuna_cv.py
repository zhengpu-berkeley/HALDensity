"""Optuna-based hyperparameter tuners for censored density estimation.

This module provides:
1. CensoredOptunaHyperparameterTuner: Joint CV tuner for IPCW or EM estimators
2. TwoStageCensoredTuner: Separates IPCW tuning (fast) from EM tuning (focused)
3. EMStageTuner: Standalone tuner for EM refinement with pre-fitted initial estimator

Tunable Parameters:
- CensoredOptunaHyperparameterTuner:
  - WeightedCVXPYEstimator: basis_order, norm_constraint
  - EMIPCWEstimator: basis_order, norm_constraint, m_step_norm_multiplier (0.5-1.0)
- TwoStageCensoredTuner:
  - Stage 1: norm_constraint, basis_order
  - Stage 2: m_step_norm_multiplier (0.5-1.0)
- EMStageTuner: m_step_norm_multiplier (0.5-1.0)

All other parameters use sensible defaults and can be overridden but are not tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Callable, Optional
import optuna
from sklearn.model_selection import KFold
from tqdm import tqdm

from .em import EMIPCWEstimator, EMStage, EMStageResult
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .metrics import incomplete_loglik


ESTIMATORS = {
    "EMIPCWEstimator": EMIPCWEstimator,
    "WeightedCVXPYEstimator": WeightedCVXPYEstimator,
}


class CensoredOptunaHyperparameterTuner:
    """
    Censoring-aware CV tuner supporting:
      - EMIPCWEstimator: tunes basis_order, norm_constraint, m_step_norm_multiplier
      - WeightedCVXPYEstimator (IPCW baseline): tunes basis_order, norm_constraint
    
    Metric: 'incomplete' (sum Δ log f + (1-Δ) log S)
    
    Parameters
    ----------
    estimator_name : str
        Either "EMIPCWEstimator" or "WeightedCVXPYEstimator".
    data : pd.DataFrame
        DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.
    n_grid_points : int
        Number of grid points for density evaluation.
    param_overrides : dict | None
        Override default tunable ranges or fixed parameters.
    defaults : dict | None
        Override fixed (non-tunable) parameters.
    silent : bool
        Whether to suppress progress output.
    """
    
    def __init__(
        self,
        estimator_name: str,
        data: pd.DataFrame,
        cv_folds: int = 5,
        random_state: int = 42,
        n_grid_points: int = 200,
        param_overrides: Optional[dict[str, Any]] = None,
        defaults: Optional[dict[str, Any]] = None,
        silent: bool = True,
    ):
        if estimator_name not in ESTIMATORS:
            raise ValueError(f"Unsupported estimator '{estimator_name}'. Available: {list(ESTIMATORS.keys())}")
        
        self.estimator_name = estimator_name
        self.estimator_class = ESTIMATORS[estimator_name]
        self.data = data.reset_index(drop=True)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_grid_points = n_grid_points
        self.param_overrides = param_overrides or {}
        self.silent = silent

        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        
        # Fixed defaults (non-tunable, but can be overridden)
        self._defaults = {
            "tol": 1e-4,
            "m_imputations": 20,
            "max_em_iter": 50,
            "em_tol": 1e-3,
            "init_solver": "ECOS",
            "m_step_solver": "ECOS",
            "e_step_n_grid": 1000,
            "use_sc_adjustment": False,
            "solver": "ECOS",
            "use_secondary_solver": True,
        }
        if defaults:
            self._defaults.update(defaults)

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest tunable parameters for the trial."""
        ovr = self.param_overrides
        
        # Tunable: basis_order
        basis_order_spec = ovr.get("basis_order", [0, 1, 2])
        if isinstance(basis_order_spec, (list, tuple)):
            basis_order = trial.suggest_categorical("basis_order", list(basis_order_spec))
        else:
            basis_order = basis_order_spec
        
        # Tunable: norm_constraint
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
        
        params = {
            "basis_order": basis_order,
            "norm_constraint": norm_constraint,
        }
        
        # EMIPCWEstimator also tunes m_step_norm_multiplier
        if self.estimator_name == "EMIPCWEstimator":
            mult_spec = ovr.get("m_step_norm_multiplier", {"low": 0.5, "high": 1.0, "log": True})
            if isinstance(mult_spec, dict):
                m_step_norm_multiplier = trial.suggest_float(
                    "m_step_norm_multiplier",
                    mult_spec["low"],
                    mult_spec["high"],
                    log=mult_spec.get("log", True),
                )
            else:
                m_step_norm_multiplier = mult_spec
            params["m_step_norm_multiplier"] = m_step_norm_multiplier
        
        return params

    def _evaluate(self, params: dict[str, Any]) -> float:
        """Evaluate parameters via cross-validation."""
        scores = []
        
        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)

            try:
                if self.estimator_name == "WeightedCVXPYEstimator":
                    # IPCW baseline on uncensored only
                    km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
                    T_vals = np.asarray(train_df["T"].values, dtype=float)
                    Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)
                    w = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
                    unc_mask = Delta_vals == 1
                    df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
                    w_unc = w[unc_mask]
                    
                    est = WeightedCVXPYEstimator(
                        tol=self._defaults["tol"],
                        norm_constraint=params["norm_constraint"],
                        n_grid_points=self.n_grid_points,
                        basis_order=params["basis_order"],
                        solver=self._defaults["solver"],
                        use_secondary_solver=self._defaults["use_secondary_solver"],
                    ).fit(df_unc, sample_weights=w_unc)
                else:
                    # EMIPCWEstimator
                    m_step_norm = params["norm_constraint"] * params["m_step_norm_multiplier"]
                    est = EMIPCWEstimator(
                        tol=self._defaults["tol"],
                        norm_constraint=params["norm_constraint"],
                        n_grid_points=self.n_grid_points,
                        basis_order=params["basis_order"],
                        m_imputations=self._defaults["m_imputations"],
                        max_em_iter=self._defaults["max_em_iter"],
                        em_tol=self._defaults["em_tol"],
                        use_sc_adjustment=self._defaults["use_sc_adjustment"],
                        init_solver=self._defaults["init_solver"],
                        m_step_solver=self._defaults["m_step_solver"],
                        init_norm_constraint=params["norm_constraint"],
                        m_step_norm_constraint=m_step_norm,
                        e_step_n_grid=self._defaults["e_step_n_grid"],
                        rng_seed=self.random_state,
                        verbose=False,
                    ).fit(train_df)
                
                score = incomplete_loglik(est, val_df, time_col="T", delta_col="Delta")
                scores.append(score)
            except Exception:
                scores.append(float("-inf"))

        mean_score = float(np.mean(scores))
        # Optuna minimizes; we want to maximize log-likelihood
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = self._suggest_params(trial)
        return self._evaluate(params)

    def optimize(self, n_trials: int = 50) -> dict[str, Any]:
        """
        Run Optuna optimization.
        
        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.
            
        Returns
        -------
        dict
            Contains 'best_params', 'best_metric_value', and 'study'.
        """
        self.study = optuna.create_study(direction="minimize")
        
        if not self.silent:
            progress = tqdm(total=n_trials, desc="Optuna CV", unit="trial")
            best_metric = {"value": float("-inf")}

            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                metric = -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                progress.set_postfix({
                    "LL": f"{metric:.2f}",
                    "best": f"{best_metric['value']:.2f}",
                    "order": trial.params.get("basis_order", "?"),
                })

            try:
                self.study.optimize(
                    self._objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                    callbacks=[update_progress],
                )
            finally:
                progress.close()
        else:
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                show_progress_bar=False,
            )
        
        self.best_params = self.study.best_params.copy()
        self.best_metric_value = -self.study.best_value
        
        if not self.silent:
            print(f"\nBest params: {self.best_params}")
            print(f"Best metric: {self.best_metric_value:.4f}")
        
        return {
            "best_params": self.best_params,
            "best_metric_value": self.best_metric_value,
            "study": self.study,
        }

    def fit_best_model(self) -> Any:
        """
        Fit final model on full data with best parameters.
        
        Returns
        -------
        Any
            The fitted estimator.
        """
        if self.best_params is None:
            raise ValueError("Run optimize() first")
        
        params = self.best_params
        
        if self.estimator_name == "WeightedCVXPYEstimator":
            km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
            T_vals = np.asarray(self.data["T"].values, dtype=float)
            Delta_vals = np.asarray(self.data["Delta"].values, dtype=int)
            w = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
            unc_mask = Delta_vals == 1
            df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
            w_unc = w[unc_mask]
            
            return WeightedCVXPYEstimator(
                tol=self._defaults["tol"],
                norm_constraint=params["norm_constraint"],
                n_grid_points=self.n_grid_points,
                basis_order=params["basis_order"],
                solver=self._defaults["solver"],
                use_secondary_solver=self._defaults["use_secondary_solver"],
            ).fit(df_unc, sample_weights=w_unc)
        else:
            m_step_norm = params["norm_constraint"] * params["m_step_norm_multiplier"]
            return EMIPCWEstimator(
                tol=self._defaults["tol"],
                norm_constraint=params["norm_constraint"],
                n_grid_points=self.n_grid_points,
                basis_order=params["basis_order"],
                m_imputations=self._defaults["m_imputations"],
                max_em_iter=self._defaults["max_em_iter"],
                em_tol=self._defaults["em_tol"],
                use_sc_adjustment=self._defaults["use_sc_adjustment"],
                init_solver=self._defaults["init_solver"],
                m_step_solver=self._defaults["m_step_solver"],
                init_norm_constraint=params["norm_constraint"],
                m_step_norm_constraint=m_step_norm,
                e_step_n_grid=self._defaults["e_step_n_grid"],
                rng_seed=self.random_state,
                verbose=not self.silent,
            ).fit(self.data)


class TwoStageCensoredTuner:
    """
    Two-stage hyperparameter tuner for censored density estimation.

    Stage 1: Fast IPCW-only tuning
        - Tunes: norm_constraint, basis_order
        - Uses: WeightedCVXPYEstimator (no EM)
        - Metric: Incomplete-data log-likelihood

    Stage 2: EM refinement tuning
        - Tunes: m_step_norm_multiplier only (log scale 0.5 to 1.0)
        - Fixed: All Stage 1 params + EM defaults
        - Uses: EMStage with IPCW initial
        - Metric: Incomplete-data log-likelihood

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.
    n_grid_points : int
        Number of grid points for density evaluation.
    stage1_param_ranges : dict | None
        Override default Stage 1 parameter ranges.
    stage2_param_ranges : dict | None
        Override default Stage 2 parameter ranges.
    em_defaults : dict | None
        Override default EM parameters for Stage 2.
    silent : bool
        Whether to suppress progress output.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = 5,
        random_state: int = 42,
        n_grid_points: int = 200,
        stage1_param_ranges: Optional[dict[str, Any]] = None,
        stage2_param_ranges: Optional[dict[str, Any]] = None,
        em_defaults: Optional[dict[str, Any]] = None,
        silent: bool = False,
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_grid_points = n_grid_points
        self.silent = silent

        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Stage 1 defaults: norm_constraint, basis_order
        self.stage1_ranges = {
            "norm_constraint": {"low": 1.0, "high": 1000.0, "log": True},
            "basis_order": [0, 1, 2],
        }
        if stage1_param_ranges:
            self.stage1_ranges.update(stage1_param_ranges)

        # Stage 2 defaults: m_step_norm_multiplier only (0.5-1.0)
        self.stage2_ranges = {
            "m_step_norm_multiplier": {"low": 0.5, "high": 1.0, "log": True},
        }
        if stage2_param_ranges:
            self.stage2_ranges.update(stage2_param_ranges)

        # Fixed EM defaults for Stage 2
        self.em_defaults = {
            "m_imputations": 20,
            "max_em_iter": 10,
            "em_tol": 1e-3,
            "m_step_solver": "ECOS",
            "e_step_n_grid": 1000,
        }
        if em_defaults:
            self.em_defaults.update(em_defaults)

        # Results storage
        self.stage1_study: Optional[optuna.Study] = None
        self.stage2_study: Optional[optuna.Study] = None
        self.stage1_best_params: Optional[dict[str, Any]] = None
        self.stage2_best_params: Optional[dict[str, Any]] = None
        self.best_params: Optional[dict[str, Any]] = None

    def _create_s_c_predict(self, km: KaplanMeier) -> Callable[[np.ndarray], np.ndarray]:
        """Create S_c prediction wrapper that returns numpy array."""
        def s_c_predict(t: np.ndarray) -> np.ndarray:
            return np.atleast_1d(km.predict(t))
        return s_c_predict

    def _fit_ipcw_estimator(
        self,
        train_df: pd.DataFrame,
        norm_constraint: float,
        basis_order: int,
    ) -> tuple[WeightedCVXPYEstimator, KaplanMeier]:
        """Fit IPCW estimator on training data."""
        km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
        s_c_predict = self._create_s_c_predict(km)

        T_vals = np.asarray(train_df["T"].values, dtype=float)
        Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)

        weights = compute_ipcw_weights(T_vals, Delta_vals, s_c_predict)
        uncensored_mask = Delta_vals == 1
        ipcw_data = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        ipcw_weights = weights[uncensored_mask]

        estimator = WeightedCVXPYEstimator(
            tol=1e-4,
            norm_constraint=norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=basis_order,
            solver="ECOS",
            use_secondary_solver=True,
        )
        estimator.fit(ipcw_data, sample_weights=ipcw_weights)

        return estimator, km

    # =========================================================================
    # Stage 1: IPCW-only tuning
    # =========================================================================

    def _stage1_suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameters for Stage 1 trial."""
        params = {}

        # norm_constraint
        nc_range = self.stage1_ranges["norm_constraint"]
        if isinstance(nc_range, dict):
            params["norm_constraint"] = trial.suggest_float(
                "norm_constraint",
                nc_range["low"],
                nc_range["high"],
                log=nc_range.get("log", True),
            )
        else:
            params["norm_constraint"] = nc_range

        # basis_order
        bo_range = self.stage1_ranges["basis_order"]
        if isinstance(bo_range, (list, tuple)):
            params["basis_order"] = trial.suggest_categorical("basis_order", list(bo_range))
        else:
            params["basis_order"] = bo_range

        return params

    def _stage1_evaluate(self, params: dict[str, Any]) -> float:
        """Evaluate Stage 1 parameters via cross-validation."""
        scores = []

        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)

            try:
                estimator, _ = self._fit_ipcw_estimator(
                    train_df,
                    norm_constraint=params["norm_constraint"],
                    basis_order=params["basis_order"],
                )
                score = incomplete_loglik(estimator, val_df, time_col="T", delta_col="Delta")
                scores.append(score)
            except Exception:
                scores.append(float("-inf"))

        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def _stage1_objective(self, trial: optuna.Trial) -> float:
        """Optuna objective for Stage 1."""
        params = self._stage1_suggest_params(trial)
        return self._stage1_evaluate(params)

    def run_stage1(self, n_trials: int = 30) -> dict[str, Any]:
        """
        Run Stage 1: IPCW hyperparameter tuning.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.

        Returns
        -------
        dict
            Best parameters from Stage 1.
        """
        self.stage1_study = optuna.create_study(direction="minimize")

        if not self.silent:
            progress = tqdm(total=n_trials, desc="Stage 1 (IPCW)", unit="trial")
            best_metric = {"value": float("-inf")}

            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                metric = -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                progress.set_postfix({
                    "LL": f"{metric:.2f}",
                    "best": f"{best_metric['value']:.2f}",
                    "order": trial.params.get("basis_order", "?"),
                })

            try:
                self.stage1_study.optimize(
                    self._stage1_objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                    callbacks=[update_progress],
                )
            finally:
                progress.close()
        else:
            self.stage1_study.optimize(
                self._stage1_objective,
                n_trials=n_trials,
                show_progress_bar=False,
            )

        self.stage1_best_params = self.stage1_study.best_params.copy()

        if not self.silent:
            print(f"\nStage 1 Best Params: {self.stage1_best_params}")
            print(f"Stage 1 Best LL: {-self.stage1_study.best_value:.4f}")

        return self.stage1_best_params

    # =========================================================================
    # Stage 2: EM refinement tuning
    # =========================================================================

    def _stage2_suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameters for Stage 2 trial."""
        params = {}

        mult_range = self.stage2_ranges["m_step_norm_multiplier"]
        if isinstance(mult_range, dict):
            params["m_step_norm_multiplier"] = trial.suggest_float(
                "m_step_norm_multiplier",
                mult_range["low"],
                mult_range["high"],
                log=mult_range.get("log", True),
            )
        else:
            params["m_step_norm_multiplier"] = mult_range

        return params

    def _stage2_evaluate(
        self,
        stage1_params: dict[str, Any],
        stage2_params: dict[str, Any],
    ) -> float:
        """Evaluate Stage 2 parameters via cross-validation."""
        norm_constraint = stage1_params["norm_constraint"]
        basis_order = stage1_params["basis_order"]
        m_step_norm_multiplier = stage2_params["m_step_norm_multiplier"]
        m_step_norm_constraint = norm_constraint * m_step_norm_multiplier

        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(self.data)):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)

            try:
                ipcw_estimator, km = self._fit_ipcw_estimator(
                    train_df,
                    norm_constraint=norm_constraint,
                    basis_order=basis_order,
                )

                em_stage = EMStage(
                    m_imputations=self.em_defaults["m_imputations"],
                    max_em_iter=self.em_defaults["max_em_iter"],
                    em_tol=self.em_defaults["em_tol"],
                    norm_constraint=m_step_norm_constraint,
                    n_grid_points=self.n_grid_points,
                    m_step_solver=self.em_defaults["m_step_solver"],
                    e_step_n_grid=self.em_defaults["e_step_n_grid"],
                    verbose=False,
                    rng_seed=self.random_state + fold_idx * 1000,
                )

                s_c_predict = self._create_s_c_predict(km)
                em_result = em_stage.run(
                    initial_estimator=ipcw_estimator,
                    data=train_df,
                    S_c_predict=s_c_predict,
                )

                score = incomplete_loglik(
                    em_result.final_estimator,
                    val_df,
                    time_col="T",
                    delta_col="Delta",
                )
                scores.append(score)

            except Exception:
                scores.append(float("-inf"))

        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def _stage2_objective(self, trial: optuna.Trial) -> float:
        """Optuna objective for Stage 2."""
        if self.stage1_best_params is None:
            raise ValueError("Stage 1 must be run before Stage 2")
        stage2_params = self._stage2_suggest_params(trial)
        return self._stage2_evaluate(self.stage1_best_params, stage2_params)

    def run_stage2(
        self,
        stage1_params: Optional[dict[str, Any]] = None,
        n_trials: int = 20,
    ) -> dict[str, Any]:
        """
        Run Stage 2: EM refinement tuning.

        Parameters
        ----------
        stage1_params : dict | None
            Stage 1 parameters. If None, uses stored stage1_best_params.
        n_trials : int
            Number of Optuna trials.

        Returns
        -------
        dict
            Best parameters from Stage 2.
        """
        if stage1_params is not None:
            self.stage1_best_params = stage1_params
        if self.stage1_best_params is None:
            raise ValueError("Stage 1 must be run first or stage1_params provided")

        self.stage2_study = optuna.create_study(direction="minimize")

        if not self.silent:
            progress = tqdm(total=n_trials, desc="Stage 2 (EM)", unit="trial")
            best_metric = {"value": float("-inf")}

            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                metric = -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                mult = trial.params.get("m_step_norm_multiplier", "?")
                mult_str = f"{mult:.2f}" if isinstance(mult, float) else str(mult)
                progress.set_postfix({
                    "LL": f"{metric:.2f}",
                    "best": f"{best_metric['value']:.2f}",
                    "mult": mult_str,
                })

            try:
                self.stage2_study.optimize(
                    self._stage2_objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                    callbacks=[update_progress],
                )
            finally:
                progress.close()
        else:
            self.stage2_study.optimize(
                self._stage2_objective,
                n_trials=n_trials,
                show_progress_bar=False,
            )

        self.stage2_best_params = self.stage2_study.best_params.copy()

        if not self.silent:
            print(f"\nStage 2 Best Params: {self.stage2_best_params}")
            print(f"Stage 2 Best LL: {-self.stage2_study.best_value:.4f}")

        return self.stage2_best_params

    # =========================================================================
    # Combined optimization
    # =========================================================================

    def optimize(
        self,
        n_trials_stage1: int = 30,
        n_trials_stage2: int = 20,
    ) -> dict[str, Any]:
        """
        Run both stages sequentially.

        Parameters
        ----------
        n_trials_stage1 : int
            Number of trials for Stage 1 (IPCW tuning).
        n_trials_stage2 : int
            Number of trials for Stage 2 (EM tuning).

        Returns
        -------
        dict
            Combined best parameters from both stages.
        """
        if not self.silent:
            print("=" * 60)
            print("Two-Stage Hyperparameter Tuning")
            print("=" * 60)

        self.run_stage1(n_trials_stage1)
        self.run_stage2(n_trials=n_trials_stage2)

        self.best_params = {
            **self.stage1_best_params,
            **self.stage2_best_params,
        }

        if not self.silent:
            print("\n" + "=" * 60)
            print("Final Best Parameters:")
            for key, val in self.best_params.items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")
            print("=" * 60)

        return self.best_params

    def fit_best_model(self) -> EMStageResult:
        """
        Fit final model on full data with best parameters from both stages.

        Returns
        -------
        EMStageResult
            Result containing the final refined estimator.
        """
        if self.best_params is None:
            raise ValueError("Run optimize() first")

        norm_constraint = self.best_params["norm_constraint"]
        basis_order = self.best_params["basis_order"]
        m_step_norm_multiplier = self.best_params["m_step_norm_multiplier"]
        m_step_norm_constraint = norm_constraint * m_step_norm_multiplier

        ipcw_estimator, km = self._fit_ipcw_estimator(
            self.data,
            norm_constraint=norm_constraint,
            basis_order=basis_order,
        )

        em_stage = EMStage(
            m_imputations=self.em_defaults["m_imputations"],
            max_em_iter=self.em_defaults["max_em_iter"],
            em_tol=self.em_defaults["em_tol"],
            norm_constraint=m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            m_step_solver=self.em_defaults["m_step_solver"],
            e_step_n_grid=self.em_defaults["e_step_n_grid"],
            verbose=not self.silent,
            rng_seed=self.random_state,
        )

        s_c_predict = self._create_s_c_predict(km)
        em_result = em_stage.run(
            initial_estimator=ipcw_estimator,
            data=self.data,
            S_c_predict=s_c_predict,
        )

        return em_result

    def get_stage1_study(self) -> Optional[optuna.Study]:
        """Return Stage 1 Optuna study."""
        return self.stage1_study

    def get_stage2_study(self) -> Optional[optuna.Study]:
        """Return Stage 2 Optuna study."""
        return self.stage2_study


class EMStageTuner:
    """
    Standalone tuner for EM refinement parameters.

    Given a pre-fitted initial estimator, this tuner optimizes the
    `m_step_norm_multiplier` parameter for EMStage refinement.

    This is useful when you already have a fitted IPCW estimator and want
    to find the optimal EM refinement parameters without re-tuning the
    initial estimator parameters.

    Tunable: m_step_norm_multiplier (0.5 to 1.0, log scale)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
    initial_estimator : Any
        A pre-fitted estimator with theta_hat, _grid_points_hal, basis_order,
        and get_density_at_points() method.
    S_c_predict : Callable
        Function to predict censoring survival probability S_c(t).
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.
    n_grid_points : int
        Number of grid points for density evaluation.
    param_ranges : dict | None
        Override default parameter ranges.
    em_defaults : dict | None
        Override default EM parameters.
    silent : bool
        Whether to suppress progress output.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_estimator: Any,
        S_c_predict: Callable[[np.ndarray], np.ndarray],
        cv_folds: int = 5,
        random_state: int = 42,
        n_grid_points: int = 200,
        param_ranges: Optional[dict[str, Any]] = None,
        em_defaults: Optional[dict[str, Any]] = None,
        silent: bool = False,
    ):
        self.data = data.reset_index(drop=True)
        self.initial_estimator = initial_estimator
        self.S_c_predict = S_c_predict
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_grid_points = n_grid_points
        self.silent = silent

        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Extract base norm_constraint from initial estimator
        self.base_norm_constraint = getattr(
            initial_estimator, "norm_constraint",
            getattr(initial_estimator, "_norm_constraint", 100.0)
        )
        self.basis_order = int(initial_estimator.basis_order)

        # Default parameter ranges (0.5-1.0)
        self.param_ranges = {
            "m_step_norm_multiplier": {"low": 0.5, "high": 1.0, "log": True},
        }
        if param_ranges:
            self.param_ranges.update(param_ranges)

        # Fixed EM defaults
        self.em_defaults = {
            "m_imputations": 20,
            "max_em_iter": 10,
            "em_tol": 1e-3,
            "m_step_solver": "ECOS",
            "e_step_n_grid": 1000,
        }
        if em_defaults:
            self.em_defaults.update(em_defaults)

        # Results storage
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None

    def _fit_initial_on_fold(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[WeightedCVXPYEstimator, KaplanMeier]:
        """Fit IPCW estimator on training fold with same params as initial."""
        km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")

        def s_c_predict(t: np.ndarray) -> np.ndarray:
            return np.atleast_1d(km.predict(t))

        T_vals = np.asarray(train_df["T"].values, dtype=float)
        Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)

        weights = compute_ipcw_weights(T_vals, Delta_vals, s_c_predict)
        uncensored_mask = Delta_vals == 1
        ipcw_data = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        ipcw_weights = weights[uncensored_mask]

        estimator = WeightedCVXPYEstimator(
            tol=1e-4,
            norm_constraint=self.base_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver="ECOS",
            use_secondary_solver=True,
        )
        estimator.fit(ipcw_data, sample_weights=ipcw_weights)

        return estimator, km

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameters for trial."""
        params = {}

        mult_range = self.param_ranges["m_step_norm_multiplier"]
        if isinstance(mult_range, dict):
            params["m_step_norm_multiplier"] = trial.suggest_float(
                "m_step_norm_multiplier",
                mult_range["low"],
                mult_range["high"],
                log=mult_range.get("log", True),
            )
        else:
            params["m_step_norm_multiplier"] = mult_range

        return params

    def _evaluate(self, params: dict[str, Any]) -> float:
        """Evaluate parameters via cross-validation."""
        m_step_norm_multiplier = params["m_step_norm_multiplier"]
        m_step_norm_constraint = self.base_norm_constraint * m_step_norm_multiplier

        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(self.data)):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)

            try:
                ipcw_estimator, km = self._fit_initial_on_fold(train_df)

                def s_c_predict(t: np.ndarray) -> np.ndarray:
                    return np.atleast_1d(km.predict(t))

                em_stage = EMStage(
                    m_imputations=self.em_defaults["m_imputations"],
                    max_em_iter=self.em_defaults["max_em_iter"],
                    em_tol=self.em_defaults["em_tol"],
                    norm_constraint=m_step_norm_constraint,
                    n_grid_points=self.n_grid_points,
                    m_step_solver=self.em_defaults["m_step_solver"],
                    e_step_n_grid=self.em_defaults["e_step_n_grid"],
                    verbose=False,
                    rng_seed=self.random_state + fold_idx * 1000,
                )

                em_result = em_stage.run(
                    initial_estimator=ipcw_estimator,
                    data=train_df,
                    S_c_predict=s_c_predict,
                )

                score = incomplete_loglik(
                    em_result.final_estimator,
                    val_df,
                    time_col="T",
                    delta_col="Delta",
                )
                scores.append(score)

            except Exception:
                scores.append(float("-inf"))

        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = self._suggest_params(trial)
        return self._evaluate(params)

    def optimize(self, n_trials: int = 20) -> dict[str, Any]:
        """
        Run Optuna optimization for EM parameters.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.

        Returns
        -------
        dict
            Best parameters found.
        """
        self.study = optuna.create_study(direction="minimize")

        if not self.silent:
            print("=" * 60)
            print("EMStage Hyperparameter Tuning")
            print(f"Base norm_constraint: {self.base_norm_constraint:.2f}")
            print(f"Basis order: {self.basis_order}")
            print("=" * 60)

            progress = tqdm(total=n_trials, desc="EMStage Tuning", unit="trial")
            best_metric = {"value": float("-inf")}

            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                metric = -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                mult = trial.params.get("m_step_norm_multiplier", "?")
                mult_str = f"{mult:.2f}" if isinstance(mult, float) else str(mult)
                progress.set_postfix({
                    "LL": f"{metric:.2f}",
                    "best": f"{best_metric['value']:.2f}",
                    "mult": mult_str,
                })

            try:
                self.study.optimize(
                    self._objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                    callbacks=[update_progress],
                )
            finally:
                progress.close()
        else:
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                show_progress_bar=False,
            )

        self.best_params = self.study.best_params.copy()
        self.best_metric_value = -self.study.best_value

        if not self.silent:
            print(f"\nBest Parameters: {self.best_params}")
            print(f"Best LL: {self.best_metric_value:.4f}")

        return self.best_params

    def fit_best_model(self) -> EMStageResult:
        """
        Fit final model on full data with best parameters.

        Returns
        -------
        EMStageResult
            Result containing the final refined estimator.
        """
        if self.best_params is None:
            raise ValueError("Run optimize() first")

        m_step_norm_multiplier = self.best_params["m_step_norm_multiplier"]
        m_step_norm_constraint = self.base_norm_constraint * m_step_norm_multiplier

        em_stage = EMStage(
            m_imputations=self.em_defaults["m_imputations"],
            max_em_iter=self.em_defaults["max_em_iter"],
            em_tol=self.em_defaults["em_tol"],
            norm_constraint=m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            m_step_solver=self.em_defaults["m_step_solver"],
            e_step_n_grid=self.em_defaults["e_step_n_grid"],
            verbose=not self.silent,
            rng_seed=self.random_state,
        )

        em_result = em_stage.run(
            initial_estimator=self.initial_estimator,
            data=self.data,
            S_c_predict=self.S_c_predict,
        )

        return em_result

    def get_study(self) -> Optional[optuna.Study]:
        """Return the Optuna study."""
        return self.study

