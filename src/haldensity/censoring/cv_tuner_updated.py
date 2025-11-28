"""Hyperparameter tuners for censored density estimation.

This module provides:
1. EMStageTuner: Standalone tuner for EM refinement with a pre-fitted initial estimator
2. TwoStageCensoredTuner: Two-stage tuner that separates IPCW and EM tuning

The key insight is that EM is slow, so we want to minimize EM iterations during tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Optional
import optuna
from sklearn.model_selection import KFold
from tqdm import tqdm

from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .em_stage import EMStage, EMStageResult
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .metrics import incomplete_loglik
from typing import Callable


class EMStageTuner:
    """
    Standalone tuner for EM refinement parameters.

    Given a pre-fitted initial estimator, this tuner optimizes the
    `m_step_norm_multiplier` parameter for EMStage refinement.

    This is useful when you already have a fitted IPCW estimator and want
    to find the optimal EM refinement parameters without re-tuning the
    initial estimator parameters.

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
        Override default parameter ranges. Default tunes m_step_norm_multiplier.
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

        # Default parameter ranges
        self.param_ranges = {
            "m_step_norm_multiplier": {"low": 0.5, "high": 10.0, "log": True},
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

        def s_c_predict(t):
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
                # Fit initial estimator on this fold
                ipcw_estimator, km = self._fit_initial_on_fold(train_df)

                def s_c_predict(t):
                    return np.atleast_1d(km.predict(t))

                # Run EMStage
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

                # Evaluate on validation set
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

        # Run EMStage on full data using the original initial estimator
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


class TwoStageCensoredTuner:
    """
    Two-stage hyperparameter tuner for censored density estimation.

    Stage 1: Fast IPCW-only tuning
        - Tunes: norm_constraint, basis_order
        - Uses: WeightedCVXPYEstimator (no EM)
        - Metric: Incomplete-data log-likelihood

    Stage 2: EM refinement tuning
        - Tunes: m_step_norm_multiplier only (log scale 0.5 to 10.0)
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

        # Stage 2 defaults: m_step_norm_multiplier only
        self.stage2_ranges = {
            "m_step_norm_multiplier": {"low": 0.5, "high": 10.0, "log": True},
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

    def _create_s_c_predict(self, km: KaplanMeier):
        """Create S_c prediction wrapper that returns numpy array."""
        def s_c_predict(t):
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
                # If fitting fails, return a very bad score
                scores.append(float("-inf"))

        mean_score = float(np.mean(scores))
        # Optuna minimizes, we want to maximize log-likelihood
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

        # m_step_norm_multiplier
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
                # Fit initial IPCW estimator
                ipcw_estimator, km = self._fit_ipcw_estimator(
                    train_df,
                    norm_constraint=norm_constraint,
                    basis_order=basis_order,
                )

                # Run EMStage
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

                # Evaluate on validation set
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

        # Combine results
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

        # Fit IPCW on full data
        ipcw_estimator, km = self._fit_ipcw_estimator(
            self.data,
            norm_constraint=norm_constraint,
            basis_order=basis_order,
        )

        # Run EMStage on full data
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

