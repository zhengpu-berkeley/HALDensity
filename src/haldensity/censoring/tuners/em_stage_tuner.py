"""Standalone EM stage tuner for pre-fitted initial estimators.

Optimizes m_step_norm_multiplier for EMStage refinement.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from tqdm import tqdm

from haldensity.censoring.core.models import EMStageResult, EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.em_stage import EMStage
from haldensity.censoring.right.metrics import incomplete_loglik
from ._base import IPCWFittingMixin


class EMStageTuner(IPCWFittingMixin):
    """Standalone tuner for EM refinement parameters.

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
        Function to predict censoring survival probability S_C(t).
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

    Examples
    --------
    >>> # Fit initial IPCW estimator
    >>> initial_est = RightCensoredIPCWEstimator(...).fit(...)
    >>> 
    >>> # Tune EM refinement
    >>> tuner = EMStageTuner(data, initial_est, km.predict)
    >>> tuner.optimize(n_trials=20)
    >>> result = tuner.fit_best_model()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_estimator: Any,
        S_c_predict: Callable[[np.ndarray], np.ndarray],
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
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
            "m_imputations": EM_DEFAULTS.m_imputations,
            "max_em_iter": 10,  # Fewer iterations for CV
            "em_tol": EM_DEFAULTS.em_tol,
            "m_step_solver": EM_DEFAULTS.m_step_solver,
            "e_step_n_grid": EM_DEFAULTS.e_step_n_grid,
        }
        if em_defaults:
            self.em_defaults.update(em_defaults)

        # Results storage
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None

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
                ipcw_estimator, km = self.fit_ipcw_estimator(
                    train_df,
                    norm_constraint=self.base_norm_constraint,
                    basis_order=self.basis_order,
                )

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
        """Run Optuna optimization for EM parameters.

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
            progress = tqdm(total=n_trials, desc="EMStage Tuning", unit="trial")
            best_metric = {"value": float("-inf")}

            def update_progress(
                study: optuna.Study, trial: optuna.trial.FrozenTrial
            ) -> None:
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

        return self.best_params

    def fit_best_model(self) -> EMStageResult:
        """Fit final model on full data with best parameters.

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

