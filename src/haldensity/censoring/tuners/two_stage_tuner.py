"""Two-stage hyperparameter tuner for censored density estimation.

Stage 1: Fast IPCW-only tuning for norm_constraint and basis_order
Stage 2: EM refinement tuning for m_step_norm_multiplier
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


class TwoStageCensoredTuner(IPCWFittingMixin):
    """Two-stage hyperparameter tuner for censored density estimation.

    Stage 1: Fast IPCW-only tuning
        - Tunes: norm_constraint, basis_order
        - Uses: RightCensoredIPCWEstimator (no EM)
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

    Examples
    --------
    >>> tuner = TwoStageCensoredTuner(data=data, cv_folds=5)
    >>> best_params = tuner.optimize(n_trials_stage1=30, n_trials_stage2=20)
    >>> result = tuner.fit_best_model()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
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
            "m_imputations": EM_DEFAULTS.m_imputations,
            "max_em_iter": 10,  # Fewer iterations for CV
            "em_tol": EM_DEFAULTS.em_tol,
            "m_step_solver": EM_DEFAULTS.m_step_solver,
            "e_step_n_grid": EM_DEFAULTS.e_step_n_grid,
        }
        if em_defaults:
            self.em_defaults.update(em_defaults)

        # Results storage
        self.stage1_study: Optional[optuna.Study] = None
        self.stage2_study: Optional[optuna.Study] = None
        self.stage1_best_params: Optional[dict[str, Any]] = None
        self.stage2_best_params: Optional[dict[str, Any]] = None
        self.best_params: Optional[dict[str, Any]] = None

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
                estimator, _ = self.fit_ipcw_estimator(
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
        """Run Stage 1: IPCW hyperparameter tuning.

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

            def update_progress(
                study: optuna.Study, trial: optuna.trial.FrozenTrial
            ) -> None:
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
                ipcw_estimator, km = self.fit_ipcw_estimator(
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

                s_c_predict = self.create_s_c_predict(km)
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
        """Run Stage 2: EM refinement tuning.

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
        return self.stage2_best_params

    # =========================================================================
    # Combined optimization
    # =========================================================================

    def optimize(
        self,
        n_trials_stage1: int = 30,
        n_trials_stage2: int = 20,
    ) -> dict[str, Any]:
        """Run both stages sequentially.

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
        self.run_stage1(n_trials_stage1)
        self.run_stage2(n_trials=n_trials_stage2)

        self.best_params = {
            **self.stage1_best_params,
            **self.stage2_best_params,
        }

        return self.best_params

    def fit_best_model(self) -> EMStageResult:
        """Fit final model on full data with best parameters from both stages.

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

        ipcw_estimator, km = self.fit_ipcw_estimator(
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

        s_c_predict = self.create_s_c_predict(km)
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

