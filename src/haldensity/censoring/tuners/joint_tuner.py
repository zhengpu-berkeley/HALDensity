"""Joint Optuna-based hyperparameter tuner for censored density estimation.

Supports both IPCW-only and EM estimators with unified CV interface.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from tqdm import tqdm

from haldensity.censoring.core.models import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.ipcw_estimator import RightCensoredIPCWEstimator
from haldensity.censoring.right.metrics import incomplete_loglik
from ._base import IPCWFittingMixin, get_estimator_class


class RightCensoredOptunaHyperparameterTuner(IPCWFittingMixin):
    """Censoring-aware CV tuner supporting IPCW and EM estimators.

    Tunable parameters:
    - RightCensoredIPCWEstimator: basis_order, norm_constraint
    - RightCensoredEMEstimator: basis_order, norm_constraint, m_step_norm_multiplier

    Metric: Incomplete-data log-likelihood (sum Delta * log f + (1-Delta) * log S)

    Parameters
    ----------
    estimator_name : str
        Either "RightCensoredEMEstimator" or "RightCensoredIPCWEstimator".
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

    Examples
    --------
    >>> tuner = RightCensoredOptunaHyperparameterTuner(
    ...     estimator_name="RightCensoredEMEstimator",
    ...     data=data,
    ...     cv_folds=5,
    ... )
    >>> result = tuner.optimize(n_trials=50)
    >>> best_model = tuner.fit_best_model()
    """

    def __init__(
        self,
        estimator_name: str,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[dict[str, Any]] = None,
        defaults: Optional[dict[str, Any]] = None,
        silent: bool = True,
    ):
        # Validate estimator name
        get_estimator_class(estimator_name)  # Raises if invalid

        self.estimator_name = estimator_name
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

        # Fixed defaults (non-tunable)
        self._defaults = {
            "tol": EM_DEFAULTS.tol,
            "m_imputations": EM_DEFAULTS.m_imputations,
            "max_em_iter": EM_DEFAULTS.max_em_iter,
            "em_tol": EM_DEFAULTS.em_tol,
            "init_solver": EM_DEFAULTS.init_solver,
            "m_step_solver": EM_DEFAULTS.m_step_solver,
            "e_step_n_grid": EM_DEFAULTS.e_step_n_grid,
            "use_sc_adjustment": EM_DEFAULTS.use_sc_adjustment,
            "solver": TUNER_DEFAULTS.solver,
            "use_secondary_solver": TUNER_DEFAULTS.use_secondary_solver,
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

        # RightCensoredEMEstimator also tunes m_step_norm_multiplier
        if self.estimator_name in ("RightCensoredEMEstimator"):
            mult_spec = ovr.get(
                "m_step_norm_multiplier", {"low": 0.5, "high": 1.0, "log": True}
            )
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
                if self.estimator_name in ("RightCensoredIPCWEstimator"):
                    # IPCW baseline on uncensored only
                    km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
                    T_vals = np.asarray(train_df["T"].values, dtype=float)
                    Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)
                    w = compute_ipcw_weights(
                        T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t))
                    )
                    unc_mask = Delta_vals == 1
                    df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
                    w_unc = w[unc_mask]

                    est = RightCensoredIPCWEstimator(
                        tol=self._defaults["tol"],
                        norm_constraint=params["norm_constraint"],
                        n_grid_points=self.n_grid_points,
                        basis_order=params["basis_order"],
                        solver=self._defaults["solver"],
                        use_secondary_solver=self._defaults["use_secondary_solver"],
                    ).fit(df_unc, sample_weights=w_unc)
                else:
                    # EMIPCWEstimator / RightCensoredEMEstimator
                    from haldensity.censoring.right.em_estimator import (
                        RightCensoredEMEstimator,
                    )

                    m_step_norm = (
                        params["norm_constraint"] * params["m_step_norm_multiplier"]
                    )
                    est = RightCensoredEMEstimator(
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
        """Run Optuna optimization.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.

        Returns
        -------
        dict
            Contains 'best_params', 'best_metric_value', and 'study'.
        """
        # Use seeded sampler for reproducibility
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

        if not self.silent:
            progress = tqdm(total=n_trials, desc="Optuna CV", unit="trial")
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

        return {
            "best_params": self.best_params,
            "best_metric_value": self.best_metric_value,
            "study": self.study,
        }

    def fit_best_model(self) -> Any:
        """Fit final model on full data with best parameters.

        Returns
        -------
        Any
            The fitted estimator.
        """
        if self.best_params is None:
            raise ValueError("Run optimize() first")

        params = self.best_params

        if self.estimator_name in ("RightCensoredIPCWEstimator"):
            km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
            T_vals = np.asarray(self.data["T"].values, dtype=float)
            Delta_vals = np.asarray(self.data["Delta"].values, dtype=int)
            w = compute_ipcw_weights(
                T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t))
            )
            unc_mask = Delta_vals == 1
            df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
            w_unc = w[unc_mask]

            return RightCensoredIPCWEstimator(
                tol=self._defaults["tol"],
                norm_constraint=params["norm_constraint"],
                n_grid_points=self.n_grid_points,
                basis_order=params["basis_order"],
                solver=self._defaults["solver"],
                use_secondary_solver=self._defaults["use_secondary_solver"],
            ).fit(df_unc, sample_weights=w_unc)
        else:
            from haldensity.censoring.right.em_estimator import RightCensoredEMEstimator

            m_step_norm = params["norm_constraint"] * params["m_step_norm_multiplier"]
            return RightCensoredEMEstimator(
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

