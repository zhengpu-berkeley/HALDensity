"""Optuna-based hyperparameter tuner for interval-censored density estimation.

Mirrors `RightCensoredOptunaHyperparameterTuner` but uses:
- midpoint initializer for model fitting
- interval likelihood: sum log(F(R) - F(L)) for validation scoring
- same conservative adjustment logic
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from tqdm import tqdm

from haldensity.censoring.core.models import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.interval.midpoint_estimator import IntervalCensoredMidpointEstimator
from haldensity.censoring.interval.em_estimator import IntervalCensoredEMEstimator
from haldensity.censoring.interval.metrics import incomplete_loglik_interval


class IntervalCensoredOptunaHyperparameterTuner:
    """Interval-censor CV tuner supporting midpoint-only and EM estimators.

    Tunable parameters:
    - IntervalCensoredMidpointEstimator: basis_order, norm_constraint
    - IntervalCensoredEMEstimator: basis_order, norm_constraint, m_step_norm_multiplier
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
        use_conservative_adjustment: bool = TUNER_DEFAULTS.use_conservative_adjustment,
        conservative_k_percent: float = TUNER_DEFAULTS.conservative_k_percent,
        conservative_max_steps: int = TUNER_DEFAULTS.conservative_max_steps,
        conservative_step_pct: float = TUNER_DEFAULTS.conservative_step_pct,
        silent: bool = True,
        L_col: str = "L",
        R_col: str = "R",
    ):
        if estimator_name not in (
            "IntervalCensoredMidpointEstimator",
            "IntervalCensoredEMEstimator",
        ):
            raise ValueError(
                "estimator_name must be one of {'IntervalCensoredMidpointEstimator', 'IntervalCensoredEMEstimator'}"
            )
        self.estimator_name = estimator_name
        self.data = data.reset_index(drop=True)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.param_overrides = param_overrides or {}
        self.silent = bool(silent)
        self.L_col = str(L_col)
        self.R_col = str(R_col)

        # Conservative adjustment settings
        self.use_conservative_adjustment = bool(use_conservative_adjustment)
        self.conservative_k_percent = float(conservative_k_percent)
        self.conservative_max_steps = int(conservative_max_steps)
        self.conservative_step_pct = float(conservative_step_pct)

        self.kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None

        self.optuna_params: Optional[dict[str, Any]] = None
        self.conservative_params: Optional[dict[str, Any]] = None
        self.adjustment_results: Optional[list[dict[str, Any]]] = None

        # Fixed defaults (non-tunable)
        self._defaults = {
            "tol": EM_DEFAULTS.tol,
            "m_imputations": EM_DEFAULTS.m_imputations,
            "max_em_iter": EM_DEFAULTS.max_em_iter,
            "em_tol": EM_DEFAULTS.em_tol,
            "init_solver": EM_DEFAULTS.init_solver,
            "m_step_solver": EM_DEFAULTS.m_step_solver,
            "e_step_n_grid": EM_DEFAULTS.e_step_n_grid,
            "solver": TUNER_DEFAULTS.solver,
            "use_secondary_solver": TUNER_DEFAULTS.use_secondary_solver,
        }
        if defaults:
            self._defaults.update(defaults)

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        ovr = self.param_overrides

        basis_order_spec = ovr.get("basis_order", [0, 1, 2])
        if isinstance(basis_order_spec, (list, tuple)):
            basis_order = trial.suggest_categorical("basis_order", list(basis_order_spec))
        else:
            basis_order = basis_order_spec

        norm_spec = ovr.get("norm_constraint", {"low": 1.0, "high": 1000.0, "log": True})
        if isinstance(norm_spec, dict):
            norm_constraint = trial.suggest_float(
                "norm_constraint", norm_spec["low"], norm_spec["high"], log=norm_spec.get("log", True)
            )
        else:
            norm_constraint = norm_spec

        params: dict[str, Any] = {"basis_order": basis_order, "norm_constraint": norm_constraint}

        if self.estimator_name == "IntervalCensoredEMEstimator":
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

    def _evaluate_folds(self, params: dict[str, Any]) -> list[float]:
        scores: list[float] = []
        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)
            try:
                if self.estimator_name == "IntervalCensoredMidpointEstimator":
                    est = IntervalCensoredMidpointEstimator(
                        tol=self._defaults["tol"],
                        norm_constraint=params["norm_constraint"],
                        n_grid_points=self.n_grid_points,
                        basis_order=params["basis_order"],
                        solver=self._defaults["solver"],
                        use_secondary_solver=self._defaults["use_secondary_solver"],
                        include_intercept_in_constraint=False,
                    ).fit(train_df, L_col=self.L_col, R_col=self.R_col)
                else:
                    m_step_norm = params["norm_constraint"] * params["m_step_norm_multiplier"]
                    est = IntervalCensoredEMEstimator(
                        tol=self._defaults["tol"],
                        norm_constraint=params["norm_constraint"],
                        n_grid_points=self.n_grid_points,
                        basis_order=params["basis_order"],
                        m_imputations=self._defaults["m_imputations"],
                        max_em_iter=self._defaults["max_em_iter"],
                        em_tol=self._defaults["em_tol"],
                        init_solver=self._defaults["init_solver"],
                        m_step_solver=self._defaults["m_step_solver"],
                        init_norm_constraint=params["norm_constraint"],
                        m_step_norm_constraint=m_step_norm,
                        e_step_n_grid=self._defaults["e_step_n_grid"],
                        rng_seed=self.random_state,
                        verbose=False,
                        L_col=self.L_col,
                        R_col=self.R_col,
                    ).fit(train_df)

                score = incomplete_loglik_interval(est, val_df, L_col=self.L_col, R_col=self.R_col)
                scores.append(float(score))
            except Exception:
                scores.append(float("-inf"))
        return scores

    def _evaluate(self, params: dict[str, Any]) -> float:
        scores = self._evaluate_folds(params)
        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        return self._evaluate(params)

    def _compute_cv_score_with_sd(
        self,
        norm_constraint: float,
        basis_order: int,
        m_step_norm_multiplier: Optional[float] = None,
    ) -> tuple[float, float, list[float]]:
        params: dict[str, Any] = {"norm_constraint": norm_constraint, "basis_order": basis_order}
        if m_step_norm_multiplier is not None:
            params["m_step_norm_multiplier"] = m_step_norm_multiplier
        fold_scores = self._evaluate_folds(params)
        valid_scores = [s for s in fold_scores if np.isfinite(s)]
        mean_score = float(np.mean(valid_scores)) if valid_scores else float("-inf")
        sd_score = float(np.std(valid_scores, ddof=1)) if len(valid_scores) >= 2 else 0.0
        return mean_score, sd_score, fold_scores

    def _apply_conservative_adjustment(self) -> dict[str, Any]:
        if self.optuna_params is None:
            raise ValueError("Optuna optimization must run first")

        optuna_nc = float(self.optuna_params["norm_constraint"])
        basis_order = int(self.optuna_params["basis_order"])
        m_step_mult = self.optuna_params.get("m_step_norm_multiplier")

        optuna_cv_ll, optuna_cv_sd, _ = self._compute_cv_score_with_sd(optuna_nc, basis_order, m_step_mult)
        threshold = optuna_cv_ll - self.conservative_k_percent * optuna_cv_sd

        step_size = self.conservative_step_pct * optuna_nc
        best_conservative_nc = optuna_nc

        self.adjustment_results = [
            {
                "step": 0,
                "norm_constraint": optuna_nc,
                "cv_ll": optuna_cv_ll,
                "cv_sd": optuna_cv_sd,
                "above_threshold": True,
            }
        ]

        for step in range(1, self.conservative_max_steps + 1):
            candidate_nc = optuna_nc - step * step_size
            if candidate_nc <= 0:
                break
            cand_cv_ll, cand_cv_sd, _ = self._compute_cv_score_with_sd(candidate_nc, basis_order, m_step_mult)
            above_threshold = cand_cv_ll >= threshold
            self.adjustment_results.append(
                {
                    "step": step,
                    "norm_constraint": candidate_nc,
                    "cv_ll": cand_cv_ll,
                    "cv_sd": cand_cv_sd,
                    "above_threshold": above_threshold,
                }
            )
            if above_threshold:
                best_conservative_nc = candidate_nc
            else:
                break

        self.conservative_params = {"norm_constraint": float(best_conservative_nc), "basis_order": int(basis_order)}
        if m_step_mult is not None:
            self.conservative_params["m_step_norm_multiplier"] = float(m_step_mult)
        return self.conservative_params

    def optimize(self, n_trials: int = 50) -> dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

        if not self.silent:
            progress = tqdm(total=n_trials, desc="IntervalCensored Optuna CV", unit="trial")
            best_metric = {"value": float("-inf")}

            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                metric = -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                progress.set_postfix({"LL": f"{metric:.2f}", "best": f"{best_metric['value']:.2f}"})

            try:
                self.study.optimize(
                    self._objective,
                    n_trials=int(n_trials),
                    show_progress_bar=False,
                    callbacks=[update_progress],
                )
            finally:
                progress.close()
        else:
            self.study.optimize(self._objective, n_trials=int(n_trials), show_progress_bar=False)

        self.optuna_params = self.study.best_params.copy()
        self.best_metric_value = -float(self.study.best_value)

        if self.use_conservative_adjustment:
            self._apply_conservative_adjustment()
            self.best_params = dict(self.conservative_params or {})
        else:
            self.best_params = dict(self.optuna_params)

        return {
            "best_params": self.best_params,
            "optuna_params": self.optuna_params,
            "conservative_params": self.conservative_params,
            "best_metric_value": self.best_metric_value,
            "study": self.study,
            "adjustment_results": self.adjustment_results,
        }

    def fit_best_model(self) -> Any:
        if self.best_params is None:
            raise ValueError("No best params found. Run optimize() first.")

        if self.estimator_name == "IntervalCensoredMidpointEstimator":
            return IntervalCensoredMidpointEstimator(
                tol=self._defaults["tol"],
                norm_constraint=float(self.best_params["norm_constraint"]),
                n_grid_points=self.n_grid_points,
                basis_order=int(self.best_params["basis_order"]),
                solver=self._defaults["solver"],
                use_secondary_solver=self._defaults["use_secondary_solver"],
                include_intercept_in_constraint=False,
            ).fit(self.data, L_col=self.L_col, R_col=self.R_col)

        m_step_mult = float(self.best_params.get("m_step_norm_multiplier", 1.0))
        m_step_norm = float(self.best_params["norm_constraint"]) * m_step_mult
        return IntervalCensoredEMEstimator(
            tol=self._defaults["tol"],
            norm_constraint=float(self.best_params["norm_constraint"]),
            n_grid_points=self.n_grid_points,
            basis_order=int(self.best_params["basis_order"]),
            m_imputations=self._defaults["m_imputations"],
            max_em_iter=self._defaults["max_em_iter"],
            em_tol=self._defaults["em_tol"],
            init_solver=self._defaults["init_solver"],
            m_step_solver=self._defaults["m_step_solver"],
            init_norm_constraint=float(self.best_params["norm_constraint"]),
            m_step_norm_constraint=m_step_norm,
            e_step_n_grid=self._defaults["e_step_n_grid"],
            rng_seed=self.random_state,
            verbose=False,
            L_col=self.L_col,
            R_col=self.R_col,
        ).fit(self.data)


