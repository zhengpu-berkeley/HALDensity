"""Joint Optuna-based hyperparameter tuner for censored density estimation.

Supports both IPCW-only and EM estimators with unified CV interface.
Includes conservative 1-SE adjustment to reduce CV oversmoothing.
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

    Conservative Adjustment:
    - By default, applies 1-SE conservative adjustment to reduce CV oversmoothing
    - Finds smallest norm_constraint where CV_LL >= max_CV_LL - k% * SD
    - Disable with `use_conservative_adjustment=False` to use raw Optuna result

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
    use_conservative_adjustment : bool
        Whether to apply conservative 1-SE adjustment after Optuna (default: True).
    conservative_k_percent : float
        Threshold as percentage of SD (default: 0.05 = 5%).
    conservative_max_steps : int
        Maximum steps for conservative search (default: 50).
    conservative_step_pct : float
        Step size as percentage of Optuna best (default: 0.02 = 2%).
    silent : bool
        Whether to suppress progress output.

    Examples
    --------
    >>> # Default: with conservative adjustment
    >>> tuner = RightCensoredOptunaHyperparameterTuner(
    ...     estimator_name="RightCensoredIPCWEstimator",
    ...     data=data,
    ...     cv_folds=5,
    ... )
    >>> result = tuner.optimize(n_trials=50)
    >>> print(result['optuna_params'])       # Raw Optuna result
    >>> print(result['conservative_params']) # Adjusted result (used by default)

    >>> # To disable conservative adjustment:
    >>> tuner = RightCensoredOptunaHyperparameterTuner(
    ...     estimator_name="RightCensoredIPCWEstimator",
    ...     data=data,
    ...     use_conservative_adjustment=False,
    ... )
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

        # Conservative adjustment settings
        self.use_conservative_adjustment = use_conservative_adjustment
        self.conservative_k_percent = conservative_k_percent
        self.conservative_max_steps = conservative_max_steps
        self.conservative_step_pct = conservative_step_pct

        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None

        # Conservative adjustment results
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

    def _evaluate_folds(self, params: dict[str, Any]) -> list[float]:
        """Evaluate parameters across all CV folds. Returns list of fold scores."""
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

        return scores

    def _evaluate(self, params: dict[str, Any]) -> float:
        """Evaluate parameters via cross-validation. Returns negated mean for Optuna."""
        scores = self._evaluate_folds(params)
        mean_score = float(np.mean(scores))
        # Optuna minimizes; we want to maximize log-likelihood
        return -mean_score if np.isfinite(mean_score) else float("inf")

    def _compute_cv_score_with_sd(
        self,
        norm_constraint: float,
        basis_order: int,
        m_step_norm_multiplier: Optional[float] = None,
    ) -> tuple[float, float, list[float]]:
        """Compute CV score and standard deviation for given parameters.

        Returns
        -------
        tuple
            (mean_score, sd_score, fold_scores)
        """
        params = {"norm_constraint": norm_constraint, "basis_order": basis_order}
        if m_step_norm_multiplier is not None:
            params["m_step_norm_multiplier"] = m_step_norm_multiplier

        fold_scores = self._evaluate_folds(params)

        # Filter out failed folds
        valid_scores = [s for s in fold_scores if np.isfinite(s)]
        mean_score = float(np.mean(valid_scores)) if valid_scores else float("-inf")
        sd_score = float(np.std(valid_scores, ddof=1)) if len(valid_scores) >= 2 else 0.0

        return mean_score, sd_score, fold_scores

    def _apply_conservative_adjustment(self) -> dict[str, Any]:
        """Apply conservative 1-SE adjustment to reduce oversmoothing.

        Searches for the smallest norm_constraint that achieves:
        CV_LL >= max_CV_LL - k% * SD

        Returns
        -------
        dict
            Adjusted parameters dict
        """
        if self.optuna_params is None:
            raise ValueError("Optuna optimization must run first")

        optuna_nc = self.optuna_params["norm_constraint"]
        basis_order = self.optuna_params["basis_order"]
        m_step_mult = self.optuna_params.get("m_step_norm_multiplier")

        if not self.silent:
            print(f"\n{'='*60}")
            print(f"Conservative {self.conservative_k_percent*100:.0f}% SD Adjustment")
            print(f"{'='*60}")
            print(f"Optuna best norm_constraint: {optuna_nc:.4f}")

        # Step 1: Compute CV score and SD at Optuna best
        optuna_cv_ll, optuna_cv_sd, _ = self._compute_cv_score_with_sd(
            optuna_nc, basis_order, m_step_mult
        )

        if not self.silent:
            print(f"CV LL at Optuna best: {optuna_cv_ll:.4f} (SD: {optuna_cv_sd:.4f})")

        # Step 2: Compute threshold
        threshold = optuna_cv_ll - self.conservative_k_percent * optuna_cv_sd

        if not self.silent:
            print(
                f"Threshold: {optuna_cv_ll:.4f} - "
                f"{self.conservative_k_percent}*{optuna_cv_sd:.4f} = {threshold:.4f}"
            )

        # Step 3: Forward search for smaller norm_constraint
        step_size = self.conservative_step_pct * optuna_nc
        best_conservative_nc = optuna_nc
        best_conservative_ll = optuna_cv_ll

        self.adjustment_results = [
            {
                "step": 0,
                "norm_constraint": optuna_nc,
                "cv_ll": optuna_cv_ll,
                "cv_sd": optuna_cv_sd,
                "above_threshold": True,
            }
        ]

        if not self.silent:
            print(f"\nSearching smaller norm_constraints (step_size={step_size:.4f})...")

        for step in range(1, self.conservative_max_steps + 1):
            candidate_nc = optuna_nc - step * step_size

            if candidate_nc <= 0:
                if not self.silent:
                    print(f"  Stopping: norm_constraint would be <= 0")
                break

            cand_cv_ll, cand_cv_sd, _ = self._compute_cv_score_with_sd(
                candidate_nc, basis_order, m_step_mult
            )

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
                best_conservative_ll = cand_cv_ll
            else:
                if not self.silent:
                    print(
                        f"  Step {step}: NC={candidate_nc:.4f}, "
                        f"CV_LL={cand_cv_ll:.4f} < threshold"
                    )
                    print(f"  Stopping search.")
                break

        # Build conservative params
        self.conservative_params = {
            "norm_constraint": best_conservative_nc,
            "basis_order": basis_order,
        }
        if m_step_mult is not None:
            self.conservative_params["m_step_norm_multiplier"] = m_step_mult

        if not self.silent:
            reduction = optuna_nc - best_conservative_nc
            reduction_pct = (reduction / optuna_nc * 100) if optuna_nc > 0 else 0
            print(f"\n{'='*60}")
            print(f"Conservative Selection Complete")
            print(f"{'='*60}")
            print(f"Optuna NC:       {optuna_nc:.4f}")
            print(f"Conservative NC: {best_conservative_nc:.4f}")
            print(f"Reduction:       {reduction:.4f} ({reduction_pct:.1f}%)")

        return self.conservative_params

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = self._suggest_params(trial)
        return self._evaluate(params)

    def optimize(self, n_trials: int = 50) -> dict[str, Any]:
        """Run Optuna optimization with optional conservative adjustment.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.

        Returns
        -------
        dict
            Contains:
            - 'best_params': Final parameters (conservative if enabled, else Optuna)
            - 'optuna_params': Raw Optuna best parameters
            - 'conservative_params': Conservative-adjusted parameters (if enabled)
            - 'best_metric_value': CV log-likelihood at Optuna best
            - 'study': Optuna study object
            - 'adjustment_results': List of adjustment search steps (if enabled)
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
                metric = (
                    -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                )
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                progress.set_postfix(
                    {
                        "LL": f"{metric:.2f}",
                        "best": f"{best_metric['value']:.2f}",
                        "order": trial.params.get("basis_order", "?"),
                    }
                )

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

        # Store Optuna results
        self.optuna_params = self.study.best_params.copy()
        self.best_metric_value = -self.study.best_value

        # Apply conservative adjustment if enabled
        if self.use_conservative_adjustment:
            self._apply_conservative_adjustment()
            self.best_params = self.conservative_params
        else:
            self.conservative_params = None
            self.adjustment_results = None
            self.best_params = self.optuna_params

        return {
            "best_params": self.best_params,
            "optuna_params": self.optuna_params,
            "conservative_params": self.conservative_params,
            "best_metric_value": self.best_metric_value,
            "study": self.study,
            "adjustment_results": self.adjustment_results,
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
