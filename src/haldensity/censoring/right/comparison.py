"""Experiment-oriented comparison helpers for right-censored survival HAL/TMLE."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.metrics import incomplete_loglik, ipcw_loglik
from haldensity.censoring.tuners._base import TuningResult
from haldensity.censoring.tuners.right_tuners import (
    RightCensoredInitTuner,
    RightCensoredObservedFISTATuner,
    RightCensoredObservedFPGDTuner,
)
from haldensity.targeting.right_censored_survival.learner import (
    right_censored_survival_targeting_M_step,
)


MethodName = str


METHOD_LABELS: dict[MethodName, str] = {
    "M1": "IPCW-HAL tuned by observed-data log-likelihood",
    "M2": "IPCW-HAL tuned by IPCW validation loss",
    "M3": "Observed-data HAL-MLE via FISTA",
    "M4": "Observed-data HAL-MLE via FPGD",
}


@dataclass
class RightCensoredTruth:
    """Truth functions used for simulation evaluation."""

    density_fn: Callable[[np.ndarray], np.ndarray]
    survival_fn: Callable[[np.ndarray], np.ndarray]
    event_sampler: Optional[Callable[[np.random.Generator, int], np.ndarray]] = None
    name: str = "truth"


@dataclass
class RightCensoredMethodFit:
    """Fitted initial estimator plus tuning/runtime metadata."""

    method: MethodName
    label: str
    estimator: Any
    km: KaplanMeier
    tuning_result: TuningResult
    fit_runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    targeted_fit: Optional[dict[str, Any]] = None


def simulate_beta_uniform_right_censored(
    n: int,
    *,
    seed: int = 0,
    event_beta_shape: tuple[float, float] = (2.2, 2.5),
    censor_low: float = 0.05,
    censor_high: float = 0.95,
) -> dict[str, Any]:
    """Simulate a simple beta-event / uniform-censoring right-censored dataset."""
    rng = np.random.default_rng(seed)
    a, b = event_beta_shape
    event_times = rng.beta(a, b, size=int(n))
    censor_times = rng.uniform(float(censor_low), float(censor_high), size=int(n))
    observed_t = np.minimum(event_times, censor_times)
    delta = (event_times <= censor_times).astype(int)
    observed_data = pd.DataFrame({"T": observed_t, "Delta": delta})

    truth = RightCensoredTruth(
        density_fn=lambda x: beta_dist.pdf(np.asarray(x, dtype=float), a, b),
        survival_fn=lambda x: beta_dist.sf(np.asarray(x, dtype=float), a, b),
        event_sampler=lambda sampler_rng, size: sampler_rng.beta(a, b, size=int(size)),
        name=f"Beta({a}, {b}) event time",
    )
    return {
        "observed_data": observed_data,
        "event_times": event_times,
        "censor_times": censor_times,
        "truth": truth,
    }


def evaluate_density_on_grid(estimator: Any, grid: np.ndarray) -> np.ndarray:
    """Evaluate the fitted density on an arbitrary grid."""
    pts = np.asarray(grid, dtype=float).ravel()
    return np.asarray(estimator.get_density_at_points(pts), dtype=float)


def evaluate_survival_on_grid(estimator: Any, grid: np.ndarray) -> np.ndarray:
    """Evaluate the fitted survival function on an arbitrary grid."""
    grid_midpoints, density_grid = estimator.get_density()
    delta_j = np.asarray(estimator.delta_j, dtype=float)
    survival_grid = np.cumsum((density_grid * delta_j)[::-1])[::-1]
    edge_survival = np.concatenate((survival_grid, [0.0]))
    grid_edges = np.asarray(estimator.grid_points, dtype=float)
    pts = np.asarray(grid, dtype=float).ravel()
    return np.interp(pts, grid_edges, edge_survival, left=1.0, right=0.0)


def integrated_density_error(
    estimator: Any,
    truth_density_fn: Callable[[np.ndarray], np.ndarray],
    *,
    grid: Optional[np.ndarray] = None,
) -> float:
    """Approximate integrated absolute density error on [0, 1]."""
    eval_grid = np.linspace(0.0, 1.0, 1001) if grid is None else np.asarray(grid, dtype=float)
    mid = 0.5 * (eval_grid[:-1] + eval_grid[1:])
    delta = np.diff(eval_grid)
    est_density = evaluate_density_on_grid(estimator, mid)
    truth_density = np.asarray(truth_density_fn(mid), dtype=float)
    return float(np.sum(np.abs(est_density - truth_density) * delta))


def integrated_survival_error(
    estimator: Any,
    truth_survival_fn: Callable[[np.ndarray], np.ndarray],
    *,
    grid: Optional[np.ndarray] = None,
) -> float:
    """Approximate integrated absolute survival error on [0, 1]."""
    eval_grid = np.linspace(0.0, 1.0, 1001) if grid is None else np.asarray(grid, dtype=float)
    est_survival = evaluate_survival_on_grid(estimator, eval_grid)
    truth_survival = np.asarray(truth_survival_fn(eval_grid), dtype=float)
    delta = np.diff(eval_grid, prepend=eval_grid[0])
    return float(np.sum(np.abs(est_survival - truth_survival) * delta))


def pointwise_survival_error(
    estimator: Any,
    target_time: float,
    truth_survival_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Absolute survival error at a single target time."""
    target = np.asarray([target_time], dtype=float)
    est_val = float(evaluate_survival_on_grid(estimator, target)[0])
    truth_val = float(np.asarray(truth_survival_fn(target), dtype=float)[0])
    return abs(est_val - truth_val)


def extract_optimization_diagnostics(estimator: Any) -> dict[str, Any]:
    """Collect common runtime / convergence diagnostics across methods."""
    results = estimator.get_results()
    diagnostics = {
        "n_selected_knots": int(results.get("n_selected_knots", 0)),
        "n_iterations_run": results.get("n_iterations_run"),
        "converged": results.get("converged"),
        "final_objective_value": None,
        "final_step": results.get("final_step"),
        "final_learning_rate": results.get("final_learning_rate"),
        "optimization_history": results.get("optimization_history"),
    }
    history = diagnostics["optimization_history"]
    if isinstance(history, list) and history:
        final_history = history[-1]
        diagnostics["final_objective_value"] = -float(final_history.get("log_likelihood", np.nan))
    return diagnostics


def fit_right_censored_initial_estimator(
    method: MethodName,
    observed_data: pd.DataFrame,
    *,
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 0,
    n_grid_points: int = 200,
    param_overrides: Optional[dict[str, Any]] = None,
    init_tuner_kwargs: Optional[dict[str, Any]] = None,
    fista_kwargs: Optional[dict[str, Any]] = None,
    fpgd_kwargs: Optional[dict[str, Any]] = None,
) -> RightCensoredMethodFit:
    """Fit one of the four right-censored initial estimators."""
    method_key = str(method).strip().upper()
    if method_key not in METHOD_LABELS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHOD_LABELS)}")

    overrides = dict(param_overrides or {})
    init_kwargs = dict(init_tuner_kwargs or {})
    tic = time.perf_counter()

    if method_key == "M1":
        tuner = RightCensoredInitTuner(
            data=observed_data,
            cv_folds=cv_folds,
            random_state=random_state,
            n_grid_points=n_grid_points,
            param_overrides=overrides,
            validation_metric="observed_loglik",
            **init_kwargs,
        )
    elif method_key == "M2":
        tuner = RightCensoredInitTuner(
            data=observed_data,
            cv_folds=cv_folds,
            random_state=random_state,
            n_grid_points=n_grid_points,
            param_overrides=overrides,
            validation_metric="ipcw_loglik",
            **init_kwargs,
        )
    elif method_key == "M3":
        tuner = RightCensoredObservedFISTATuner(
            data=observed_data,
            cv_folds=cv_folds,
            random_state=random_state,
            n_grid_points=n_grid_points,
            param_overrides=overrides,
            fista_kwargs=fista_kwargs,
            silent=bool(init_kwargs.get("silent", True)),
        )
    else:
        tuner = RightCensoredObservedFPGDTuner(
            data=observed_data,
            cv_folds=cv_folds,
            random_state=random_state,
            n_grid_points=n_grid_points,
            param_overrides=overrides,
            fpgd_kwargs=fpgd_kwargs,
            silent=bool(init_kwargs.get("silent", True)),
        )

    tuning_result = tuner.optimize(n_trials=n_trials)
    fit_runtime_seconds = time.perf_counter() - tic
    km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")
    estimator = tuning_result.estimator
    metadata = {
        "best_params": dict(tuning_result.best_params),
        "best_metric_value": tuning_result.metadata.get("best_metric_value"),
        "validation_metric": tuning_result.metadata.get("validation_metric"),
    }
    metadata.update(extract_optimization_diagnostics(estimator))

    return RightCensoredMethodFit(
        method=method_key,
        label=METHOD_LABELS[method_key],
        estimator=estimator,
        km=km,
        tuning_result=tuning_result,
        fit_runtime_seconds=float(fit_runtime_seconds),
        metadata=metadata,
    )


def evaluate_right_censored_initial_fit(
    fit: RightCensoredMethodFit,
    evaluation_data: pd.DataFrame,
    *,
    truth: Optional[RightCensoredTruth] = None,
    target_times: Optional[Sequence[float]] = None,
    integration_grid: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute shared initial-estimator metrics on one evaluation dataset."""
    target_arr = None if target_times is None else np.asarray(target_times, dtype=float).ravel()
    row = {
        "method": fit.method,
        "label": fit.label,
        "observed_loglik": incomplete_loglik(fit.estimator, evaluation_data, time_col="T", delta_col="Delta"),
        "ipcw_loglik": ipcw_loglik(
            fit.estimator,
            evaluation_data,
            time_col="T",
            delta_col="Delta",
            km=fit.km,
        ),
        "runtime_seconds": float(fit.fit_runtime_seconds),
        "n_selected_knots": fit.metadata.get("n_selected_knots"),
        "n_iterations_run": fit.metadata.get("n_iterations_run"),
        "converged": fit.metadata.get("converged"),
        "final_objective_value": fit.metadata.get("final_objective_value"),
        "final_step": fit.metadata.get("final_step"),
        "final_learning_rate": fit.metadata.get("final_learning_rate"),
        "best_params": fit.metadata.get("best_params"),
        "validation_metric": fit.metadata.get("validation_metric"),
    }
    if truth is not None:
        row["integrated_density_error"] = integrated_density_error(
            fit.estimator,
            truth.density_fn,
            grid=integration_grid,
        )
        row["integrated_survival_error"] = integrated_survival_error(
            fit.estimator,
            truth.survival_fn,
            grid=integration_grid,
        )
        if target_arr is not None and target_arr.size > 0:
            point_errors = [
                pointwise_survival_error(fit.estimator, float(t0), truth.survival_fn)
                for t0 in target_arr
            ]
            row["mean_pointwise_survival_error"] = float(np.mean(point_errors))
            row["max_pointwise_survival_error"] = float(np.max(point_errors))
    return row


def evaluate_right_censored_initial_pointwise(
    fit: RightCensoredMethodFit,
    target_times: Sequence[float],
    *,
    truth: Optional[RightCensoredTruth] = None,
) -> pd.DataFrame:
    """Return one row per target time for the initial survival estimates."""
    rows = []
    for t0 in np.asarray(target_times, dtype=float).ravel():
        est_survival = float(evaluate_survival_on_grid(fit.estimator, np.asarray([t0], dtype=float))[0])
        row = {
            "method": fit.method,
            "label": fit.label,
            "t0": float(t0),
            "psi_init": est_survival,
        }
        if truth is not None:
            truth_survival = float(np.asarray(truth.survival_fn(np.asarray([t0], dtype=float)), dtype=float)[0])
            row["truth_survival"] = truth_survival
            row["abs_error"] = abs(est_survival - truth_survival)
            row["squared_error"] = (est_survival - truth_survival) ** 2
        rows.append(row)
    return pd.DataFrame(rows)


def run_right_censored_initial_estimator_experiment(
    observed_data: pd.DataFrame,
    *,
    test_data: Optional[pd.DataFrame] = None,
    target_times: Optional[Sequence[float]] = None,
    truth: Optional[RightCensoredTruth] = None,
    methods: Sequence[MethodName] = ("M1", "M2", "M3", "M4"),
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 0,
    n_grid_points: int = 200,
    method_param_overrides: Optional[dict[str, dict[str, Any]]] = None,
    init_tuner_kwargs: Optional[dict[str, Any]] = None,
    fista_kwargs: Optional[dict[str, Any]] = None,
    fpgd_kwargs: Optional[dict[str, Any]] = None,
    run_tmle: bool = True,
    tmle_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Fit M1-M4 on one dataset and optionally run pointwise TMLE for each."""
    evaluation_data = observed_data if test_data is None else test_data
    overrides_by_method = method_param_overrides or {}
    fitted_methods: dict[str, RightCensoredMethodFit] = {}

    initial_rows: list[dict[str, Any]] = []
    initial_pointwise_tables: list[pd.DataFrame] = []
    tmle_rows: list[pd.DataFrame] = []
    tmle_results: dict[str, dict[str, Any]] = {}

    for method in methods:
        method_key = str(method).strip().upper()
        fit = fit_right_censored_initial_estimator(
            method_key,
            observed_data,
            cv_folds=cv_folds,
            n_trials=n_trials,
            random_state=random_state,
            n_grid_points=n_grid_points,
            param_overrides=overrides_by_method.get(method_key),
            init_tuner_kwargs=init_tuner_kwargs,
            fista_kwargs=fista_kwargs,
            fpgd_kwargs=fpgd_kwargs,
        )
        fitted_methods[method_key] = fit
        initial_rows.append(
            evaluate_right_censored_initial_fit(
                fit,
                evaluation_data,
                truth=truth,
                target_times=target_times,
            )
        )
        if target_times is not None:
            initial_pointwise_tables.append(
                evaluate_right_censored_initial_pointwise(fit, target_times, truth=truth)
            )

        if run_tmle and target_times is not None:
            targeted_fit = right_censored_survival_targeting_M_step(
                initial_estimator=fit.estimator,
                observed_data=observed_data,
                targeting_points=np.asarray(target_times, dtype=float),
                km=fit.km,
                **(tmle_kwargs or {}),
            )
            fit.targeted_fit = targeted_fit
            tmle_results[method_key] = targeted_fit
            tmle_summary = targeted_fit["summary"].copy()
            tmle_summary["method"] = fit.method
            tmle_summary["label"] = fit.label
            if truth is not None:
                truth_vals = np.asarray(truth.survival_fn(tmle_summary["t0"].to_numpy(dtype=float)), dtype=float)
                tmle_summary["truth_survival"] = truth_vals
                tmle_summary["plugin_bias"] = tmle_summary["psi_init"] - truth_vals
                tmle_summary["targeted_bias"] = tmle_summary["psi_star"] - truth_vals
                tmle_summary["plugin_squared_error"] = np.square(tmle_summary["plugin_bias"])
                tmle_summary["targeted_squared_error"] = np.square(tmle_summary["targeted_bias"])
                tmle_summary["plugin_abs_error"] = np.abs(tmle_summary["plugin_bias"])
                tmle_summary["targeted_abs_error"] = np.abs(tmle_summary["targeted_bias"])
            tmle_summary["epsilon_abs"] = np.abs(tmle_summary["epsilon"])
            tmle_rows.append(tmle_summary)

    return {
        "fitted_methods": fitted_methods,
        "initial_summary": pd.DataFrame(initial_rows),
        "initial_pointwise_summary": (
            pd.concat(initial_pointwise_tables, ignore_index=True)
            if initial_pointwise_tables
            else pd.DataFrame()
        ),
        "tmle_summary": pd.concat(tmle_rows, ignore_index=True) if tmle_rows else pd.DataFrame(),
        "tmle_results": tmle_results,
        "metadata": {
            "methods": [str(m).strip().upper() for m in methods],
            "n_observations": int(observed_data.shape[0]),
            "n_eval_observations": int(evaluation_data.shape[0]),
            "target_times": None if target_times is None else list(np.asarray(target_times, dtype=float)),
        },
    }
