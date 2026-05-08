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
from haldensity.censoring.right.observed_mle import RightCensoredObservedL1MLE
from haldensity.censoring.tuners._base import TuningResult
from haldensity.censoring.tuners.right_tuners import (
    RightCensoredInitTuner,
    RightCensoredObservedFISTATuner,
    RightCensoredObservedFPGDTuner,
)
from haldensity.targeting.right_censored_survival.learner import (
    right_censored_survival_targeting_M_step,
    right_censored_survival_targeting_M_step_v2,
)


MethodName = str


METHOD_LABELS: dict[MethodName, str] = {
    "M1": "IPCW-HAL tuned by observed-data log-likelihood",
    "M2": "IPCW-HAL tuned by IPCW validation loss",
    "M3": "Observed-data HAL-MLE via FISTA",
    "M4": "Observed-data HAL-MLE via FPGD",
}

DEFAULT_CV_IPCW_L1MLE_FACTORS: tuple[float, ...] = tuple(
    float(x) for x in np.round(np.linspace(1.0, 2.0, 11), 2)
)
DEFAULT_TARGETING_KWARGS: dict[str, Any] = {
    "mode": "iterative",
    "clip": 1e-6,
    "survival_clip": 1e-8,
    "max_iter": 25,
    "min_abs_eps": 1e-10,
    "min_score_tol": 1e-8,
    "store_pointwise_arrays": False,
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
        "conservative_selection_rule": tuning_result.metadata.get("conservative_selection_rule"),
        "conservative_se_multiplier": tuning_result.metadata.get("conservative_se_multiplier"),
        "conservative_params": tuning_result.metadata.get("conservative_params"),
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


def _extract_truth_functions(
    truth: Optional[RightCensoredTruth | dict[str, Any]],
) -> tuple[Optional[Callable[[np.ndarray], np.ndarray]], Optional[Callable[[np.ndarray], np.ndarray]]]:
    if truth is None:
        return None, None
    if isinstance(truth, RightCensoredTruth):
        return truth.survival_fn, truth.density_fn
    if isinstance(truth, dict):
        surv = truth.get("survival_fn")
        dens = truth.get("density_fn")
        if callable(surv) and callable(dens):
            return surv, dens
    raise TypeError(
        "truth must be None, RightCensoredTruth, or a dict with callable keys "
        "{'survival_fn', 'density_fn'}"
    )


def _normalize_norm_factors(
    values: Optional[Sequence[float]],
) -> list[float]:
    factors = list(DEFAULT_CV_IPCW_L1MLE_FACTORS) if values is None else [float(x) for x in values]
    if len(factors) == 0:
        raise ValueError("norm_constraint_factors must contain at least one value")
    if any(x <= 0.0 for x in factors):
        raise ValueError("norm_constraint_factors must contain positive values")
    return sorted({float(x) for x in factors})


def _compress_theta_to_support(estimator: Any, support: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=float)
    all_knots = np.asarray(estimator._grid_points_hal, dtype=float)
    theta_full = np.asarray(estimator.theta_hat, dtype=float)
    basis_order = int(estimator.basis_order)
    poly_cols = basis_order if basis_order > 0 else 0
    knot_start = 1 + poly_cols

    theta_selected = np.zeros(knot_start + support.size, dtype=float)
    theta_selected[:knot_start] = theta_full[:knot_start]
    for i, knot in enumerate(support):
        idx = np.where(np.isclose(all_knots, knot, atol=1e-10, rtol=0.0))[0]
        if idx.size == 0:
            raise RuntimeError("Selected support knot was not found in the fitted working grid.")
        theta_selected[knot_start + i] = theta_full[knot_start + int(idx[0])]
    return theta_selected


def _compress_theta_to_selected_support(estimator: Any) -> np.ndarray:
    return _compress_theta_to_support(
        estimator,
        np.asarray(estimator.grid_points_hal_selected, dtype=float),
    )


def _fit_stage1_cv_ipcw(
    observed_data: pd.DataFrame,
    *,
    cv_folds: int,
    n_trials: int,
    random_state: int,
    n_grid_points: int,
    param_overrides: Optional[dict[str, Any]],
    stage1_tuner_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    kwargs = dict(stage1_tuner_kwargs or {})
    kwargs.setdefault("silent", True)
    kwargs.setdefault("use_conservative_adjustment", False)

    tic = time.perf_counter()
    tuner = RightCensoredInitTuner(
        data=observed_data,
        cv_folds=int(cv_folds),
        random_state=int(random_state),
        n_grid_points=int(n_grid_points),
        param_overrides=param_overrides,
        validation_metric="ipcw_loglik",
        **kwargs,
    )
    tuning_result = tuner.optimize(n_trials=int(n_trials))
    runtime_seconds = float(time.perf_counter() - tic)
    stage1_estimator = tuning_result.estimator
    stage1_km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")
    return {
        "tuning_result": tuning_result,
        "estimator": stage1_estimator,
        "km": stage1_km,
        "runtime_seconds": runtime_seconds,
    }


def _fit_cv_ipcw_l1mle_path(
    observed_data: pd.DataFrame,
    stage1_estimator: Any,
    *,
    norm_constraint_factors: Optional[Sequence[float]],
    l1_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    support = np.asarray(stage1_estimator.grid_points_hal_selected, dtype=float)
    base_norm = float(stage1_estimator.norm_constraint)
    factor_grid = np.asarray(_normalize_norm_factors(norm_constraint_factors), dtype=float)
    anchor_idx = int(np.argmin(np.abs(factor_grid - 1.0)))
    fit_order = (
        [anchor_idx]
        + list(range(anchor_idx + 1, len(factor_grid)))
        + list(range(anchor_idx - 1, -1, -1))
    )

    kwargs = dict(l1_kwargs or {})
    forbidden = {"working_grid_points", "norm_constraint", "basis_order", "warm_start_theta"}
    overlap = forbidden.intersection(kwargs)
    if overlap:
        bad = ", ".join(sorted(overlap))
        raise ValueError(
            "l1_kwargs must not override fixed path arguments; remove keys: "
            f"{bad}"
        )

    path_runs: dict[float, dict[str, Any]] = {}
    warm_start = _compress_theta_to_selected_support(stage1_estimator)
    for idx in fit_order:
        factor = float(factor_grid[idx])
        norm_constraint = float(base_norm * factor)
        tic = time.perf_counter()
        estimator = RightCensoredObservedL1MLE(
            working_grid_points=support,
            norm_constraint=norm_constraint,
            basis_order=int(stage1_estimator.basis_order),
            warm_start_theta=warm_start,
            **kwargs,
        ).fit(observed_data, time_col="T", delta_col="Delta")
        runtime_seconds = float(time.perf_counter() - tic)
        results = estimator.get_results()
        path_runs[factor] = {
            "factor": factor,
            "norm_constraint": norm_constraint,
            "estimator": estimator,
            "runtime_seconds": runtime_seconds,
            "observed_loglik": float(incomplete_loglik(estimator, observed_data)),
            "n_iterations_run": int(results.get("n_iterations_run", 0)),
            "converged": bool(results.get("converged", False)),
            "support_size": int(len(results.get("working_grid_points", support))),
        }
        warm_start = np.asarray(estimator.theta_hat, dtype=float).copy()

    path_df = (
        pd.DataFrame(
            [
                {
                    "norm_factor": float(factor),
                    **{k: v for k, v in run.items() if k not in {"factor", "estimator"}},
                }
                for factor, run in sorted(path_runs.items())
            ]
        )
        .sort_values("norm_factor")
        .reset_index(drop=True)
    )
    return {
        "support": support,
        "base_norm_constraint": float(base_norm),
        "runtime_seconds": float(path_df["runtime_seconds"].sum()),
        "path_runs": {factor: path_runs[factor] for factor in sorted(path_runs)},
        "path_df": path_df,
    }


def _build_plugin_summary_for_estimator(
    estimator: Any,
    evaluation_points: np.ndarray,
    *,
    survival_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    density_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> pd.DataFrame:
    t0 = np.asarray(evaluation_points, dtype=float).ravel()
    summary = pd.DataFrame({"t0": t0})
    summary["psi_init"] = np.asarray(evaluate_survival_on_grid(estimator, t0), dtype=float)
    summary["estimated_density"] = np.asarray(estimator.get_density_at_points(t0), dtype=float)
    if survival_fn is not None:
        truth_survival = np.asarray(survival_fn(t0), dtype=float)
        summary["truth_survival"] = truth_survival
        summary["abs_error_plugin"] = np.abs(summary["psi_init"].to_numpy(dtype=float) - truth_survival)
        summary["sq_error_plugin"] = np.square(summary["psi_init"].to_numpy(dtype=float) - truth_survival)
    if density_fn is not None:
        truth_density = np.asarray(density_fn(t0), dtype=float)
        summary["truth_density"] = truth_density
        summary["density_abs_error"] = np.abs(summary["estimated_density"].to_numpy(dtype=float) - truth_density)
        summary["density_sq_error"] = np.square(summary["estimated_density"].to_numpy(dtype=float) - truth_density)
    return summary.sort_values("t0").reset_index(drop=True)


def fit_right_censored_cv_ipcw_l1mle_plugin(
    observed_data: pd.DataFrame,
    *,
    targeting_points: Sequence[float],
    truth: Optional[RightCensoredTruth | dict[str, Any]] = None,
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 0,
    n_grid_points: int = 200,
    stage1_param_overrides: Optional[dict[str, Any]] = None,
    stage1_tuner_kwargs: Optional[dict[str, Any]] = None,
    norm_constraint_factors: Optional[Sequence[float]] = None,
    l1_kwargs: Optional[dict[str, Any]] = None,
    no_undersmoothing_factor: float = 1.0,
) -> dict[str, Any]:
    """Fit CV-IPCW-L1MLE and return the no-undersmoothing plug-in estimator.

    This API computes the post-IPCW fixed-support L1 path and returns the
    plug-in survival/density summary at the factor closest to
    ``no_undersmoothing_factor`` (default ``1.0``).
    """
    survival_fn, density_fn = _extract_truth_functions(truth)
    target_arr = np.asarray(targeting_points, dtype=float).ravel()
    if target_arr.size == 0:
        raise ValueError("targeting_points must contain at least one time point")

    stage1_fit = _fit_stage1_cv_ipcw(
        observed_data,
        cv_folds=cv_folds,
        n_trials=n_trials,
        random_state=random_state,
        n_grid_points=n_grid_points,
        param_overrides=stage1_param_overrides,
        stage1_tuner_kwargs=stage1_tuner_kwargs,
    )
    stage1_estimator = stage1_fit["estimator"]
    path_fit = _fit_cv_ipcw_l1mle_path(
        observed_data,
        stage1_estimator,
        norm_constraint_factors=norm_constraint_factors,
        l1_kwargs=l1_kwargs,
    )

    no_us_factor = min(path_fit["path_runs"], key=lambda f: abs(float(f) - float(no_undersmoothing_factor)))
    no_us_run = path_fit["path_runs"][float(no_us_factor)]
    plugin_summary = _build_plugin_summary_for_estimator(
        no_us_run["estimator"],
        target_arr,
        survival_fn=survival_fn,
        density_fn=density_fn,
    )
    plugin_summary["norm_factor"] = float(no_us_factor)
    plugin_summary["norm_constraint"] = float(no_us_run["norm_constraint"])

    metadata = {
        "api": "fit_right_censored_cv_ipcw_l1mle_plugin",
        "validation_metric": "ipcw_loglik",
        "targeting_points": [float(x) for x in target_arr],
        "requested_no_undersmoothing_factor": float(no_undersmoothing_factor),
        "selected_no_undersmoothing_factor": float(no_us_factor),
        "stage1_best_params": dict(stage1_fit["tuning_result"].best_params),
        "stage1_metadata": dict(stage1_fit["tuning_result"].metadata or {}),
        "path_factors": [float(x) for x in sorted(path_fit["path_runs"])],
        "n_path_factors": int(len(path_fit["path_runs"])),
        "runtime_seconds": {
            "stage1": float(stage1_fit["runtime_seconds"]),
            "path": float(path_fit["runtime_seconds"]),
            "total": float(stage1_fit["runtime_seconds"] + path_fit["runtime_seconds"]),
        },
    }

    return {
        "stage1_tuning_result": stage1_fit["tuning_result"],
        "stage1_estimator": stage1_estimator,
        "stage1_km": stage1_fit["km"],
        "stage1_runtime_seconds": float(stage1_fit["runtime_seconds"]),
        "path_runs": path_fit["path_runs"],
        "path_df": path_fit["path_df"],
        "path_runtime_seconds": float(path_fit["runtime_seconds"]),
        "no_undersmoothing_factor": float(no_us_factor),
        "no_undersmoothing_summary": plugin_summary,
        "metadata": metadata,
    }


def fit_right_censored_cv_ipcw_l1mle_undersmoothed_plugin(
    observed_data: pd.DataFrame,
    *,
    targeting_points: Sequence[float],
    truth: Optional[RightCensoredTruth | dict[str, Any]] = None,
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 0,
    n_grid_points: int = 200,
    stage1_param_overrides: Optional[dict[str, Any]] = None,
    stage1_tuner_kwargs: Optional[dict[str, Any]] = None,
    norm_constraint_factors: Optional[Sequence[float]] = None,
    l1_kwargs: Optional[dict[str, Any]] = None,
    no_undersmoothing_factor: float = 1.0,
    targeting_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Fit the undersmoothed CV-IPCW-L1MLE plug-in survival estimator.

    The path is built exactly as in ``fit_right_censored_cv_ipcw_l1mle_plugin``.
    For each target time ``t0``, the API selects the smallest factor whose final
    EIF score satisfies ``|EIF_mean_final| <= threshold_final``. If none pass,
    it falls back to the factor with the smallest absolute final EIF score.
    The returned survival estimate is always the plug-in ``psi_init`` from the
    selected nuisance fit.
    """
    base = fit_right_censored_cv_ipcw_l1mle_plugin(
        observed_data,
        targeting_points=targeting_points,
        truth=truth,
        cv_folds=cv_folds,
        n_trials=n_trials,
        random_state=random_state,
        n_grid_points=n_grid_points,
        stage1_param_overrides=stage1_param_overrides,
        stage1_tuner_kwargs=stage1_tuner_kwargs,
        norm_constraint_factors=norm_constraint_factors,
        l1_kwargs=l1_kwargs,
        no_undersmoothing_factor=no_undersmoothing_factor,
    )
    target_arr = np.asarray(targeting_points, dtype=float).ravel()
    survival_fn, density_fn = _extract_truth_functions(truth)

    targeting_opts = dict(DEFAULT_TARGETING_KWARGS)
    targeting_opts.update(dict(targeting_kwargs or {}))

    summary_frames: list[pd.DataFrame] = []
    targeted_runs: dict[float, dict[str, Any]] = {}
    total_targeting_runtime = 0.0
    for factor, run in base["path_runs"].items():
        tic = time.perf_counter()
        targeted = right_censored_survival_targeting_M_step_v2(
            initial_estimator=run["estimator"],
            observed_data=observed_data,
            targeting_points=target_arr,
            km=base["stage1_km"],
            **targeting_opts,
        )
        runtime_seconds = float(time.perf_counter() - tic)
        total_targeting_runtime += runtime_seconds
        summary = targeted["summary"].copy().sort_values("t0").reset_index(drop=True)
        summary["norm_factor"] = float(factor)
        summary["norm_constraint"] = float(run["norm_constraint"])
        summary["estimated_density"] = np.asarray(
            run["estimator"].get_density_at_points(summary["t0"].to_numpy(dtype=float)),
            dtype=float,
        )
        summary["passes_final"] = (
            np.abs(summary["eif_mean_final"].to_numpy(dtype=float))
            <= summary["threshold_final"].to_numpy(dtype=float) + 1e-12
        )
        if survival_fn is not None:
            truth_survival = np.asarray(survival_fn(summary["t0"].to_numpy(dtype=float)), dtype=float)
            summary["truth_survival"] = truth_survival
            summary["abs_error_plugin"] = np.abs(summary["psi_init"].to_numpy(dtype=float) - truth_survival)
            summary["abs_error_targeted_diagnostic"] = np.abs(
                summary["psi_star"].to_numpy(dtype=float) - truth_survival
            )
        if density_fn is not None:
            truth_density = np.asarray(density_fn(summary["t0"].to_numpy(dtype=float)), dtype=float)
            summary["truth_density"] = truth_density
            summary["density_abs_error"] = np.abs(summary["estimated_density"].to_numpy(dtype=float) - truth_density)
        summary_frames.append(summary)
        targeted_runs[float(factor)] = {
            "runtime_seconds": runtime_seconds,
            "summary": summary,
        }

    targeting_summary_long = (
        pd.concat(summary_frames, ignore_index=True)
        .sort_values(["t0", "norm_factor"])
        .reset_index(drop=True)
    )

    selected_rows: list[dict[str, Any]] = []
    for t0, group in targeting_summary_long.groupby("t0", sort=True):
        ordered = group.sort_values("norm_factor").reset_index(drop=True)
        passing = ordered[ordered["passes_final"]]
        if not passing.empty:
            selected = passing.iloc[0]
            selection_status = "smallest_passing_factor"
        else:
            selected = ordered.iloc[int(np.argmin(np.abs(ordered["eif_mean_final"].to_numpy(dtype=float))))]
            selection_status = "closest_failing_factor"
        row = {
            "t0": float(t0),
            "selected_norm_factor": float(selected["norm_factor"]),
            "selected_norm_constraint": float(selected["norm_constraint"]),
            "psi_init": float(selected["psi_init"]),
            "psi_star_targeted_diagnostic": float(selected["psi_star"]),
            "estimated_density": float(selected["estimated_density"]),
            "eif_mean_final": float(selected["eif_mean_final"]),
            "threshold_final": float(selected["threshold_final"]),
            "passes_final": bool(selected["passes_final"]),
            "continued_past_one_step": bool(selected["continued_past_one_step"]),
            "n_iterations": int(selected["n_iterations"]),
            "selection_status": selection_status,
        }
        if "truth_survival" in selected:
            truth_survival = float(selected["truth_survival"])
            row["truth_survival"] = truth_survival
            row["abs_error_plugin"] = abs(float(selected["psi_init"]) - truth_survival)
            row["abs_error_targeted_diagnostic"] = abs(float(selected["psi_star"]) - truth_survival)
        if "truth_density" in selected:
            truth_density = float(selected["truth_density"])
            row["truth_density"] = truth_density
            row["density_abs_error"] = abs(float(selected["estimated_density"]) - truth_density)
        selected_rows.append(row)

    selected_summary = pd.DataFrame(selected_rows).sort_values("t0").reset_index(drop=True)
    metadata = dict(base["metadata"])
    metadata.update({
        "api": "fit_right_censored_cv_ipcw_l1mle_undersmoothed_plugin",
        "targeting_kwargs": dict(targeting_opts),
        "n_targeting_rows": int(targeting_summary_long.shape[0]),
        "n_selected_t0": int(selected_summary.shape[0]),
        "n_fallback_t0": int(np.sum(selected_summary["selection_status"] == "closest_failing_factor")),
        "runtime_seconds": {
            "stage1": float(base["stage1_runtime_seconds"]),
            "path": float(base["path_runtime_seconds"]),
            "targeting": float(total_targeting_runtime),
            "total": float(base["stage1_runtime_seconds"] + base["path_runtime_seconds"] + total_targeting_runtime),
        },
    })

    return {
        **base,
        "undersmoothed_summary": selected_summary,
        "undersmoothed_targeting_summary_long": targeting_summary_long,
        "undersmoothed_targeted_runs": targeted_runs,
        "undersmoothed_targeting_runtime_seconds": float(total_targeting_runtime),
        "metadata": metadata,
    }
