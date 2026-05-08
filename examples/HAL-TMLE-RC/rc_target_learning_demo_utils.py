from __future__ import annotations

from typing import Any, Optional, Sequence

from joblib import Parallel, delayed, parallel_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from haldensity.censoring.right.comparison import (
    DEFAULT_TARGETING_KWARGS,
    RightCensoredTruth,
    fit_right_censored_cv_ipcw_l1mle_plugin,
    simulate_beta_uniform_right_censored,
)
from haldensity.censoring.right.km import KaplanMeier
from haldensity.targeting import (
    right_censored_density_l2_targeting_M_step,
    right_censored_entropy_targeting_M_step,
    right_censored_rmst_targeting_M_step,
    right_censored_survival_targeting_M_step_v2,
)


TARGET_OPTIONS: tuple[str, ...] = ("Survival", "RMST", "DensitySquare", "Entropy")
DEFAULT_TARGET_GRID = np.linspace(0.1, 0.9, 9)
DEFAULT_INTEGRATION_GRID = np.linspace(0.0, 1.0, 4001)
DEFAULT_SIMULATION_KWARGS: dict[str, Any] = {
    "event_beta_shape": (2.2, 2.5),
    "censor_low": 0.0,
    "censor_high": 1.2,
}
DEFAULT_STAGE1_PARAM_OVERRIDES: dict[str, Any] = {
    "basis_order": [0],
    "norm_constraint": {"low": 3.0, "high": 8.0, "log": False},
}
DEFAULT_STAGE1_TUNER_KWARGS: dict[str, Any] = {
    "silent": True,
    "use_conservative_adjustment": False,
}
DEFAULT_L1_KWARGS: dict[str, Any] = {
    "n_grid_points": 120,
    "learning_rate": 0.1,
    "n_iterations": 160,
    "ll_change_tol": 1e-3,
    "history_every": 40,
}
DEFAULT_TARGETING_OPTIONS: dict[str, Any] = {
    **DEFAULT_TARGETING_KWARGS,
    "mode": "auto",
    "targeting_gbar_floor_scale": 1.0,
}


def normalize_target_type(target_type: str) -> str:
    key = str(target_type).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "survival": "Survival",
        "rmst": "RMST",
        "densitysquare": "DensitySquare",
        "densityl2": "DensitySquare",
        "densitysq": "DensitySquare",
        "entropy": "Entropy",
    }
    if key not in mapping:
        raise ValueError(f"Unknown target_type {target_type!r}. Choose from {TARGET_OPTIONS}.")
    return mapping[key]


def target_coordinate_column(target_type: str) -> Optional[str]:
    canonical = normalize_target_type(target_type)
    if canonical == "Survival":
        return "t0"
    if canonical == "RMST":
        return "tau"
    return None


def default_target_grid(n_points: int = 9) -> np.ndarray:
    return np.linspace(0.1, 0.9, int(n_points))


def simulate_demo_data(
    n: int,
    *,
    seed: int = 0,
    event_beta_shape: tuple[float, float] = (2.2, 2.5),
    censor_low: float = 0.0,
    censor_high: float = 1.2,
) -> dict[str, Any]:
    sim = simulate_beta_uniform_right_censored(
        n=int(n),
        seed=int(seed),
        event_beta_shape=event_beta_shape,
        censor_low=censor_low,
        censor_high=censor_high,
    )
    observed_data = sim["observed_data"].copy()
    sim["diagnostics"] = {
        "n_observations": int(observed_data.shape[0]),
        "uncensored_rate": float(observed_data["Delta"].mean()),
        "censoring_rate": float(1.0 - observed_data["Delta"].mean()),
        "event_beta_shape": tuple(float(x) for x in event_beta_shape),
        "censor_low": float(censor_low),
        "censor_high": float(censor_high),
    }
    return sim


def fit_cv_ipcw_l1mle_initial(
    observed_data: pd.DataFrame,
    *,
    evaluation_grid: Optional[Sequence[float]] = None,
    cv_folds: int = 2,
    n_trials: int = 4,
    random_state: int = 0,
    n_grid_points: int = 120,
    stage1_param_overrides: Optional[dict[str, Any]] = None,
    stage1_tuner_kwargs: Optional[dict[str, Any]] = None,
    norm_constraint_factors: Optional[Sequence[float]] = None,
    l1_kwargs: Optional[dict[str, Any]] = None,
    no_undersmoothing_factor: float = 1.0,
) -> dict[str, Any]:
    grid = (
        np.asarray(DEFAULT_TARGET_GRID, dtype=float)
        if evaluation_grid is None
        else np.asarray(evaluation_grid, dtype=float).ravel()
    )
    if grid.size == 0:
        raise ValueError("evaluation_grid must contain at least one point.")

    plugin_fit = fit_right_censored_cv_ipcw_l1mle_plugin(
        observed_data,
        targeting_points=grid,
        truth=None,
        cv_folds=int(cv_folds),
        n_trials=int(n_trials),
        random_state=int(random_state),
        n_grid_points=int(n_grid_points),
        stage1_param_overrides={
            **DEFAULT_STAGE1_PARAM_OVERRIDES,
            **(stage1_param_overrides or {}),
        },
        stage1_tuner_kwargs={
            **DEFAULT_STAGE1_TUNER_KWARGS,
            **(stage1_tuner_kwargs or {}),
        },
        norm_constraint_factors=[1.0]
        if norm_constraint_factors is None
        else [float(x) for x in norm_constraint_factors],
        l1_kwargs={
            **DEFAULT_L1_KWARGS,
            **(l1_kwargs or {}),
        },
        no_undersmoothing_factor=float(no_undersmoothing_factor),
    )

    selected_factor = float(plugin_fit["no_undersmoothing_factor"])
    selected_run = plugin_fit["path_runs"][selected_factor]
    return {
        "initial_estimator": selected_run["estimator"],
        "km": plugin_fit["stage1_km"],
        "observed_data": observed_data.copy(),
        "path_df": plugin_fit["path_df"].copy(),
        "plugin_fit": plugin_fit,
        "selected_norm_factor": selected_factor,
        "selected_norm_constraint": float(selected_run["norm_constraint"]),
        "runtime_seconds": float(plugin_fit["metadata"]["runtime_seconds"]["total"]),
    }


def run_targeted_fit(
    target_type: str,
    *,
    initial_estimator: Any,
    observed_data: pd.DataFrame,
    km: Any,
    target_grid: Optional[Sequence[float]] = None,
    targeting_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    canonical = normalize_target_type(target_type)
    options = {**DEFAULT_TARGETING_OPTIONS, **(targeting_options or {})}
    points = (
        np.asarray(DEFAULT_TARGET_GRID, dtype=float)
        if target_grid is None
        else np.asarray(target_grid, dtype=float).ravel()
    )

    if canonical == "Survival":
        return right_censored_survival_targeting_M_step_v2(
            initial_estimator=initial_estimator,
            observed_data=observed_data,
            targeting_points=points,
            km=km,
            **options,
        )
    if canonical == "RMST":
        return right_censored_rmst_targeting_M_step(
            initial_estimator=initial_estimator,
            observed_data=observed_data,
            targeting_points=points,
            km=km,
            **options,
        )
    if canonical == "DensitySquare":
        return right_censored_density_l2_targeting_M_step(
            initial_estimator=initial_estimator,
            observed_data=observed_data,
            km=km,
            **options,
        )
    return right_censored_entropy_targeting_M_step(
        initial_estimator=initial_estimator,
        observed_data=observed_data,
        km=km,
        **options,
    )


def target_grid_for_type(
    target_type: str,
    *,
    shared_grid: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    canonical = normalize_target_type(target_type)
    if canonical in {"Survival", "RMST"}:
        if shared_grid is None:
            return np.asarray(DEFAULT_TARGET_GRID, dtype=float)
        return np.asarray(shared_grid, dtype=float).ravel()
    return None


def run_demo_for_target(
    target_type: str,
    *,
    sim: dict[str, Any],
    initial_fit: dict[str, Any],
    shared_grid: Optional[Sequence[float]] = None,
    targeting_options: Optional[dict[str, Any]] = None,
    integration_grid: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    canonical = normalize_target_type(target_type)
    grid = target_grid_for_type(canonical, shared_grid=shared_grid)
    targeted_fit = run_targeted_fit(
        canonical,
        initial_estimator=initial_fit["initial_estimator"],
        observed_data=sim["observed_data"],
        km=initial_fit["km"],
        target_grid=grid,
        targeting_options=targeting_options,
    )
    augmented_summary = augment_targeted_summary(
        canonical,
        targeted_fit,
        sim["truth"],
        observed_data=sim["observed_data"],
        target_grid=grid,
        integration_grid=integration_grid,
    )
    return {
        "target_type": canonical,
        "target_grid": None if grid is None else np.asarray(grid, dtype=float),
        "targeted_fit": targeted_fit,
        "augmented_summary": augmented_summary,
        "display_summary": build_display_summary(canonical, augmented_summary),
    }


def run_all_target_demos(
    *,
    sim: dict[str, Any],
    initial_fit: dict[str, Any],
    shared_grid: Optional[Sequence[float]] = None,
    targeting_options: Optional[dict[str, Any]] = None,
    integration_grid: Optional[np.ndarray] = None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for target_type in TARGET_OPTIONS:
        results[target_type] = run_demo_for_target(
            target_type,
            sim=sim,
            initial_fit=initial_fit,
            shared_grid=shared_grid,
            targeting_options=targeting_options,
            integration_grid=integration_grid,
        )
    return results


def combine_demo_display_summaries(
    demo_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for target_type, result in demo_results.items():
        table = result["display_summary"].copy()
        table.insert(0, "target_type", target_type)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    area = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(area)))


def fit_event_km(observed_data: pd.DataFrame) -> KaplanMeier:
    km_data = observed_data.loc[:, ["T", "Delta"]].copy()
    km_data["Delta_event_km"] = 1 - km_data["Delta"].to_numpy(dtype=int)
    return KaplanMeier().fit(km_data, time_col="T", delta_col="Delta_event_km")


def evaluate_event_km_survival(event_km: KaplanMeier, points: Sequence[float]) -> np.ndarray:
    return np.asarray(event_km.predict(np.asarray(points, dtype=float).ravel()), dtype=float)


def integrate_event_km_rmst(event_km: KaplanMeier, tau_points: Sequence[float]) -> np.ndarray:
    taus = np.asarray(tau_points, dtype=float).ravel()
    times, surv = event_km.stepwise_survival_()
    estimates = np.zeros_like(taus, dtype=float)
    for idx, tau in enumerate(taus):
        if tau <= 0.0:
            estimates[idx] = 0.0
            continue
        segment_starts = np.concatenate(([0.0], times))
        segment_ends = np.concatenate((times, [float(tau)]))
        segment_levels = np.concatenate(([1.0], surv))
        widths = np.clip(np.minimum(segment_ends, float(tau)) - segment_starts, 0.0, None)
        estimates[idx] = float(np.sum(widths * segment_levels))
    return estimates


def km_comparator(
    target_type: str,
    observed_data: pd.DataFrame,
    *,
    target_grid: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    canonical = normalize_target_type(target_type)
    if canonical not in {"Survival", "RMST"}:
        return None
    if target_grid is None:
        raise ValueError(f"target_grid is required for the {canonical} KM comparator.")
    event_km = fit_event_km(observed_data)
    if canonical == "Survival":
        return evaluate_event_km_survival(event_km, target_grid)
    return integrate_event_km_rmst(event_km, target_grid)


def evaluate_truth_target(
    truth: RightCensoredTruth,
    target_type: str,
    *,
    target_grid: Optional[Sequence[float]] = None,
    integration_grid: Optional[np.ndarray] = None,
) -> np.ndarray | float:
    canonical = normalize_target_type(target_type)
    grid = (
        np.asarray(DEFAULT_INTEGRATION_GRID, dtype=float)
        if integration_grid is None
        else np.asarray(integration_grid, dtype=float).ravel()
    )
    if canonical == "Survival":
        if target_grid is None:
            raise ValueError("target_grid is required for Survival truth evaluation.")
        points = np.asarray(target_grid, dtype=float).ravel()
        return np.asarray(truth.survival_fn(points), dtype=float)

    if canonical == "RMST":
        if target_grid is None:
            raise ValueError("target_grid is required for RMST truth evaluation.")
        points = np.asarray(target_grid, dtype=float).ravel()
        survival_vals = np.asarray(truth.survival_fn(grid), dtype=float)
        cumulative = _cumulative_trapezoid(survival_vals, grid)
        return np.interp(points, grid, cumulative, left=0.0, right=float(cumulative[-1]))

    mid = 0.5 * (grid[:-1] + grid[1:])
    delta = np.diff(grid)
    density_vals = np.asarray(truth.density_fn(mid), dtype=float)
    if canonical == "DensitySquare":
        return float(np.sum(np.square(density_vals) * delta))
    return float(-np.sum(density_vals * np.log(np.clip(density_vals, 1e-300, None)) * delta))


def augment_targeted_summary(
    target_type: str,
    targeted_fit: dict[str, Any],
    truth: RightCensoredTruth,
    *,
    observed_data: Optional[pd.DataFrame] = None,
    target_grid: Optional[Sequence[float]] = None,
    integration_grid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    canonical = normalize_target_type(target_type)
    summary = targeted_fit["summary"].copy()
    coord_col = target_coordinate_column(canonical)
    if coord_col is not None:
        summary = summary.sort_values(coord_col).reset_index(drop=True)
        truth_vals = np.asarray(
            evaluate_truth_target(
                truth,
                canonical,
                target_grid=summary[coord_col].to_numpy(dtype=float),
                integration_grid=integration_grid,
            ),
            dtype=float,
        )
        summary["truth"] = truth_vals
        if observed_data is not None and canonical in {"Survival", "RMST"}:
            summary["psi_km"] = np.asarray(
                km_comparator(
                    canonical,
                    observed_data=observed_data,
                    target_grid=summary[coord_col].to_numpy(dtype=float),
                ),
                dtype=float,
            )
    else:
        summary["target"] = canonical
        summary["truth"] = float(
            evaluate_truth_target(
                truth,
                canonical,
                target_grid=target_grid,
                integration_grid=integration_grid,
            )
        )

    summary["plugin_bias"] = summary["psi_init"].to_numpy(dtype=float) - summary["truth"].to_numpy(dtype=float)
    summary["one_step_bias"] = summary["psi_one_step"].to_numpy(dtype=float) - summary["truth"].to_numpy(dtype=float)
    summary["targeted_bias"] = summary["psi_final"].to_numpy(dtype=float) - summary["truth"].to_numpy(dtype=float)
    if "psi_km" in summary.columns:
        summary["km_bias"] = summary["psi_km"].to_numpy(dtype=float) - summary["truth"].to_numpy(dtype=float)
    summary["plugin_squared_error"] = np.square(summary["plugin_bias"].to_numpy(dtype=float))
    summary["targeted_squared_error"] = np.square(summary["targeted_bias"].to_numpy(dtype=float))
    if "psi_km" in summary.columns:
        summary["km_squared_error"] = np.square(summary["km_bias"].to_numpy(dtype=float))
    summary["tmle_was_used"] = summary["n_iterations"].to_numpy(dtype=float) > 0
    return summary


def build_display_summary(target_type: str, augmented_summary: pd.DataFrame) -> pd.DataFrame:
    canonical = normalize_target_type(target_type)
    coord_col = target_coordinate_column(canonical)
    columns = [
        "truth",
        "psi_km",
        "psi_init",
        "psi_one_step",
        "psi_final",
        "km_bias",
        "plugin_bias",
        "targeted_bias",
        "passes_initial_threshold",
        "status_one_step",
        "n_iterations",
        "stop_reason",
    ]
    columns = [col for col in columns if col in augmented_summary.columns]
    if coord_col is not None:
        columns = [coord_col, *columns]
    return augmented_summary.loc[:, columns].copy()


def plot_target_estimates(
    target_type: str,
    augmented_summary: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    canonical = normalize_target_type(target_type)
    coord_col = target_coordinate_column(canonical)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if coord_col is not None:
        x = augmented_summary[coord_col].to_numpy(dtype=float)
        ax.plot(x, augmented_summary["truth"].to_numpy(dtype=float), color="black", linewidth=2, label="Truth")
        if "psi_km" in augmented_summary.columns:
            ax.plot(x, augmented_summary["psi_km"].to_numpy(dtype=float), marker="^", label="KM")
        ax.plot(x, augmented_summary["psi_init"].to_numpy(dtype=float), marker="o", label="CV-IPCW-L1MLE")
        ax.plot(x, augmented_summary["psi_final"].to_numpy(dtype=float), marker="s", label="Final targeted")
        if {"ci_lower", "ci_upper"}.issubset(augmented_summary.columns):
            ax.fill_between(
                x,
                augmented_summary["ci_lower"].to_numpy(dtype=float),
                augmented_summary["ci_upper"].to_numpy(dtype=float),
                color="tab:orange",
                alpha=0.15,
                label="Final 95% CI",
            )
        ax.set_xlabel("t0" if canonical == "Survival" else "tau")
        ax.set_ylabel(canonical)
    else:
        row = augmented_summary.iloc[0]
        labels = ["Truth", "CV-IPCW-L1MLE", "One-step", "Final targeted"]
        values = [
            float(row["truth"]),
            float(row["psi_init"]),
            float(row["psi_one_step"]),
            float(row["psi_final"]),
        ]
        ax.bar(labels, values, color=["black", "tab:blue", "tab:green", "tab:orange"])
        ax.set_ylabel(canonical)

    ax.set_title(f"{canonical} targeting")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc="best")
    return ax


def plot_score_diagnostics(
    target_type: str,
    augmented_summary: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    canonical = normalize_target_type(target_type)
    coord_col = target_coordinate_column(canonical)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if coord_col is not None:
        x = augmented_summary[coord_col].to_numpy(dtype=float)
        ax.plot(
            x,
            np.abs(augmented_summary["eif_mean_initial_stage"].to_numpy(dtype=float)),
            marker="o",
            label="|EIF mean| initial",
        )
        ax.plot(
            x,
            augmented_summary["threshold_initial"].to_numpy(dtype=float),
            linestyle="--",
            label="Initial threshold",
        )
        ax.plot(
            x,
            np.abs(augmented_summary["eif_mean_final"].to_numpy(dtype=float)),
            marker="s",
            label="|EIF mean| final",
        )
        ax.plot(
            x,
            augmented_summary["threshold_final"].to_numpy(dtype=float),
            linestyle=":",
            label="Final threshold",
        )
        ax.set_xlabel("t0" if canonical == "Survival" else "tau")
    else:
        row = augmented_summary.iloc[0]
        labels = ["|EIF| initial", "threshold initial", "|EIF| final", "threshold final"]
        values = [
            abs(float(row["eif_mean_initial_stage"])),
            float(row["threshold_initial"]),
            abs(float(row["eif_mean_final"])),
            float(row["threshold_final"]),
        ]
        ax.bar(labels, values, color=["tab:blue", "tab:gray", "tab:orange", "tab:gray"])

    ax.set_ylabel("Score scale")
    ax.set_title(f"{canonical} score diagnostics")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc="best")
    return ax


def _extract_estimate_rows(
    target_type: str,
    augmented_summary: pd.DataFrame,
    *,
    replicate: int,
) -> pd.DataFrame:
    canonical = normalize_target_type(target_type)
    coord_col = target_coordinate_column(canonical)
    rows = augmented_summary.copy()
    rows["replicate"] = int(replicate)
    if coord_col is None:
        rows["target"] = canonical
        keep = ["replicate", "target", "truth", "psi_init", "psi_final"]
    else:
        keep = ["replicate", coord_col, "truth", "psi_km", "psi_init", "psi_final"]
    keep = [col for col in keep if col in rows.columns]
    return rows.loc[:, keep].copy()


def _resolve_mc_worker_count(n_workers: int, n_tasks: int) -> int:
    return max(1, min(int(n_workers), int(n_tasks)))


def _run_parallel_mc_tasks(
    worker: Any,
    tasks: list[dict[str, Any]],
    *,
    n_workers: int,
    show_progress: bool,
    progress_desc: str,
) -> list[Any]:
    if len(tasks) == 0:
        return []

    if int(n_workers) <= 1 or len(tasks) == 1:
        iterator = tasks
        if show_progress:
            iterator = tqdm(tasks, total=len(tasks), desc=progress_desc)
        return [worker(task) for task in iterator]

    max_workers = _resolve_mc_worker_count(n_workers, len(tasks))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        generator = Parallel(
            n_jobs=max_workers,
            backend="loky",
            return_as="generator_unordered",
            batch_size=1,
        )(delayed(worker)(task) for task in tasks)
        iterator = generator
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc=progress_desc)
        results = list(iterator)
    return results


def _single_target_mc_replicate_worker(task: dict[str, Any]) -> tuple[int, pd.DataFrame]:
    canonical = normalize_target_type(task["target_type"])
    rep = int(task["rep"])
    points = np.asarray(task["target_grid"], dtype=float).ravel()
    sim = simulate_demo_data(
        n=int(task["n"]),
        seed=int(task["base_seed"] + rep),
        **dict(task["simulation_kwargs"]),
    )
    init_bundle = fit_cv_ipcw_l1mle_initial(
        sim["observed_data"],
        evaluation_grid=points,
        random_state=int(task["base_seed"] + rep),
        **dict(task["initial_fit_kwargs"]),
    )
    targeted = run_targeted_fit(
        canonical,
        initial_estimator=init_bundle["initial_estimator"],
        observed_data=sim["observed_data"],
        km=init_bundle["km"],
        target_grid=points,
        targeting_options=dict(task["targeting_options"]),
    )
    augmented = augment_targeted_summary(
        canonical,
        targeted,
        sim["truth"],
        observed_data=sim["observed_data"],
        target_grid=points,
        integration_grid=(
            None
            if task["integration_grid"] is None
            else np.asarray(task["integration_grid"], dtype=float)
        ),
    )
    return rep, _extract_estimate_rows(canonical, augmented, replicate=rep)


def _all_target_mc_replicate_worker(
    task: dict[str, Any],
) -> tuple[int, dict[str, pd.DataFrame]]:
    rep = int(task["rep"])
    shared_grid = np.asarray(task["shared_grid"], dtype=float).ravel()
    sim = simulate_demo_data(
        n=int(task["n"]),
        seed=int(task["base_seed"] + rep),
        **dict(task["simulation_kwargs"]),
    )
    init_bundle = fit_cv_ipcw_l1mle_initial(
        sim["observed_data"],
        evaluation_grid=shared_grid,
        random_state=int(task["base_seed"] + rep),
        **dict(task["initial_fit_kwargs"]),
    )
    demo_results = run_all_target_demos(
        sim=sim,
        initial_fit=init_bundle,
        shared_grid=shared_grid,
        targeting_options=dict(task["targeting_options"]),
        integration_grid=(
            None
            if task["integration_grid"] is None
            else np.asarray(task["integration_grid"], dtype=float)
        ),
    )
    return (
        rep,
        {
            target_type: _extract_estimate_rows(
                target_type,
                result["augmented_summary"],
                replicate=rep,
            )
            for target_type, result in demo_results.items()
        },
    )


def _leave_one_out_oracle_coverage(values: np.ndarray, truth: float) -> float:
    vals = np.asarray(values, dtype=float)
    if vals.size <= 2:
        return float("nan")
    covered = []
    for idx in range(vals.size):
        others = np.delete(vals, idx)
        oracle_sd = float(np.std(others, ddof=1))
        if not np.isfinite(oracle_sd):
            covered.append(np.nan)
            continue
        covered.append(abs(float(vals[idx]) - truth) <= 1.96 * oracle_sd)
    return float(np.nanmean(covered))


def summarize_mc_metrics(target_type: str, estimates: pd.DataFrame) -> pd.DataFrame:
    canonical = normalize_target_type(target_type)
    coord_col = target_coordinate_column(canonical) or "target"
    metric_rows: list[dict[str, Any]] = []
    for coord_value, group in estimates.groupby(coord_col, sort=True):
        truth = float(group["truth"].iloc[0])
        stage_pairs: list[tuple[str, str]] = []
        if "psi_km" in group.columns:
            stage_pairs.append(("psi_km", "km"))
        stage_pairs.extend((("psi_init", "initial"), ("psi_final", "final")))
        for stage_col, stage_label in stage_pairs:
            vals = group[stage_col].to_numpy(dtype=float)
            bias = float(np.mean(vals - truth))
            variance = float(np.var(vals, ddof=1)) if vals.size > 1 else float("nan")
            mse = float(np.mean(np.square(vals - truth)))
            oracle_sd = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
            row = {
                "stage": stage_label,
                "truth": truth,
                "mean_estimate": float(np.mean(vals)),
                "bias": bias,
                "variance": variance,
                "mse": mse,
                "oracle_sd": oracle_sd,
                "oracle_coverage": _leave_one_out_oracle_coverage(vals, truth),
                "n_replicates": int(vals.size),
            }
            row[coord_col] = coord_value
            metric_rows.append(row)

    summary = pd.DataFrame(metric_rows)
    stage_order = [stage for stage in ("km", "initial", "final") if stage in summary["stage"].unique()]
    summary["stage"] = pd.Categorical(summary["stage"], categories=stage_order, ordered=True)
    if target_coordinate_column(canonical) is not None:
        summary = summary.sort_values([coord_col, "stage"]).reset_index(drop=True)
    else:
        summary = summary.sort_values(["stage"]).reset_index(drop=True)
    return summary


def run_small_mc(
    target_type: str,
    *,
    n_reps: int = 6,
    n: int = 160,
    base_seed: int = 1000,
    target_grid: Optional[Sequence[float]] = None,
    simulation_kwargs: Optional[dict[str, Any]] = None,
    initial_fit_kwargs: Optional[dict[str, Any]] = None,
    targeting_options: Optional[dict[str, Any]] = None,
    integration_grid: Optional[np.ndarray] = None,
    n_workers: int = 8,
    show_progress: bool = True,
) -> dict[str, Any]:
    canonical = normalize_target_type(target_type)
    points = (
        np.asarray(DEFAULT_TARGET_GRID, dtype=float)
        if target_grid is None
        else np.asarray(target_grid, dtype=float).ravel()
    )
    sim_kwargs = {**DEFAULT_SIMULATION_KWARGS, **(simulation_kwargs or {})}
    init_kwargs = dict(initial_fit_kwargs or {})
    target_opts = dict(targeting_options or {})
    tasks = [
        {
            "target_type": canonical,
            "rep": int(rep),
            "n": int(n),
            "base_seed": int(base_seed),
            "target_grid": points,
            "simulation_kwargs": sim_kwargs,
            "initial_fit_kwargs": init_kwargs,
            "targeting_options": target_opts,
            "integration_grid": integration_grid,
        }
        for rep in range(int(n_reps))
    ]
    worker_results = _run_parallel_mc_tasks(
        _single_target_mc_replicate_worker,
        tasks,
        n_workers=int(n_workers),
        show_progress=bool(show_progress),
        progress_desc=f"{canonical} MC",
    )
    estimate_rows = [rows for _, rows in sorted(worker_results, key=lambda x: int(x[0]))]

    estimates = pd.concat(estimate_rows, ignore_index=True)
    metrics = summarize_mc_metrics(canonical, estimates)
    return {
        "estimates": estimates,
        "metrics": metrics,
        "metadata": {
            "target_type": canonical,
            "n_reps": int(n_reps),
            "n": int(n),
            "base_seed": int(base_seed),
            "target_grid": points.tolist(),
            "simulation_kwargs": sim_kwargs,
            "initial_fit_kwargs": init_kwargs,
            "targeting_options": target_opts,
            "n_workers": int(_resolve_mc_worker_count(n_workers, n_reps)),
            "show_progress": bool(show_progress),
            "parallel_backend": "joblib-loky-processes",
        },
    }


def run_all_target_small_mc(
    *,
    n_reps: int = 6,
    n: int = 160,
    base_seed: int = 1000,
    shared_grid: Optional[Sequence[float]] = None,
    simulation_kwargs: Optional[dict[str, Any]] = None,
    initial_fit_kwargs: Optional[dict[str, Any]] = None,
    targeting_options: Optional[dict[str, Any]] = None,
    integration_grid: Optional[np.ndarray] = None,
    n_workers: int = 8,
    show_progress: bool = True,
) -> dict[str, Any]:
    shared_grid_arr = (
        np.asarray(DEFAULT_TARGET_GRID, dtype=float)
        if shared_grid is None
        else np.asarray(shared_grid, dtype=float).ravel()
    )
    sim_kwargs = {**DEFAULT_SIMULATION_KWARGS, **(simulation_kwargs or {})}
    init_kwargs = dict(initial_fit_kwargs or {})
    target_opts = dict(targeting_options or {})

    estimate_rows_by_target: dict[str, list[pd.DataFrame]] = {target: [] for target in TARGET_OPTIONS}
    tasks = [
        {
            "rep": int(rep),
            "n": int(n),
            "base_seed": int(base_seed),
            "shared_grid": shared_grid_arr,
            "simulation_kwargs": sim_kwargs,
            "initial_fit_kwargs": init_kwargs,
            "targeting_options": target_opts,
            "integration_grid": integration_grid,
        }
        for rep in range(int(n_reps))
    ]
    worker_results = _run_parallel_mc_tasks(
        _all_target_mc_replicate_worker,
        tasks,
        n_workers=int(n_workers),
        show_progress=bool(show_progress),
        progress_desc="All-target MC",
    )
    for _, estimate_map in sorted(worker_results, key=lambda x: int(x[0])):
        for target_type, rows in estimate_map.items():
            estimate_rows_by_target[target_type].append(rows)

    by_target: dict[str, dict[str, Any]] = {}
    metric_tables: list[pd.DataFrame] = []
    estimate_tables: list[pd.DataFrame] = []
    for target_type in TARGET_OPTIONS:
        estimates = pd.concat(estimate_rows_by_target[target_type], ignore_index=True)
        metrics = summarize_mc_metrics(target_type, estimates)
        by_target[target_type] = {
            "estimates": estimates,
            "metrics": metrics,
            "metadata": {
                "target_type": target_type,
                "n_reps": int(n_reps),
                "n": int(n),
            },
        }
        metrics_with_label = metrics.copy()
        metrics_with_label.insert(0, "target_type", target_type)
        estimates_with_label = estimates.copy()
        estimates_with_label.insert(0, "target_type", target_type)
        metric_tables.append(metrics_with_label)
        estimate_tables.append(estimates_with_label)

    return {
        "by_target": by_target,
        "metrics": pd.concat(metric_tables, ignore_index=True),
        "estimates": pd.concat(estimate_tables, ignore_index=True),
        "metadata": {
            "target_types": list(TARGET_OPTIONS),
            "n_reps": int(n_reps),
            "n": int(n),
            "base_seed": int(base_seed),
            "shared_grid": shared_grid_arr.tolist(),
            "simulation_kwargs": sim_kwargs,
            "initial_fit_kwargs": init_kwargs,
            "targeting_options": target_opts,
            "shared_initial_fit_per_replicate": True,
            "n_workers": int(_resolve_mc_worker_count(n_workers, n_reps)),
            "show_progress": bool(show_progress),
            "parallel_backend": "joblib-loky-processes",
        },
    }


def mc_metric_overview(mc_all_targets: dict[str, Any]) -> pd.DataFrame:
    metrics = mc_all_targets["metrics"].copy()
    coord_cols = [col for col in ("t0", "tau", "target") if col in metrics.columns]
    group_cols = ["target_type", "stage", *coord_cols]
    return metrics.loc[:, group_cols + ["bias", "variance", "mse", "oracle_coverage", "n_replicates"]]


def plot_mc_metrics(
    target_type: str,
    metrics: pd.DataFrame,
    *,
    metric: str = "mse",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    canonical = normalize_target_type(target_type)
    coord_col = target_coordinate_column(canonical)
    if metric not in metrics.columns:
        raise ValueError(f"Unknown metric {metric!r}.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if coord_col is None:
        ordered = metrics.sort_values("stage").reset_index(drop=True)
        ax.bar(ordered["stage"], ordered[metric].to_numpy(dtype=float), color=["tab:blue", "tab:orange"])
        ax.set_xlabel("stage")
    else:
        for stage, group in metrics.groupby("stage", sort=True, observed=False):
            ax.plot(
                group[coord_col].to_numpy(dtype=float),
                group[metric].to_numpy(dtype=float),
                marker="o",
                label=stage,
            )
        ax.set_xlabel("t0" if canonical == "Survival" else "tau")

    ax.set_ylabel(metric)
    ax.set_title(f"{canonical} MC {metric}")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc="best")
    return ax
