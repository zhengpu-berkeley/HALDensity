from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from haldensity.censoring.right.estimators import RightCensoredInitEstimator
from haldensity.censoring.right.km import KaplanMeier


@dataclass
class RCInitialFit:
    estimator: RightCensoredInitEstimator
    basis_order: int
    grid_edges: np.ndarray
    grid_midpoints: np.ndarray
    delta_j: np.ndarray
    density_grid: np.ndarray
    log_density_grid: np.ndarray
    survival_grid: np.ndarray
    edge_survival: np.ndarray
    theta_hat: np.ndarray
    grid_points_hal_selected: np.ndarray


@dataclass
class RCTargetGrid:
    t0: float
    grid_edges: np.ndarray
    grid_midpoints: np.ndarray
    delta_j: np.ndarray
    density_grid: np.ndarray
    log_density_grid: np.ndarray
    survival_grid: np.ndarray
    edge_survival: np.ndarray
    t0_inserted: bool


@dataclass
class RCCensoringCache:
    km: KaplanMeier
    jump_times: np.ndarray
    gbar_right: np.ndarray
    gbar_left: np.ndarray
    jump_masses: np.ndarray
    clip: float


@dataclass
class PointwiseTMLEResult:
    t0: float
    psi_init: float
    psi_star: float
    epsilon: float
    estimand_variance: float
    standard_error: float
    score_at_zero: float
    score_at_solution: float
    objective_at_solution: float
    eif_mean: float
    solve_method: str
    converged: bool
    diagnostics: dict[str, Any]


@dataclass
class PointwiseTMLEState:
    t0: float
    grid_edges: np.ndarray
    grid_midpoints: np.ndarray
    delta_j: np.ndarray
    log_density_grid: np.ndarray
    density_grid: np.ndarray
    survival_grid: np.ndarray
    edge_survival: np.ndarray
    psi_current: float
    target_grid_augmented_with_t0: bool
    censoring_cache: RCCensoringCache
    raw_direction: Optional[np.ndarray] = None
    centered_direction: Optional[np.ndarray] = None
    eif_values: Optional[np.ndarray] = None
    eif_mean: float = float("nan")
    eif_sigma: float = float("nan")
    score_at_zero: float = float("nan")


def _compute_survival_from_grid(density_grid: np.ndarray, delta_j: np.ndarray) -> np.ndarray:
    mass = density_grid * delta_j
    survival = np.cumsum(mass[::-1])[::-1]
    return np.clip(survival, 0.0, 1.0)


def _interp_density(grid: np.ndarray, density: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.interp(pts, grid, density, left=float(density[0]), right=float(density[-1]))


def _interp_survival_from_edges(
    grid_edges: np.ndarray, edge_survival: np.ndarray, points: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.interp(pts, grid_edges, edge_survival, left=1.0, right=0.0)


def _interp_direction(grid: np.ndarray, direction: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.interp(pts, grid, direction, left=float(direction[0]), right=float(direction[-1]))


def _normalize_targeting_points(
    targeting_points: Union[float, Sequence[float], np.ndarray]
) -> np.ndarray:
    points = np.atleast_1d(np.asarray(targeting_points, dtype=float))
    if points.ndim != 1:
        raise ValueError("targeting_points must be a scalar or one-dimensional sequence.")
    if np.any((points <= 0.0) | (points >= 1.0)):
        raise ValueError("This prototype expects target times strictly inside (0, 1).")
    return points


def _build_initial_fit(estimator: RightCensoredInitEstimator) -> RCInitialFit:
    grid_midpoints, density_grid = estimator.get_density()
    grid_midpoints = np.asarray(grid_midpoints, dtype=float)
    delta_j = np.asarray(estimator.delta_j, dtype=float).copy()
    density_grid = np.asarray(density_grid, dtype=float)
    survival_grid = _compute_survival_from_grid(density_grid, delta_j)
    if getattr(estimator, "grid_points", None) is not None:
        grid_edges = np.asarray(estimator.grid_points, dtype=float).copy()
    else:
        grid_edges = np.concatenate(([0.0], np.cumsum(delta_j)))
        grid_edges[-1] = 1.0
    edge_survival = np.concatenate((survival_grid, [0.0]))
    log_density_grid = np.log(np.clip(density_grid, 1e-300, None))
    if estimator.theta_hat is None or estimator.grid_points_hal_selected is None:
        raise RuntimeError("Stage 1 estimator is missing fitted HAL attributes.")
    return RCInitialFit(
        estimator=estimator,
        basis_order=int(estimator.basis_order),
        grid_edges=grid_edges,
        grid_midpoints=grid_midpoints,
        delta_j=delta_j,
        density_grid=density_grid,
        log_density_grid=log_density_grid,
        survival_grid=survival_grid,
        edge_survival=edge_survival,
        theta_hat=np.asarray(estimator.theta_hat, dtype=float).copy(),
        grid_points_hal_selected=np.asarray(estimator.grid_points_hal_selected, dtype=float).copy(),
    )


def _build_pointwise_target_grid(initial_fit: RCInitialFit, t0: float) -> RCTargetGrid:
    if np.any(np.isclose(initial_fit.grid_edges, t0, atol=1e-12, rtol=0.0)):
        grid_edges = initial_fit.grid_edges.copy()
        t0_inserted = False
    else:
        grid_edges = np.sort(np.concatenate((initial_fit.grid_edges, np.array([t0], dtype=float))))
        t0_inserted = True

    grid_midpoints = (grid_edges[:-1] + grid_edges[1:]) / 2.0
    delta_j = np.diff(grid_edges)
    density_grid = np.asarray(
        initial_fit.estimator.get_density_at_points(grid_midpoints), dtype=float
    )
    density_mass = float(np.sum(density_grid * delta_j))
    density_grid = density_grid / density_mass
    log_density_grid = np.log(np.clip(density_grid, 1e-300, None))
    survival_grid = _compute_survival_from_grid(density_grid, delta_j)
    edge_survival = np.concatenate((survival_grid, [0.0]))

    return RCTargetGrid(
        t0=float(t0),
        grid_edges=grid_edges,
        grid_midpoints=grid_midpoints,
        delta_j=delta_j,
        density_grid=density_grid,
        log_density_grid=log_density_grid,
        survival_grid=survival_grid,
        edge_survival=edge_survival,
        t0_inserted=t0_inserted,
    )


def _build_censoring_cache(km: KaplanMeier, clip: float) -> RCCensoringCache:
    jump_times, gbar_right = km.stepwise_survival_()
    jump_times = np.asarray(jump_times, dtype=float)
    gbar_right = np.asarray(gbar_right, dtype=float)
    if jump_times.size == 0:
        gbar_left = np.empty(0, dtype=float)
        jump_masses = np.empty(0, dtype=float)
    else:
        gbar_left = np.concatenate(([1.0], gbar_right[:-1]))
        jump_masses = np.clip(gbar_left - gbar_right, 0.0, None)
    return RCCensoringCache(
        km=km,
        jump_times=jump_times,
        gbar_right=gbar_right,
        gbar_left=gbar_left,
        jump_masses=jump_masses,
        clip=float(clip),
    )


def _evaluate_gbar(cache: RCCensoringCache, t: np.ndarray | float) -> np.ndarray:
    vals = np.asarray(cache.km.predict(t), dtype=float)
    return np.maximum(vals, cache.clip)


def _evaluate_gbar_left(cache: RCCensoringCache, t: np.ndarray | float) -> np.ndarray:
    ts = np.asarray(t, dtype=float)
    if cache.jump_times.size == 0:
        out = np.ones_like(ts, dtype=float)
    else:
        idx = np.searchsorted(cache.jump_times, ts, side="left") - 1
        out = np.where(
            idx >= 0,
            cache.gbar_right[np.clip(idx, 0, len(cache.gbar_right) - 1)],
            1.0,
        )
    return np.maximum(out, cache.clip)


def _evaluate_survival(
    grid_like: RCInitialFit | RCTargetGrid, t: np.ndarray | float, survival_clip: float
) -> np.ndarray:
    vals = _interp_survival_from_edges(grid_like.grid_edges, grid_like.edge_survival, t)
    return np.maximum(vals, survival_clip)


def _compute_full_data_gradient_t0(
    target_grid: RCTargetGrid, t0: float, survival_clip: float
) -> tuple[np.ndarray, float]:
    s_t0 = float(_evaluate_survival(target_grid, np.array([t0]), survival_clip)[0])
    gradient_grid = (target_grid.grid_midpoints > t0).astype(float) - s_t0
    return gradient_grid, s_t0


def _center_direction_on_grid(
    direction_grid: np.ndarray, density_grid: np.ndarray, delta_j: np.ndarray
) -> tuple[np.ndarray, float]:
    direction_mean = float(np.sum(direction_grid * density_grid * delta_j))
    return direction_grid - direction_mean, direction_mean


def _compute_dfg_t0_on_grid(
    target_grid: RCTargetGrid,
    censoring_cache: RCCensoringCache,
    t0: float,
    survival_clip: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    grid = target_grid.grid_midpoints
    full_gradient_grid, s_t0 = _compute_full_data_gradient_t0(target_grid, t0, survival_clip)
    gbar_t0_right = float(_evaluate_gbar(censoring_cache, np.array([t0]))[0])
    first_term = (grid > t0).astype(float) / gbar_t0_right

    active_jump_mask = censoring_cache.jump_times <= t0
    jump_times = censoring_cache.jump_times[active_jump_mask]
    jump_masses = censoring_cache.jump_masses[active_jump_mask]

    if jump_times.size == 0:
        second_term = np.zeros_like(grid)
        increments = np.empty(0, dtype=float)
        gbar_right_u = np.empty(0, dtype=float)
    else:
        s_u = np.maximum(_evaluate_survival(target_grid, jump_times, survival_clip), survival_clip)
        gbar_right_u = _evaluate_gbar(censoring_cache, jump_times)
        increments = s_t0 * jump_masses / (s_u * np.square(gbar_right_u))
        cumulative_increments = np.cumsum(increments)
        cutoff_points = np.minimum(grid, t0)
        cutoff_idx = np.searchsorted(jump_times, cutoff_points, side="right") - 1
        second_term = np.where(
            cutoff_idx >= 0,
            cumulative_increments[np.clip(cutoff_idx, 0, len(cumulative_increments) - 1)],
            0.0,
        )

    raw_direction = first_term - second_term
    centered_direction, raw_mean = _center_direction_on_grid(
        raw_direction, target_grid.density_grid, target_grid.delta_j
    )
    direction_details = {
        "full_gradient_grid": full_gradient_grid,
        "s_t0": s_t0,
        "gbar_convention": "right_continuous",
        "stage1_ipcw_convention": "repo_default",
        "target_grid_augmented_with_t0": target_grid.t0_inserted,
        "gbar_t0_right": gbar_t0_right,
        "jump_times": jump_times,
        "gbar_jump_denom": (
            np.square(gbar_right_u) if jump_times.size > 0 else np.empty(0, dtype=float)
        ),
        "jump_increments": increments,
        "first_term_grid": first_term,
        "cumulative_jump_term_grid": second_term,
        "raw_mean": raw_mean,
    }
    return raw_direction, centered_direction, direction_details


def _tilted_density_and_survival(
    target_grid: RCTargetGrid, centered_direction_grid: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    log_unnormalized = target_grid.log_density_grid + eps * centered_direction_grid
    shift = float(np.max(log_unnormalized))
    density_unnormalized = np.exp(np.clip(log_unnormalized - shift, -700, 700))
    normalizer = float(np.sum(density_unnormalized * target_grid.delta_j))
    density_tilted = density_unnormalized / normalizer
    survival_tilted = _compute_survival_from_grid(density_tilted, target_grid.delta_j)
    edge_survival_tilted = np.concatenate((survival_tilted, [0.0]))
    return density_tilted, survival_tilted, edge_survival_tilted, normalizer


def _tmle_loglik_t0(
    eps: float,
    target_grid: RCTargetGrid,
    centered_direction_grid: np.ndarray,
    data: pd.DataFrame,
) -> float:
    density_tilted, _, edge_survival_tilted, _ = _tilted_density_and_survival(
        target_grid, centered_direction_grid, eps
    )
    observed_t = data["T"].to_numpy(dtype=float)
    delta = data["Delta"].to_numpy(dtype=float)
    p_obs = np.maximum(
        _interp_density(target_grid.grid_midpoints, density_tilted, observed_t), 1e-300
    )
    s_obs = np.maximum(
        _interp_survival_from_edges(target_grid.grid_edges, edge_survival_tilted, observed_t),
        1e-300,
    )
    return float(np.sum(delta * np.log(p_obs) + (1.0 - delta) * np.log(s_obs)))


def _tmle_score_t0(
    eps: float,
    target_grid: RCTargetGrid,
    centered_direction_grid: np.ndarray,
    data: pd.DataFrame,
) -> float:
    density_tilted, _, edge_survival_tilted, _ = _tilted_density_and_survival(
        target_grid, centered_direction_grid, eps
    )
    observed_t = data["T"].to_numpy(dtype=float)
    delta = data["Delta"].to_numpy(dtype=float)

    overall_mean = float(np.sum(centered_direction_grid * density_tilted * target_grid.delta_j))
    tail_weighted = np.cumsum(
        (centered_direction_grid * density_tilted * target_grid.delta_j)[::-1]
    )[::-1]
    edge_tail_weighted = np.concatenate((tail_weighted, [0.0]))

    direction_obs = _interp_direction(target_grid.grid_midpoints, centered_direction_grid, observed_t)
    tail_num_obs = _interp_survival_from_edges(target_grid.grid_edges, edge_tail_weighted, observed_t)
    tail_surv_obs = np.maximum(
        _interp_survival_from_edges(target_grid.grid_edges, edge_survival_tilted, observed_t),
        1e-300,
    )
    tail_mean_obs = tail_num_obs / tail_surv_obs

    score_values = delta * direction_obs + (1.0 - delta) * tail_mean_obs - overall_mean
    return float(np.sum(score_values))


def _tmle_eif_values(
    eps: float,
    target_grid: RCTargetGrid,
    centered_direction_grid: np.ndarray,
    data: pd.DataFrame,
) -> np.ndarray:
    density_tilted, _, edge_survival_tilted, _ = _tilted_density_and_survival(
        target_grid, centered_direction_grid, eps
    )
    observed_t = data["T"].to_numpy(dtype=float)
    delta = data["Delta"].to_numpy(dtype=float)

    overall_mean = float(np.sum(centered_direction_grid * density_tilted * target_grid.delta_j))
    tail_weighted = np.cumsum(
        (centered_direction_grid * density_tilted * target_grid.delta_j)[::-1]
    )[::-1]
    edge_tail_weighted = np.concatenate((tail_weighted, [0.0]))

    direction_obs = _interp_direction(target_grid.grid_midpoints, centered_direction_grid, observed_t)
    tail_num_obs = _interp_survival_from_edges(target_grid.grid_edges, edge_tail_weighted, observed_t)
    tail_surv_obs = np.maximum(
        _interp_survival_from_edges(target_grid.grid_edges, edge_survival_tilted, observed_t),
        1e-300,
    )
    tail_mean_obs = tail_num_obs / tail_surv_obs
    return delta * direction_obs + (1.0 - delta) * tail_mean_obs - overall_mean


def _estimate_eic_variance(eic_values: np.ndarray) -> tuple[float, float]:
    eic_values = np.asarray(eic_values, dtype=float)
    n_obs = eic_values.shape[0]
    if n_obs <= 1:
        return 0.0, 0.0
    estimand_variance = float(np.var(eic_values, ddof=1) / n_obs)
    standard_error = float(np.sqrt(max(estimand_variance, 0.0)))
    return estimand_variance, standard_error


def _find_score_bracket(
    score_fn,
    start: float,
    growth: float,
    max_width: float,
) -> tuple[tuple[float, float] | None, list[tuple[float, float, float]]]:
    evaluations: list[tuple[float, float, float]] = []
    score_zero = float(score_fn(0.0))
    if np.isclose(score_zero, 0.0, atol=1e-10):
        evaluations.append((0.0, 0.0, score_zero))
        return (0.0, 0.0), evaluations

    width = float(start)
    while width <= max_width:
        left = -width
        right = width
        score_left = float(score_fn(left))
        score_right = float(score_fn(right))
        evaluations.extend([(left, width, score_left), (right, width, score_right)])
        if score_left * score_zero < 0.0:
            return (left, 0.0), evaluations
        if score_zero * score_right < 0.0:
            return (0.0, right), evaluations
        if score_left * score_right < 0.0:
            return (left, right), evaluations
        width *= growth
    return None, evaluations


def _solve_epsilon_t0(
    target_grid: RCTargetGrid,
    centered_direction_grid: np.ndarray,
    data: pd.DataFrame,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
) -> tuple[float, bool, dict[str, Any]]:
    score_fn = lambda eps: _tmle_score_t0(eps, target_grid, centered_direction_grid, data)
    objective_fn = lambda eps: _tmle_loglik_t0(eps, target_grid, centered_direction_grid, data)

    score_at_zero = float(score_fn(0.0))
    bracket, bracket_trace = _find_score_bracket(
        score_fn,
        start=eps_bracket_start,
        growth=eps_bracket_growth,
        max_width=eps_bracket_max,
    )

    if bracket is not None:
        if bracket[0] == bracket[1]:
            epsilon_hat = float(bracket[0])
        else:
            epsilon_hat = float(brentq(score_fn, bracket[0], bracket[1]))
        return epsilon_hat, True, {
            "method": "score_root",
            "score_at_zero": score_at_zero,
            "bracket": bracket,
            "bracket_trace": bracket_trace,
        }

    optimum = minimize_scalar(
        lambda eps: -objective_fn(eps),
        bounds=eps_fallback_bounds,
        method="bounded",
    )
    epsilon_hat = float(optimum.x)
    return epsilon_hat, bool(optimum.success), {
        "method": "bounded_fallback",
        "score_at_zero": score_at_zero,
        "fallback_bounds": eps_fallback_bounds,
        "bracket": None,
        "bracket_trace": bracket_trace,
    }


def _target_survival_t0(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    data: pd.DataFrame,
    t0: float,
    survival_clip: float,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
    compute_diagnostic_curves: bool,
) -> PointwiseTMLEResult:
    target_grid = _build_pointwise_target_grid(initial_fit, t0)
    raw_direction, centered_direction, direction_details = _compute_dfg_t0_on_grid(
        target_grid, censoring_cache, t0, survival_clip
    )
    psi_init = float(_evaluate_survival(target_grid, np.array([t0]), survival_clip)[0])
    epsilon_hat, converged, solver_details = _solve_epsilon_t0(
        target_grid,
        centered_direction,
        data,
        eps_bracket_start=eps_bracket_start,
        eps_bracket_growth=eps_bracket_growth,
        eps_bracket_max=eps_bracket_max,
        eps_fallback_bounds=eps_fallback_bounds,
    )
    density_star, survival_star, edge_survival_star, normalizer = _tilted_density_and_survival(
        target_grid, centered_direction, epsilon_hat
    )
    psi_star = float(
        _interp_survival_from_edges(target_grid.grid_edges, edge_survival_star, np.array([t0]))[0]
    )
    score_at_solution = _tmle_score_t0(epsilon_hat, target_grid, centered_direction, data)
    objective_at_solution = _tmle_loglik_t0(epsilon_hat, target_grid, centered_direction, data)
    eic_values = _tmle_eif_values(epsilon_hat, target_grid, centered_direction, data)
    estimand_variance, standard_error = _estimate_eic_variance(eic_values)
    ci_lower = float(max(0.0, psi_star - 1.96 * standard_error))
    ci_upper = float(min(1.0, psi_star + 1.96 * standard_error))
    if compute_diagnostic_curves:
        eps_window = max(1.0, abs(epsilon_hat) + 0.5)
        diagnostic_eps_grid = np.linspace(epsilon_hat - eps_window, epsilon_hat + eps_window, 161)
        diagnostic_loglik = np.array(
            [_tmle_loglik_t0(eps, target_grid, centered_direction, data) for eps in diagnostic_eps_grid],
            dtype=float,
        )
        diagnostic_score = np.array(
            [_tmle_score_t0(eps, target_grid, centered_direction, data) for eps in diagnostic_eps_grid],
            dtype=float,
        )
    else:
        diagnostic_eps_grid = np.empty(0, dtype=float)
        diagnostic_loglik = np.empty(0, dtype=float)
        diagnostic_score = np.empty(0, dtype=float)

    diagnostics = {
        "target_grid_edges": target_grid.grid_edges,
        "target_grid_midpoints": target_grid.grid_midpoints,
        "target_delta_j": target_grid.delta_j,
        "density_before_grid": target_grid.density_grid,
        "raw_direction_grid": raw_direction,
        "centered_direction_grid": centered_direction,
        "density_star_grid": density_star,
        "survival_star_grid": survival_star,
        "edge_survival_star_grid": edge_survival_star,
        "normalizer": normalizer,
        "direction_details": direction_details,
        "solver_details": solver_details,
        "eif_like_values": eic_values,
        "estimand_variance": estimand_variance,
        "standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "gbar_right_on_target_grid": _evaluate_gbar(censoring_cache, target_grid.grid_midpoints),
        "gbar_left_on_target_grid": _evaluate_gbar_left(censoring_cache, target_grid.grid_midpoints),
        "diagnostic_eps_grid": diagnostic_eps_grid,
        "diagnostic_loglik_values": diagnostic_loglik,
        "diagnostic_score_values": diagnostic_score,
    }
    return PointwiseTMLEResult(
        t0=float(t0),
        psi_init=psi_init,
        psi_star=psi_star,
        epsilon=epsilon_hat,
        estimand_variance=estimand_variance,
        standard_error=standard_error,
        score_at_zero=float(solver_details["score_at_zero"]),
        score_at_solution=float(score_at_solution),
        objective_at_solution=float(objective_at_solution),
        eif_mean=float(np.mean(eic_values)),
        solve_method=str(solver_details["method"]),
        converged=converged,
        diagnostics=diagnostics,
    )


def _serialize_initial_fit(initial_fit: RCInitialFit) -> dict[str, Any]:
    return {
        "estimator": initial_fit.estimator,
        "basis_order": initial_fit.basis_order,
        "grid_eval": initial_fit.grid_edges,
        "grid_midpoints": initial_fit.grid_midpoints,
        "delta_j": initial_fit.delta_j,
        "estimated_density": initial_fit.density_grid,
        "estimated_survival": initial_fit.survival_grid,
        "edge_survival": initial_fit.edge_survival,
        "theta_hat": initial_fit.theta_hat,
        "theta_selected": initial_fit.theta_hat,
        "grid_points_hal_selected": initial_fit.grid_points_hal_selected,
    }


def _serialize_censoring_cache(cache: RCCensoringCache) -> dict[str, Any]:
    return {
        "km": cache.km,
        "jump_times": cache.jump_times,
        "gbar_right": cache.gbar_right,
        "gbar_left": cache.gbar_left,
        "jump_masses": cache.jump_masses,
        "clip": cache.clip,
    }


def _package_pointwise_fit(
    result: PointwiseTMLEResult,
    theta_selected: np.ndarray,
    store_pointwise_arrays: bool,
) -> dict[str, Any]:
    fit = {
        "t0": result.t0,
        "targeting_points": np.asarray([result.t0], dtype=float),
        "theta_targeting": np.asarray([result.epsilon], dtype=float),
        "theta_selected": np.asarray(theta_selected, dtype=float).copy(),
        "epsilon": result.epsilon,
        "psi_init": result.psi_init,
        "psi_star": result.psi_star,
        "estimand_variance": result.estimand_variance,
        "standard_error": result.standard_error,
        "ci_lower": result.diagnostics["ci_lower"],
        "ci_upper": result.diagnostics["ci_upper"],
        "score_at_zero": result.score_at_zero,
        "score_at_solution": result.score_at_solution,
        "objective_at_solution": result.objective_at_solution,
        "eif_mean": result.eif_mean,
        "solve_method": result.solve_method,
        "converged": result.converged,
        "target_grid_augmented_with_t0": result.diagnostics["direction_details"][
            "target_grid_augmented_with_t0"
        ],
    }
    if store_pointwise_arrays:
        fit.update(
            {
                "grid_eval": result.diagnostics["target_grid_edges"],
                "grid_midpoints": result.diagnostics["target_grid_midpoints"],
                "delta_j": result.diagnostics["target_delta_j"],
                "estimated_density": result.diagnostics["density_star_grid"],
                "estimated_survival": result.diagnostics["survival_star_grid"],
                "targeted_survival_grid": result.diagnostics["survival_star_grid"],
                "target_grid_edges": result.diagnostics["target_grid_edges"],
                "target_grid_midpoints": result.diagnostics["target_grid_midpoints"],
                "target_delta_j": result.diagnostics["target_delta_j"],
                "density_before_grid": result.diagnostics["density_before_grid"],
                "raw_direction": result.diagnostics["raw_direction_grid"],
                "centered_direction": result.diagnostics["centered_direction_grid"],
                "eic_values": result.diagnostics["eif_like_values"],
                "diagnostics": result.diagnostics,
            }
        )
    return fit


def _state_to_target_grid(state: PointwiseTMLEState) -> RCTargetGrid:
    return RCTargetGrid(
        t0=float(state.t0),
        grid_edges=state.grid_edges.copy(),
        grid_midpoints=state.grid_midpoints.copy(),
        delta_j=state.delta_j.copy(),
        density_grid=state.density_grid.copy(),
        log_density_grid=state.log_density_grid.copy(),
        survival_grid=state.survival_grid.copy(),
        edge_survival=state.edge_survival.copy(),
        t0_inserted=bool(state.target_grid_augmented_with_t0),
    )


def _initialize_pointwise_state(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    t0: float,
) -> PointwiseTMLEState:
    target_grid = _build_pointwise_target_grid(initial_fit, float(t0))
    psi_init = float(
        _interp_survival_from_edges(target_grid.grid_edges, target_grid.edge_survival, np.array([t0]))[0]
    )
    return PointwiseTMLEState(
        t0=float(t0),
        grid_edges=target_grid.grid_edges.copy(),
        grid_midpoints=target_grid.grid_midpoints.copy(),
        delta_j=target_grid.delta_j.copy(),
        log_density_grid=target_grid.log_density_grid.copy(),
        density_grid=target_grid.density_grid.copy(),
        survival_grid=target_grid.survival_grid.copy(),
        edge_survival=target_grid.edge_survival.copy(),
        psi_current=psi_init,
        target_grid_augmented_with_t0=bool(target_grid.t0_inserted),
        censoring_cache=censoring_cache,
    )


def _evaluate_pointwise_state(
    state: PointwiseTMLEState,
    observed_data: pd.DataFrame,
    survival_clip: float,
) -> tuple[PointwiseTMLEState, dict[str, Any]]:
    target_grid = _state_to_target_grid(state)
    raw_direction, centered_direction, direction_details = _compute_dfg_t0_on_grid(
        target_grid,
        state.censoring_cache,
        state.t0,
        survival_clip,
    )
    eif_values = _tmle_eif_values(0.0, target_grid, centered_direction, observed_data)
    eif_mean = float(np.mean(eif_values))
    eif_sigma = float(np.std(eif_values, ddof=1)) if eif_values.size > 1 else 0.0
    score_at_zero = float(_tmle_score_t0(0.0, target_grid, centered_direction, observed_data))
    psi_current = float(
        _interp_survival_from_edges(state.grid_edges, state.edge_survival, np.array([state.t0]))[0]
    )
    evaluated_state = PointwiseTMLEState(
        t0=state.t0,
        grid_edges=state.grid_edges.copy(),
        grid_midpoints=state.grid_midpoints.copy(),
        delta_j=state.delta_j.copy(),
        log_density_grid=state.log_density_grid.copy(),
        density_grid=state.density_grid.copy(),
        survival_grid=state.survival_grid.copy(),
        edge_survival=state.edge_survival.copy(),
        psi_current=psi_current,
        target_grid_augmented_with_t0=state.target_grid_augmented_with_t0,
        censoring_cache=state.censoring_cache,
        raw_direction=raw_direction,
        centered_direction=centered_direction,
        eif_values=eif_values,
        eif_mean=eif_mean,
        eif_sigma=eif_sigma,
        score_at_zero=score_at_zero,
    )
    return evaluated_state, {"direction_details": direction_details}


def _one_local_tmle_update(
    current_state: PointwiseTMLEState,
    observed_data: pd.DataFrame,
    *,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
) -> tuple[PointwiseTMLEState, dict[str, Any]]:
    if current_state.centered_direction is None:
        raise ValueError("current_state must be evaluated before calling _one_local_tmle_update.")

    target_grid = _state_to_target_grid(current_state)
    epsilon_hat, converged_inner, solver_details = _solve_epsilon_t0(
        target_grid,
        current_state.centered_direction,
        observed_data,
        eps_bracket_start=eps_bracket_start,
        eps_bracket_growth=eps_bracket_growth,
        eps_bracket_max=eps_bracket_max,
        eps_fallback_bounds=eps_fallback_bounds,
    )
    score_at_solution = float(
        _tmle_score_t0(epsilon_hat, target_grid, current_state.centered_direction, observed_data)
    )
    objective_at_zero = float(
        _tmle_loglik_t0(0.0, target_grid, current_state.centered_direction, observed_data)
    )
    objective_at_solution = float(
        _tmle_loglik_t0(epsilon_hat, target_grid, current_state.centered_direction, observed_data)
    )
    solve_method = str(solver_details["method"])

    finite_update = bool(np.all(np.isfinite([epsilon_hat, score_at_solution, objective_at_solution])))
    accepted_update = bool(finite_update and converged_inner)
    if solve_method == "bounded_fallback":
        score_improved = abs(score_at_solution) <= 0.9 * max(abs(current_state.score_at_zero), 1e-12)
        objective_improved = objective_at_solution >= objective_at_zero - 1e-10
        accepted_update = bool(accepted_update and score_improved and objective_improved)

    if not accepted_update:
        return current_state, {
            "epsilon": float(epsilon_hat),
            "score_at_solution": score_at_solution,
            "objective_at_solution": objective_at_solution,
            "solve_method": solve_method,
            "converged_inner": bool(converged_inner),
            "accepted_update": False,
        }

    density_star, survival_star, edge_survival_star, _ = _tilted_density_and_survival(
        target_grid,
        current_state.centered_direction,
        epsilon_hat,
    )
    updated_state = PointwiseTMLEState(
        t0=current_state.t0,
        grid_edges=current_state.grid_edges.copy(),
        grid_midpoints=current_state.grid_midpoints.copy(),
        delta_j=current_state.delta_j.copy(),
        log_density_grid=np.log(np.clip(density_star, 1e-300, None)),
        density_grid=density_star,
        survival_grid=survival_star,
        edge_survival=edge_survival_star,
        psi_current=float(
            _interp_survival_from_edges(
                current_state.grid_edges, edge_survival_star, np.array([current_state.t0])
            )[0]
        ),
        target_grid_augmented_with_t0=current_state.target_grid_augmented_with_t0,
        censoring_cache=current_state.censoring_cache,
    )
    return updated_state, {
        "epsilon": float(epsilon_hat),
        "score_at_solution": score_at_solution,
        "objective_at_solution": objective_at_solution,
        "solve_method": solve_method,
        "converged_inner": bool(converged_inner),
        "accepted_update": True,
    }


def _iterate_tmle_t0(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    observed_data: pd.DataFrame,
    t0: float,
    *,
    survival_clip: float,
    max_iter: int,
    min_abs_eps: float,
    min_score_tol: float,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
) -> dict[str, Any]:
    state = _initialize_pointwise_state(initial_fit, censoring_cache, t0)
    history_rows: list[dict[str, Any]] = []
    n_obs = observed_data.shape[0]
    stop_reason = "max_iter"

    for iteration in range(max_iter + 1):
        state, _ = _evaluate_pointwise_state(state, observed_data, survival_clip)
        tolerance = float(max(min_score_tol, state.eif_sigma / (np.sqrt(n_obs) * np.log(n_obs))))
        row = {
            "iteration": iteration,
            "psi": state.psi_current,
            "eif_mean": state.eif_mean,
            "sigma": state.eif_sigma,
            "stop_tolerance": tolerance,
            "score_at_zero": state.score_at_zero,
            "epsilon": float("nan"),
            "score_at_solution": float("nan"),
            "objective_at_solution": float("nan"),
            "solve_method": None,
            "converged_inner": None,
            "accepted_update": False,
            "status": "evaluated",
        }
        if abs(state.eif_mean) <= tolerance:
            row["status"] = "score_tolerance"
            history_rows.append(row)
            stop_reason = "score_tolerance"
            break
        if iteration == max_iter:
            row["status"] = "max_iter"
            history_rows.append(row)
            stop_reason = "max_iter"
            break

        updated_state, update_info = _one_local_tmle_update(
            state,
            observed_data,
            eps_bracket_start=eps_bracket_start,
            eps_bracket_growth=eps_bracket_growth,
            eps_bracket_max=eps_bracket_max,
            eps_fallback_bounds=eps_fallback_bounds,
        )
        row.update(update_info)
        history_rows.append(row)

        if not update_info["accepted_update"]:
            history_rows[-1]["status"] = "solver_failure"
            stop_reason = "solver_failure"
            break

        state = updated_state
        if abs(float(update_info["epsilon"])) < min_abs_eps:
            history_rows[-1]["status"] = "epsilon_tiny"
            stop_reason = "epsilon_tiny"
            break

    final_state, _ = _evaluate_pointwise_state(state, observed_data, survival_clip)
    estimand_variance, standard_error = _estimate_eic_variance(final_state.eif_values)
    history = pd.DataFrame(history_rows)
    epsilon_path = (
        history.loc[history["accepted_update"], "epsilon"].to_numpy(dtype=float).tolist()
        if len(history) > 0
        else []
    )
    return {
        "summary": {
            "t0": float(t0),
            "psi_star": float(final_state.psi_current),
            "eif_mean": float(final_state.eif_mean),
            "estimand_variance": float(estimand_variance),
            "standard_error": float(standard_error),
            "score_at_zero": float(final_state.score_at_zero),
            "n_iterations": int(np.sum(history["accepted_update"])) if len(history) > 0 else 0,
            "stop_reason": stop_reason,
            "epsilon_path": epsilon_path,
            "eif_mean_path": history["eif_mean"].to_numpy(dtype=float).tolist() if len(history) > 0 else [float(final_state.eif_mean)],
            "sigma_path": history["sigma"].to_numpy(dtype=float).tolist() if len(history) > 0 else [float(final_state.eif_sigma)],
            "psi_path": history["psi"].to_numpy(dtype=float).tolist() if len(history) > 0 else [float(final_state.psi_current)],
        },
        "history": history,
        "final_state": final_state,
    }


class RightCensoredSurvivalTargetLearner:
    """Pointwise HAL-TMLE for right-censored survival targets."""

    def __init__(
        self,
        *,
        clip: float = 1e-6,
        survival_clip: float = 1e-8,
        eps_bracket_start: float = 0.25,
        eps_bracket_growth: float = 2.0,
        eps_bracket_max: float = 16.0,
        eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
        store_pointwise_arrays: bool = False,
    ) -> None:
        self.clip = float(clip)
        self.survival_clip = float(survival_clip)
        self.eps_bracket_start = float(eps_bracket_start)
        self.eps_bracket_growth = float(eps_bracket_growth)
        self.eps_bracket_max = float(eps_bracket_max)
        self.eps_fallback_bounds = tuple(float(x) for x in eps_fallback_bounds)
        self.store_pointwise_arrays = bool(store_pointwise_arrays)

    def run_m_step(
        self,
        initial_estimator: RightCensoredInitEstimator,
        observed_data: pd.DataFrame,
        targeting_points: Union[float, Sequence[float], np.ndarray],
        *,
        km: Optional[KaplanMeier] = None,
        store_pointwise_arrays: Optional[bool] = None,
    ) -> dict[str, Any]:
        if not isinstance(observed_data, pd.DataFrame):
            raise TypeError("observed_data must be a pandas DataFrame.")
        required_cols = {"T", "Delta"}
        missing = required_cols.difference(observed_data.columns)
        if missing:
            raise ValueError(f"observed_data is missing required columns: {sorted(missing)}")

        targeting_points_arr = _normalize_targeting_points(targeting_points)
        if km is None:
            km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")

        initial_fit = _build_initial_fit(initial_estimator)
        censoring_cache = _build_censoring_cache(km, clip=self.clip)
        store_arrays = (
            self.store_pointwise_arrays
            if store_pointwise_arrays is None
            else bool(store_pointwise_arrays)
        )

        results = [
            _target_survival_t0(
                initial_fit,
                censoring_cache,
                observed_data,
                float(t0),
                survival_clip=self.survival_clip,
                eps_bracket_start=self.eps_bracket_start,
                eps_bracket_growth=self.eps_bracket_growth,
                eps_bracket_max=self.eps_bracket_max,
                eps_fallback_bounds=self.eps_fallback_bounds,
                compute_diagnostic_curves=store_arrays,
            )
            for t0 in targeting_points_arr
        ]

        summary = pd.DataFrame(
            {
                "t0": [r.t0 for r in results],
                "psi_init": [r.psi_init for r in results],
                "psi_star": [r.psi_star for r in results],
                "epsilon": [r.epsilon for r in results],
                "estimand_variance": [r.estimand_variance for r in results],
                "standard_error": [r.standard_error for r in results],
                "ci_lower": [r.diagnostics["ci_lower"] for r in results],
                "ci_upper": [r.diagnostics["ci_upper"] for r in results],
                "score_at_zero": [r.score_at_zero for r in results],
                "score_at_solution": [r.score_at_solution for r in results],
                "objective_at_solution": [r.objective_at_solution for r in results],
                "eif_mean": [r.eif_mean for r in results],
                "solve_method": [r.solve_method for r in results],
                "converged": [r.converged for r in results],
                "target_grid_augmented_with_t0": [
                    r.diagnostics["direction_details"]["target_grid_augmented_with_t0"]
                    for r in results
                ],
            }
        )

        return {
            "targeting_points": targeting_points_arr,
            "summary": summary,
            "pointwise_fits": [
                _package_pointwise_fit(
                    r,
                    theta_selected=initial_fit.theta_hat,
                    store_pointwise_arrays=store_arrays,
                )
                for r in results
            ],
            "initial_fit": _serialize_initial_fit(initial_fit),
            "censoring_cache": _serialize_censoring_cache(censoring_cache),
            "metadata": {
                "n_targets": int(targeting_points_arr.size),
                "n_observations": int(observed_data.shape[0]),
                "clip": self.clip,
                "survival_clip": self.survival_clip,
                "eps_bracket_start": self.eps_bracket_start,
                "eps_bracket_growth": self.eps_bracket_growth,
                "eps_bracket_max": self.eps_bracket_max,
                "eps_fallback_bounds": self.eps_fallback_bounds,
                "store_pointwise_arrays": store_arrays,
                "pointwise_design": True,
            },
        }

    def get_estimand_variance(
        self,
        targeted_fit: dict[str, Any],
        observed_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> np.ndarray:
        _ = observed_data
        _ = kwargs
        summary = targeted_fit["summary"]
        if not isinstance(summary, pd.DataFrame) or "estimand_variance" not in summary.columns:
            raise ValueError("targeted_fit does not contain a valid summary with estimand_variance.")
        return summary["estimand_variance"].to_numpy(dtype=float)


def right_censored_survival_targeting_M_step(
    initial_estimator: RightCensoredInitEstimator,
    observed_data: pd.DataFrame,
    targeting_points: Union[float, Sequence[float], np.ndarray],
    *,
    km: Optional[KaplanMeier] = None,
    clip: float = 1e-6,
    survival_clip: float = 1e-8,
    eps_bracket_start: float = 0.25,
    eps_bracket_growth: float = 2.0,
    eps_bracket_max: float = 16.0,
    eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
    store_pointwise_arrays: bool = False,
) -> dict[str, Any]:
    learner = RightCensoredSurvivalTargetLearner(
        clip=clip,
        survival_clip=survival_clip,
        eps_bracket_start=eps_bracket_start,
        eps_bracket_growth=eps_bracket_growth,
        eps_bracket_max=eps_bracket_max,
        eps_fallback_bounds=eps_fallback_bounds,
        store_pointwise_arrays=store_pointwise_arrays,
    )
    return learner.run_m_step(
        initial_estimator=initial_estimator,
        observed_data=observed_data,
        targeting_points=targeting_points,
        km=km,
    )


def right_censored_survival_estimand_variance(
    targeted_fit: dict[str, Any],
    observed_data: Optional[pd.DataFrame] = None,
    targeting_points: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    **kwargs,
) -> np.ndarray:
    _ = targeting_points
    learner = RightCensoredSurvivalTargetLearner(**kwargs)
    return learner.get_estimand_variance(targeted_fit=targeted_fit, observed_data=observed_data)


def right_censored_survival_targeting_M_step_v2(
    initial_estimator: RightCensoredInitEstimator,
    observed_data: pd.DataFrame,
    targeting_points: Union[float, Sequence[float], np.ndarray],
    *,
    km: Optional[KaplanMeier] = None,
    mode: str = "auto",
    one_step_eif_gate: float = 1e-8,
    clip: float = 1e-6,
    survival_clip: float = 1e-8,
    eps_bracket_start: float = 0.25,
    eps_bracket_growth: float = 2.0,
    eps_bracket_max: float = 16.0,
    eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
    max_iter: int = 25,
    min_abs_eps: float = 1e-10,
    min_score_tol: float = 1e-8,
    store_pointwise_arrays: bool = False,
) -> dict[str, Any]:
    if mode not in {"auto", "one_step", "iterative"}:
        raise ValueError("mode must be one of {'auto', 'one_step', 'iterative'}.")
    if not isinstance(observed_data, pd.DataFrame):
        raise TypeError("observed_data must be a pandas DataFrame.")
    required_cols = {"T", "Delta"}
    missing = required_cols.difference(observed_data.columns)
    if missing:
        raise ValueError(f"observed_data is missing required columns: {sorted(missing)}")

    targeting_points_arr = _normalize_targeting_points(targeting_points)
    if km is None:
        km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")

    initial_fit = _build_initial_fit(initial_estimator)
    censoring_cache = _build_censoring_cache(km, clip=clip)
    one_step_results = [
        _target_survival_t0(
            initial_fit,
            censoring_cache,
            observed_data,
            float(t0),
            survival_clip=survival_clip,
            eps_bracket_start=eps_bracket_start,
            eps_bracket_growth=eps_bracket_growth,
            eps_bracket_max=eps_bracket_max,
            eps_fallback_bounds=eps_fallback_bounds,
            compute_diagnostic_curves=store_pointwise_arrays,
        )
        for t0 in targeting_points_arr
    ]

    summary_rows: list[dict[str, Any]] = []
    pointwise_fits: list[dict[str, Any]] = []
    for one_step in one_step_results:
        init_state = _initialize_pointwise_state(initial_fit, censoring_cache, one_step.t0)
        init_state, _ = _evaluate_pointwise_state(init_state, observed_data, survival_clip)
        init_var, init_se = _estimate_eic_variance(init_state.eif_values)
        one_step_abs_eif = float(abs(one_step.eif_mean))

        if mode == "one_step":
            use_iterative = False
            decision_reason = "mode_one_step"
        elif mode == "iterative":
            use_iterative = True
            decision_reason = "mode_iterative"
        else:
            use_iterative = bool(one_step_abs_eif > one_step_eif_gate)
            decision_reason = "one_step_eif_above_gate" if use_iterative else "one_step_eif_below_gate"

        iter_bundle = None
        if use_iterative:
            iter_bundle = _iterate_tmle_t0(
                initial_fit=initial_fit,
                censoring_cache=censoring_cache,
                observed_data=observed_data,
                t0=one_step.t0,
                survival_clip=survival_clip,
                max_iter=max_iter,
                min_abs_eps=min_abs_eps,
                min_score_tol=min_score_tol,
                eps_bracket_start=eps_bracket_start,
                eps_bracket_growth=eps_bracket_growth,
                eps_bracket_max=eps_bracket_max,
                eps_fallback_bounds=eps_fallback_bounds,
            )
            final_summary = iter_bundle["summary"]
            final_state: Optional[PointwiseTMLEState] = iter_bundle["final_state"]
            history_df: pd.DataFrame = iter_bundle["history"]
            final_psi = float(final_summary["psi_star"])
            final_eif_mean = float(final_summary["eif_mean"])
            final_var = float(final_summary["estimand_variance"])
            final_se = float(final_summary["standard_error"])
            final_score_at_zero = float(final_summary["score_at_zero"])
            final_stop_reason = str(final_summary["stop_reason"])
            final_n_iterations = int(final_summary["n_iterations"])
            final_epsilon = (
                float(final_summary["epsilon_path"][-1]) if len(final_summary["epsilon_path"]) > 0 else 0.0
            )
            final_score_at_solution = (
                float(history_df.loc[history_df["accepted_update"], "score_at_solution"].iloc[-1])
                if np.any(history_df["accepted_update"])
                else float("nan")
            )
            final_objective_at_solution = (
                float(history_df.loc[history_df["accepted_update"], "objective_at_solution"].iloc[-1])
                if np.any(history_df["accepted_update"])
                else float("nan")
            )
            final_solve_method = (
                str(history_df.loc[history_df["accepted_update"], "solve_method"].iloc[-1])
                if np.any(history_df["accepted_update"])
                else "none"
            )
        else:
            final_state = None
            history_df = pd.DataFrame()
            final_psi = float(one_step.psi_star)
            final_eif_mean = float(one_step.eif_mean)
            final_var = float(one_step.estimand_variance)
            final_se = float(one_step.standard_error)
            final_score_at_zero = float(one_step.score_at_zero)
            final_stop_reason = "one_step_gate" if mode == "auto" else decision_reason
            final_n_iterations = 0
            final_epsilon = float(one_step.epsilon)
            final_score_at_solution = float(one_step.score_at_solution)
            final_objective_at_solution = float(one_step.objective_at_solution)
            final_solve_method = str(one_step.solve_method)

        ci_lower = float(max(0.0, final_psi - 1.96 * final_se))
        ci_upper = float(min(1.0, final_psi + 1.96 * final_se))
        summary_rows.append(
            {
                "t0": float(one_step.t0),
                "psi_init": float(one_step.psi_init),
                "psi_star": final_psi,
                "epsilon": final_epsilon,
                "estimand_variance": final_var,
                "standard_error": final_se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "score_at_zero": final_score_at_zero,
                "score_at_solution": final_score_at_solution,
                "objective_at_solution": final_objective_at_solution,
                "eif_mean": final_eif_mean,
                "solve_method": final_solve_method,
                "converged": bool(final_stop_reason != "solver_failure"),
                "target_grid_augmented_with_t0": bool(one_step.diagnostics["direction_details"]["target_grid_augmented_with_t0"]),
                "psi_initial_stage": float(init_state.psi_current),
                "eif_mean_initial_stage": float(init_state.eif_mean),
                "estimand_variance_initial_stage": float(init_var),
                "standard_error_initial_stage": float(init_se),
                "psi_one_step": float(one_step.psi_star),
                "eif_mean_one_step": float(one_step.eif_mean),
                "estimand_variance_one_step": float(one_step.estimand_variance),
                "standard_error_one_step": float(one_step.standard_error),
                "score_at_solution_one_step": float(one_step.score_at_solution),
                "psi_final": final_psi,
                "eif_mean_final": final_eif_mean,
                "estimand_variance_final": final_var,
                "standard_error_final": final_se,
                "used_iterative": bool(use_iterative),
                "decision_reason": decision_reason,
                "one_step_abs_eif": one_step_abs_eif,
                "one_step_eif_gate": float(one_step_eif_gate),
                "n_iterations": final_n_iterations,
                "stop_reason": final_stop_reason,
            }
        )

        fit_row: dict[str, Any] = {
            "t0": float(one_step.t0),
            "targeting_points": np.asarray([one_step.t0], dtype=float),
            "theta_targeting": np.asarray([final_epsilon], dtype=float),
            "theta_selected": np.asarray(initial_fit.theta_hat, dtype=float).copy(),
            "epsilon": final_epsilon,
            "psi_init": float(one_step.psi_init),
            "psi_star": final_psi,
            "estimand_variance": final_var,
            "standard_error": final_se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "score_at_zero": final_score_at_zero,
            "score_at_solution": final_score_at_solution,
            "objective_at_solution": final_objective_at_solution,
            "eif_mean": final_eif_mean,
            "solve_method": final_solve_method,
            "converged": bool(final_stop_reason != "solver_failure"),
            "target_grid_augmented_with_t0": bool(one_step.diagnostics["direction_details"]["target_grid_augmented_with_t0"]),
            "used_iterative": bool(use_iterative),
            "decision_reason": decision_reason,
            "one_step_abs_eif": one_step_abs_eif,
            "one_step_eif_gate": float(one_step_eif_gate),
            "n_iterations": final_n_iterations,
            "stop_reason": final_stop_reason,
            "initial_stage": {
                "psi": float(init_state.psi_current),
                "eif_mean": float(init_state.eif_mean),
                "estimand_variance": float(init_var),
                "standard_error": float(init_se),
            },
            "one_step_stage": {
                "psi": float(one_step.psi_star),
                "epsilon": float(one_step.epsilon),
                "eif_mean": float(one_step.eif_mean),
                "estimand_variance": float(one_step.estimand_variance),
                "standard_error": float(one_step.standard_error),
                "score_at_solution": float(one_step.score_at_solution),
            },
            "final_stage": {
                "psi": final_psi,
                "eif_mean": final_eif_mean,
                "estimand_variance": final_var,
                "standard_error": final_se,
            },
            "iteration_history": history_df.to_dict("records") if len(history_df) > 0 else [],
            "epsilon_path": (
                iter_bundle["summary"]["epsilon_path"] if iter_bundle is not None else []
            ),
            "eif_mean_path": (
                iter_bundle["summary"]["eif_mean_path"] if iter_bundle is not None else [float(one_step.eif_mean)]
            ),
            "sigma_path": (
                iter_bundle["summary"]["sigma_path"] if iter_bundle is not None else []
            ),
            "psi_path": (
                iter_bundle["summary"]["psi_path"] if iter_bundle is not None else [float(one_step.psi_star)]
            ),
        }

        if store_pointwise_arrays:
            if use_iterative and final_state is not None:
                fit_row.update(
                    {
                        "grid_eval": final_state.grid_edges.copy(),
                        "grid_midpoints": final_state.grid_midpoints.copy(),
                        "delta_j": final_state.delta_j.copy(),
                        "estimated_density": final_state.density_grid.copy(),
                        "estimated_survival": final_state.survival_grid.copy(),
                        "targeted_survival_grid": final_state.survival_grid.copy(),
                        "target_grid_edges": final_state.grid_edges.copy(),
                        "target_grid_midpoints": final_state.grid_midpoints.copy(),
                        "target_delta_j": final_state.delta_j.copy(),
                        "density_before_grid": one_step.diagnostics["density_before_grid"],
                        "raw_direction": final_state.raw_direction.copy() if final_state.raw_direction is not None else np.empty(0, dtype=float),
                        "centered_direction": final_state.centered_direction.copy() if final_state.centered_direction is not None else np.empty(0, dtype=float),
                        "eic_values": final_state.eif_values.copy() if final_state.eif_values is not None else np.empty(0, dtype=float),
                    }
                )
            else:
                fit_row.update(
                    {
                        "grid_eval": one_step.diagnostics["target_grid_edges"],
                        "grid_midpoints": one_step.diagnostics["target_grid_midpoints"],
                        "delta_j": one_step.diagnostics["target_delta_j"],
                        "estimated_density": one_step.diagnostics["density_star_grid"],
                        "estimated_survival": one_step.diagnostics["survival_star_grid"],
                        "targeted_survival_grid": one_step.diagnostics["survival_star_grid"],
                        "target_grid_edges": one_step.diagnostics["target_grid_edges"],
                        "target_grid_midpoints": one_step.diagnostics["target_grid_midpoints"],
                        "target_delta_j": one_step.diagnostics["target_delta_j"],
                        "density_before_grid": one_step.diagnostics["density_before_grid"],
                        "raw_direction": one_step.diagnostics["raw_direction_grid"],
                        "centered_direction": one_step.diagnostics["centered_direction_grid"],
                        "eic_values": one_step.diagnostics["eif_like_values"],
                        "diagnostics": one_step.diagnostics,
                    }
                )

        pointwise_fits.append(fit_row)

    summary = pd.DataFrame(summary_rows).sort_values("t0").reset_index(drop=True)
    return {
        "targeting_points": targeting_points_arr,
        "summary": summary,
        "pointwise_fits": pointwise_fits,
        "initial_fit": _serialize_initial_fit(initial_fit),
        "censoring_cache": _serialize_censoring_cache(censoring_cache),
        "metadata": {
            "api_version": "v2",
            "mode": mode,
            "one_step_eif_gate": float(one_step_eif_gate),
            "n_targets": int(targeting_points_arr.size),
            "n_observations": int(observed_data.shape[0]),
            "clip": float(clip),
            "survival_clip": float(survival_clip),
            "eps_bracket_start": float(eps_bracket_start),
            "eps_bracket_growth": float(eps_bracket_growth),
            "eps_bracket_max": float(eps_bracket_max),
            "eps_fallback_bounds": tuple(float(x) for x in eps_fallback_bounds),
            "max_iter": int(max_iter),
            "min_abs_eps": float(min_abs_eps),
            "min_score_tol": float(min_score_tol),
            "store_pointwise_arrays": bool(store_pointwise_arrays),
            "pointwise_design": True,
            "decision_policy": "run one-step first; iterate only when needed",
        },
    }
