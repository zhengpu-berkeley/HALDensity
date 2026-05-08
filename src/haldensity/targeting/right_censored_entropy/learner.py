from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from haldensity.censoring.right.estimators import RightCensoredInitEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.targeting._right_censored_eic import (
    _build_unbounded_wald_interval_allow_nan,
    _estimate_positivity_filtered_eic_variance,
)
from haldensity.targeting.right_censored_survival.learner import (
    RCCensoringCache,
    RCInitialFit,
    RCTargetGrid,
    _build_censoring_cache,
    _build_initial_fit,
    _center_direction_on_grid,
    _compute_targeting_gbar_floor,
    _estimate_eic_variance,
    _estimate_eic_variance_from_sigma,
    _evaluate_targeting_gbar,
    _interp_survival_from_edges,
    _iterative_tolerance_from_sigma,
    _select_variance_column,
    _serialize_censoring_cache,
    _serialize_initial_fit,
    _solve_epsilon_t0,
    _tilted_density_and_survival,
    _tmle_eif_values,
    _tmle_loglik_t0,
    _tmle_score_t0,
)


@dataclass
class EntropyTMLEState:
    grid_edges: np.ndarray
    grid_midpoints: np.ndarray
    delta_j: np.ndarray
    log_density_grid: np.ndarray
    density_grid: np.ndarray
    survival_grid: np.ndarray
    edge_survival: np.ndarray
    psi_current: float
    censoring_cache: RCCensoringCache
    raw_direction: Optional[np.ndarray] = None
    centered_direction: Optional[np.ndarray] = None
    eif_values: Optional[np.ndarray] = None
    eif_mean: float = float("nan")
    eif_sigma: float = float("nan")
    score_at_zero: float = float("nan")


def _compute_entropy_from_grid(density_grid: np.ndarray, delta_j: np.ndarray) -> float:
    density = np.clip(np.asarray(density_grid, dtype=float), 1e-300, None)
    return float(np.sum((-density * np.log(density)) * np.asarray(delta_j, dtype=float)))


def _state_to_target_grid(state: EntropyTMLEState) -> RCTargetGrid:
    return RCTargetGrid(
        t0=float("nan"),
        grid_edges=state.grid_edges.copy(),
        grid_midpoints=state.grid_midpoints.copy(),
        delta_j=state.delta_j.copy(),
        density_grid=state.density_grid.copy(),
        log_density_grid=state.log_density_grid.copy(),
        survival_grid=state.survival_grid.copy(),
        edge_survival=state.edge_survival.copy(),
        t0_inserted=False,
    )


def _initial_fit_to_target_grid(initial_fit: RCInitialFit) -> RCTargetGrid:
    return RCTargetGrid(
        t0=float("nan"),
        grid_edges=initial_fit.grid_edges.copy(),
        grid_midpoints=initial_fit.grid_midpoints.copy(),
        delta_j=initial_fit.delta_j.copy(),
        density_grid=initial_fit.density_grid.copy(),
        log_density_grid=initial_fit.log_density_grid.copy(),
        survival_grid=initial_fit.survival_grid.copy(),
        edge_survival=initial_fit.edge_survival.copy(),
        t0_inserted=False,
    )


def _evaluate_tail_entropy_score(
    target_grid: RCTargetGrid,
    points: np.ndarray,
    survival_clip: float,
) -> np.ndarray:
    phi_grid = -np.log(np.clip(target_grid.density_grid, 1e-300, None))
    tail_weighted = np.cumsum((phi_grid * target_grid.density_grid * target_grid.delta_j)[::-1])[::-1]
    edge_tail_weighted = np.concatenate((tail_weighted, [0.0]))
    tail_num = _interp_survival_from_edges(
        target_grid.grid_edges,
        edge_tail_weighted,
        np.asarray(points, dtype=float),
    )
    tail_surv = np.maximum(
        _interp_survival_from_edges(
            target_grid.grid_edges,
            target_grid.edge_survival,
            np.asarray(points, dtype=float),
        ),
        survival_clip,
    )
    return tail_num / tail_surv


def _compute_entropy_direction_on_grid(
    target_grid: RCTargetGrid,
    censoring_cache: RCCensoringCache,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    psi = _compute_entropy_from_grid(target_grid.density_grid, target_grid.delta_j)
    effective_gbar_floor = float(max(targeting_gbar_floor, censoring_cache.clip))
    phi_grid = -np.log(np.clip(target_grid.density_grid, 1e-300, None))
    gbar_grid = _evaluate_targeting_gbar(
        censoring_cache,
        target_grid.grid_midpoints,
        targeting_gbar_floor=effective_gbar_floor,
    )
    first_term = phi_grid / gbar_grid

    jump_times = censoring_cache.jump_times
    jump_masses = censoring_cache.jump_masses
    if jump_times.size == 0:
        tail_entropy_jump = np.empty(0, dtype=float)
        gbar_right_u = np.empty(0, dtype=float)
        increments = np.empty(0, dtype=float)
        cumulative_jump_term = np.zeros_like(target_grid.grid_midpoints)
    else:
        tail_entropy_jump = _evaluate_tail_entropy_score(
            target_grid,
            jump_times,
            survival_clip,
        )
        gbar_right_u = _evaluate_targeting_gbar(
            censoring_cache,
            jump_times,
            targeting_gbar_floor=effective_gbar_floor,
        )
        increments = tail_entropy_jump * jump_masses / np.square(gbar_right_u)
        cumulative_increments = np.cumsum(increments)
        cutoff_idx = np.searchsorted(jump_times, target_grid.grid_midpoints, side="right") - 1
        cumulative_jump_term = np.where(
            cutoff_idx >= 0,
            cumulative_increments[np.clip(cutoff_idx, 0, len(cumulative_increments) - 1)],
            0.0,
        )

    raw_direction = first_term - cumulative_jump_term
    centered_direction, raw_mean = _center_direction_on_grid(
        raw_direction,
        target_grid.density_grid,
        target_grid.delta_j,
    )
    direction_details = {
        "psi": psi,
        "gbar_convention": "right_continuous",
        "stage1_ipcw_convention": "repo_default",
        "targeting_gbar_floor": effective_gbar_floor,
        "jump_times": jump_times,
        "jump_masses": jump_masses,
        "gbar_jump_denom": (
            np.square(gbar_right_u) if jump_times.size > 0 else np.empty(0, dtype=float)
        ),
        "tail_entropy_jump": tail_entropy_jump,
        "jump_increments": increments,
        "first_term_grid": first_term,
        "cumulative_jump_term_grid": cumulative_jump_term,
        "raw_mean": raw_mean,
    }
    return raw_direction, centered_direction, direction_details


def _initialize_entropy_state(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
) -> EntropyTMLEState:
    target_grid = _initial_fit_to_target_grid(initial_fit)
    psi_init = _compute_entropy_from_grid(target_grid.density_grid, target_grid.delta_j)
    return EntropyTMLEState(
        grid_edges=target_grid.grid_edges.copy(),
        grid_midpoints=target_grid.grid_midpoints.copy(),
        delta_j=target_grid.delta_j.copy(),
        log_density_grid=target_grid.log_density_grid.copy(),
        density_grid=target_grid.density_grid.copy(),
        survival_grid=target_grid.survival_grid.copy(),
        edge_survival=target_grid.edge_survival.copy(),
        psi_current=psi_init,
        censoring_cache=censoring_cache,
    )


def _evaluate_entropy_state(
    state: EntropyTMLEState,
    observed_data: pd.DataFrame,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> tuple[EntropyTMLEState, dict[str, Any]]:
    target_grid = _state_to_target_grid(state)
    raw_direction, centered_direction, direction_details = _compute_entropy_direction_on_grid(
        target_grid,
        state.censoring_cache,
        survival_clip,
        targeting_gbar_floor,
    )
    eif_values = _tmle_eif_values(0.0, target_grid, centered_direction, observed_data)
    eif_mean = float(np.mean(eif_values))
    eif_sigma = float(np.std(eif_values, ddof=1)) if eif_values.size > 1 else 0.0
    score_at_zero = float(_tmle_score_t0(0.0, target_grid, centered_direction, observed_data))
    psi_current = _compute_entropy_from_grid(state.density_grid, state.delta_j)
    evaluated_state = EntropyTMLEState(
        grid_edges=state.grid_edges.copy(),
        grid_midpoints=state.grid_midpoints.copy(),
        delta_j=state.delta_j.copy(),
        log_density_grid=state.log_density_grid.copy(),
        density_grid=state.density_grid.copy(),
        survival_grid=state.survival_grid.copy(),
        edge_survival=state.edge_survival.copy(),
        psi_current=psi_current,
        censoring_cache=state.censoring_cache,
        raw_direction=raw_direction,
        centered_direction=centered_direction,
        eif_values=eif_values,
        eif_mean=eif_mean,
        eif_sigma=eif_sigma,
        score_at_zero=score_at_zero,
    )
    return evaluated_state, {"direction_details": direction_details}


def _one_entropy_tmle_update(
    current_state: EntropyTMLEState,
    observed_data: pd.DataFrame,
    *,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
) -> tuple[EntropyTMLEState, dict[str, Any]]:
    if current_state.centered_direction is None:
        raise ValueError("current_state must be evaluated before calling _one_entropy_tmle_update.")

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
    psi_star = _compute_entropy_from_grid(density_star, current_state.delta_j)
    updated_state = EntropyTMLEState(
        grid_edges=current_state.grid_edges.copy(),
        grid_midpoints=current_state.grid_midpoints.copy(),
        delta_j=current_state.delta_j.copy(),
        log_density_grid=np.log(np.clip(density_star, 1e-300, None)),
        density_grid=density_star,
        survival_grid=survival_star,
        edge_survival=edge_survival_star,
        psi_current=psi_star,
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


def _iterate_entropy_tmle(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    observed_data: pd.DataFrame,
    *,
    survival_clip: float,
    targeting_gbar_floor: float,
    max_iter: int,
    min_abs_eps: float,
    min_score_tol: float,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
) -> dict[str, Any]:
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1 for iterative TMLE.")

    state = _initialize_entropy_state(initial_fit, censoring_cache)
    psi_init = float(state.psi_current)
    history_rows: list[dict[str, Any]] = []
    n_obs = observed_data.shape[0]
    stop_reason = "max_iter"

    for iteration in range(1, max_iter + 1):
        state, _ = _evaluate_entropy_state(
            state,
            observed_data,
            survival_clip,
            targeting_gbar_floor,
        )
        updated_state, update_info = _one_entropy_tmle_update(
            state,
            observed_data,
            eps_bracket_start=eps_bracket_start,
            eps_bracket_growth=eps_bracket_growth,
            eps_bracket_max=eps_bracket_max,
            eps_fallback_bounds=eps_fallback_bounds,
        )
        row = {
            "iteration": iteration,
            "psi_before_update": state.psi_current,
            "eif_mean_before_update": state.eif_mean,
            "sigma_before_update": state.eif_sigma,
            "score_at_zero_before_update": state.score_at_zero,
            "epsilon": float("nan"),
            "score_at_solution": float("nan"),
            "objective_at_solution": float("nan"),
            "solve_method": None,
            "converged_inner": None,
            "accepted_update": False,
            "psi": state.psi_current,
            "eif_mean": state.eif_mean,
            "sigma": state.eif_sigma,
            "sigma_over_sqrtnlogn": float("nan"),
            "stop_tolerance": float("nan"),
            "score_at_zero": state.score_at_zero,
            "status": "evaluated",
        }
        row.update(update_info)

        if not update_info["accepted_update"]:
            row["status"] = "solver_failure"
            history_rows.append(row)
            stop_reason = "solver_failure"
            break

        state = updated_state
        state, _ = _evaluate_entropy_state(
            state,
            observed_data,
            survival_clip,
            targeting_gbar_floor,
        )
        raw_tolerance, tolerance = _iterative_tolerance_from_sigma(
            state.eif_sigma,
            n_obs,
            min_score_tol,
        )
        row.update(
            {
                "psi": state.psi_current,
                "eif_mean": state.eif_mean,
                "sigma": state.eif_sigma,
                "sigma_over_sqrtnlogn": raw_tolerance,
                "stop_tolerance": tolerance,
                "score_at_zero": state.score_at_zero,
            }
        )

        if abs(float(update_info["epsilon"])) < min_abs_eps:
            row["status"] = "epsilon_tiny"
            history_rows.append(row)
            stop_reason = "epsilon_tiny"
            break
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

        row["status"] = "continue"
        history_rows.append(row)

    final_state = state
    if final_state.eif_values is None:
        raise RuntimeError("Iterative entropy TMLE final state is missing EIC values.")
    estimand_variance, standard_error = _estimate_eic_variance(final_state.eif_values)
    history = pd.DataFrame(history_rows)
    epsilon_path = (
        history.loc[history["accepted_update"], "epsilon"].to_numpy(dtype=float).tolist()
        if len(history) > 0
        else []
    )
    score_at_solution_path = (
        history.loc[history["accepted_update"], "score_at_solution"].to_numpy(dtype=float).tolist()
        if len(history) > 0
        else []
    )
    objective_at_solution_path = (
        history.loc[history["accepted_update"], "objective_at_solution"].to_numpy(dtype=float).tolist()
        if len(history) > 0
        else []
    )
    solve_method_path = (
        history.loc[history["accepted_update"], "solve_method"].astype(str).tolist()
        if len(history) > 0
        else []
    )
    raw_tolerance_final, threshold_final = _iterative_tolerance_from_sigma(
        final_state.eif_sigma,
        n_obs,
        min_score_tol,
    )
    return {
        "summary": {
            "psi_init": psi_init,
            "psi_star": float(final_state.psi_current),
            "eif_mean": float(final_state.eif_mean),
            "estimand_variance": float(estimand_variance),
            "standard_error": float(standard_error),
            "score_at_zero": float(final_state.score_at_zero),
            "score_at_solution_last": (
                float(score_at_solution_path[-1]) if len(score_at_solution_path) > 0 else float("nan")
            ),
            "objective_at_solution_last": (
                float(objective_at_solution_path[-1])
                if len(objective_at_solution_path) > 0
                else float("nan")
            ),
            "solve_method_last": solve_method_path[-1] if len(solve_method_path) > 0 else "none",
            "n_iterations": int(np.sum(history["accepted_update"])) if len(history) > 0 else 0,
            "stop_reason": stop_reason,
            "epsilon_path": epsilon_path,
            "epsilon_last": float(epsilon_path[-1]) if len(epsilon_path) > 0 else float("nan"),
            "eif_mean_path": history["eif_mean"].to_numpy(dtype=float).tolist()
            if len(history) > 0
            else [float(final_state.eif_mean)],
            "sigma_path": history["sigma"].to_numpy(dtype=float).tolist()
            if len(history) > 0
            else [float(final_state.eif_sigma)],
            "psi_path": [psi_init]
            + (
                history["psi"].to_numpy(dtype=float).tolist()
                if len(history) > 0
                else [float(final_state.psi_current)]
            ),
            "sigma_final": float(final_state.eif_sigma),
            "sigma_over_sqrtnlogn_final": float(raw_tolerance_final),
            "threshold_final": float(threshold_final),
            "n_observations": int(n_obs),
        },
        "history": history,
        "final_state": final_state,
    }


def _summarize_entropy_tmle_step(
    iter_bundle: dict[str, Any],
    step: int,
    *,
    min_score_tol: float,
) -> dict[str, Any]:
    history = iter_bundle["history"]
    step_row = history.loc[history["iteration"] == int(step)]
    if len(step_row) != 1:
        raise ValueError(f"Expected exactly one row for step={step}, found {len(step_row)}.")

    row = step_row.iloc[0]
    n_obs = int(iter_bundle["summary"]["n_observations"])
    raw_tolerance, threshold = _iterative_tolerance_from_sigma(
        float(row["sigma"]),
        n_obs,
        min_score_tol,
    )
    estimand_variance, standard_error = _estimate_eic_variance_from_sigma(
        float(row["sigma"]),
        n_obs,
    )
    return {
        "step": int(step),
        "psi_star": float(row["psi"]),
        "eif_mean": float(row["eif_mean"]),
        "sigma": float(row["sigma"]),
        "estimand_variance": float(estimand_variance),
        "standard_error": float(standard_error),
        "sigma_over_sqrtnlogn": float(raw_tolerance),
        "threshold": float(threshold),
        "epsilon": float(row["epsilon"]) if np.isfinite(row["epsilon"]) else float("nan"),
        "score_at_zero": float(row["score_at_zero"]),
        "score_at_solution": (
            float(row["score_at_solution"]) if np.isfinite(row["score_at_solution"]) else float("nan")
        ),
        "objective_at_solution": (
            float(row["objective_at_solution"])
            if np.isfinite(row["objective_at_solution"])
            else float("nan")
        ),
        "solve_method": str(row["solve_method"]) if row["solve_method"] is not None else "none",
        "status": str(row["status"]),
    }


def _validate_observed_data(observed_data: pd.DataFrame) -> None:
    if not isinstance(observed_data, pd.DataFrame):
        raise TypeError("observed_data must be a pandas DataFrame.")
    required_cols = {"T", "Delta"}
    missing = required_cols.difference(observed_data.columns)
    if missing:
        raise ValueError(f"observed_data is missing required columns: {sorted(missing)}")


class RightCensoredEntropyTargetLearner:
    """HAL-TMLE targeter for event-time entropy under right censoring."""

    def __init__(
        self,
        *,
        clip: float = 1e-6,
        survival_clip: float = 1e-8,
        eps_bracket_start: float = 0.25,
        eps_bracket_growth: float = 2.0,
        eps_bracket_max: float = 16.0,
        eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
        max_iter: int = 25,
        min_abs_eps: float = 1e-10,
        min_score_tol: float = 1e-8,
        one_step_eif_gate: float = 1e-8,
        store_pointwise_arrays: bool = False,
        targeting_gbar_floor_scale: Optional[float] = 1.0,
    ) -> None:
        self.clip = float(clip)
        self.survival_clip = float(survival_clip)
        self.eps_bracket_start = float(eps_bracket_start)
        self.eps_bracket_growth = float(eps_bracket_growth)
        self.eps_bracket_max = float(eps_bracket_max)
        self.eps_fallback_bounds = (
            float(eps_fallback_bounds[0]),
            float(eps_fallback_bounds[1]),
        )
        self.max_iter = int(max_iter)
        self.min_abs_eps = float(min_abs_eps)
        self.min_score_tol = float(min_score_tol)
        self.one_step_eif_gate = float(one_step_eif_gate)
        self.store_pointwise_arrays = bool(store_pointwise_arrays)
        self.targeting_gbar_floor_scale = (
            None
            if targeting_gbar_floor_scale is None
            else float(targeting_gbar_floor_scale)
        )

    def run_m_step(
        self,
        initial_estimator: RightCensoredInitEstimator,
        observed_data: pd.DataFrame,
        *,
        km: Optional[KaplanMeier] = None,
        mode: str = "iterative",
        store_pointwise_arrays: Optional[bool] = None,
    ) -> dict[str, Any]:
        if mode not in {"auto", "one_step", "iterative"}:
            raise ValueError("mode must be one of {'auto', 'one_step', 'iterative'}.")
        _validate_observed_data(observed_data)
        if km is None:
            km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")

        initial_fit = _build_initial_fit(initial_estimator)
        censoring_cache = _build_censoring_cache(km, clip=self.clip)
        n_obs = int(observed_data.shape[0])
        targeting_gbar_floor = _compute_targeting_gbar_floor(
            n_obs=n_obs,
            scale=self.targeting_gbar_floor_scale,
            hard_min=self.clip,
        )
        store_arrays = (
            self.store_pointwise_arrays
            if store_pointwise_arrays is None
            else bool(store_pointwise_arrays)
        )

        init_state = _initialize_entropy_state(initial_fit, censoring_cache)
        init_state, _ = _evaluate_entropy_state(
            init_state,
            observed_data,
            self.survival_clip,
            targeting_gbar_floor,
        )
        if init_state.eif_values is None:
            raise RuntimeError("Initial entropy TMLE state is missing EIC values.")
        init_var_all_eic, init_se_all_eic = _estimate_eic_variance(init_state.eif_values)
        raw_init_tolerance, threshold_initial = _iterative_tolerance_from_sigma(
            init_state.eif_sigma,
            n_obs,
            self.min_score_tol,
        )
        passes_initial_threshold = bool(abs(float(init_state.eif_mean)) <= threshold_initial)
        init_filtered_eic = _estimate_positivity_filtered_eic_variance(
            eic_values=init_state.eif_values,
            target_grid=_state_to_target_grid(init_state),
            censoring_cache=censoring_cache,
            observed_data=observed_data,
            clip=targeting_gbar_floor,
        )
        init_var = float(init_filtered_eic["estimand_variance"])
        init_se = float(init_filtered_eic["standard_error"])

        if mode == "auto" and passes_initial_threshold:
            history_df = pd.DataFrame()
            final_summary = None
            final_state = init_state
            final_psi = float(init_state.psi_current)
            final_eif_mean = float(init_state.eif_mean)
            final_var_all_eic = float(init_var_all_eic)
            final_se_all_eic = float(init_se_all_eic)
            final_filtered_eic = init_filtered_eic
            final_var = float(init_var)
            final_se = float(init_se)
            final_stop_reason = "initial_score_tolerance"
            final_n_iterations = 0
            final_epsilon = 0.0
            final_score_at_solution = float(init_state.score_at_zero)
            final_objective_at_solution = float(
                _tmle_loglik_t0(
                    0.0,
                    _state_to_target_grid(init_state),
                    init_state.centered_direction,
                    observed_data,
                )
            )
            final_solve_method = "skipped_initial_gate"
            final_sigma = float(init_state.eif_sigma)
            final_raw_tolerance = float(raw_init_tolerance)
            final_threshold = float(threshold_initial)
            final_epsilon_path: list[float] = []
            final_eif_mean_path: list[float] = [float(init_state.eif_mean)]
            final_sigma_path: list[float] = [float(init_state.eif_sigma)]
            final_psi_path: list[float] = [float(init_state.psi_current)]
            one_step_stage = {
                "psi_star": float(init_state.psi_current),
                "eif_mean": float(init_state.eif_mean),
                "sigma": float(init_state.eif_sigma),
                "estimand_variance": float(init_var),
                "standard_error": float(init_se),
                "estimand_variance_all_eic": float(init_var_all_eic),
                "standard_error_all_eic": float(init_se_all_eic),
                "sigma_over_sqrtnlogn": float(raw_init_tolerance),
                "threshold": float(threshold_initial),
                "epsilon": 0.0,
                "score_at_zero": float(init_state.score_at_zero),
                "score_at_solution": float(init_state.score_at_zero),
                "objective_at_solution": float(final_objective_at_solution),
                "solve_method": "skipped_initial_gate",
                "status": "skipped_initial_gate",
            }
        else:
            one_step_bundle = _iterate_entropy_tmle(
                initial_fit=initial_fit,
                censoring_cache=censoring_cache,
                observed_data=observed_data,
                survival_clip=self.survival_clip,
                targeting_gbar_floor=targeting_gbar_floor,
                max_iter=1,
                min_abs_eps=self.min_abs_eps,
                min_score_tol=self.min_score_tol,
                eps_bracket_start=self.eps_bracket_start,
                eps_bracket_growth=self.eps_bracket_growth,
                eps_bracket_max=self.eps_bracket_max,
                eps_fallback_bounds=self.eps_fallback_bounds,
            )
            one_step_stage = _summarize_entropy_tmle_step(
                one_step_bundle,
                step=1,
                min_score_tol=self.min_score_tol,
            )
            one_step_state: EntropyTMLEState = one_step_bundle["final_state"]
            if one_step_state.eif_values is None:
                raise RuntimeError("One-step entropy TMLE state is missing EIC values.")
            one_step_var_all_eic = float(one_step_stage["estimand_variance"])
            one_step_se_all_eic = float(one_step_stage["standard_error"])
            one_step_filtered_eic = _estimate_positivity_filtered_eic_variance(
                eic_values=one_step_state.eif_values,
                target_grid=_state_to_target_grid(one_step_state),
                censoring_cache=censoring_cache,
                observed_data=observed_data,
                clip=targeting_gbar_floor,
            )
            one_step_stage["estimand_variance_all_eic"] = one_step_var_all_eic
            one_step_stage["standard_error_all_eic"] = one_step_se_all_eic
            one_step_stage["estimand_variance"] = float(one_step_filtered_eic["estimand_variance"])
            one_step_stage["standard_error"] = float(one_step_filtered_eic["standard_error"])
            run_full_iterative = mode == "iterative" or mode == "auto"
            if mode == "one_step" or not run_full_iterative:
                final_bundle = one_step_bundle
            else:
                final_bundle = _iterate_entropy_tmle(
                    initial_fit=initial_fit,
                    censoring_cache=censoring_cache,
                    observed_data=observed_data,
                    survival_clip=self.survival_clip,
                    targeting_gbar_floor=targeting_gbar_floor,
                    max_iter=self.max_iter,
                    min_abs_eps=self.min_abs_eps,
                    min_score_tol=self.min_score_tol,
                    eps_bracket_start=self.eps_bracket_start,
                    eps_bracket_growth=self.eps_bracket_growth,
                    eps_bracket_max=self.eps_bracket_max,
                    eps_fallback_bounds=self.eps_fallback_bounds,
                )

            history_df = final_bundle["history"]
            final_summary = final_bundle["summary"]
            final_state = final_bundle["final_state"]
            if final_state.eif_values is None:
                raise RuntimeError("Final entropy TMLE state is missing EIC values.")
            final_psi = float(final_summary["psi_star"])
            final_eif_mean = float(final_summary["eif_mean"])
            final_var_all_eic = float(final_summary["estimand_variance"])
            final_se_all_eic = float(final_summary["standard_error"])
            final_filtered_eic = _estimate_positivity_filtered_eic_variance(
                eic_values=final_state.eif_values,
                target_grid=_state_to_target_grid(final_state),
                censoring_cache=censoring_cache,
                observed_data=observed_data,
                clip=targeting_gbar_floor,
            )
            final_var = float(final_filtered_eic["estimand_variance"])
            final_se = float(final_filtered_eic["standard_error"])
            final_stop_reason = str(final_summary["stop_reason"])
            final_n_iterations = int(final_summary["n_iterations"])
            final_epsilon = float(final_summary["epsilon_last"])
            final_score_at_solution = float(final_summary["score_at_solution_last"])
            final_objective_at_solution = float(final_summary["objective_at_solution_last"])
            final_solve_method = str(final_summary["solve_method_last"])
            final_sigma = float(final_summary["sigma_final"])
            final_raw_tolerance = float(final_summary["sigma_over_sqrtnlogn_final"])
            final_threshold = float(final_summary["threshold_final"])
            final_epsilon_path = final_summary["epsilon_path"]
            final_eif_mean_path = final_summary["eif_mean_path"]
            final_sigma_path = final_summary["sigma_path"]
            final_psi_path = final_summary["psi_path"]

        continued_past_one_step = bool(final_n_iterations > 1)
        ci_lower, ci_upper = _build_unbounded_wald_interval_allow_nan(final_psi, init_se)
        summary = pd.DataFrame(
            [
                {
                    "target": "entropy",
                    "psi_init": float(init_state.psi_current),
                    "psi_star": final_psi,
                    "entropy_init": float(init_state.psi_current),
                    "entropy_star": final_psi,
                    "entropy_one_step": float(one_step_stage["psi_star"]),
                    "entropy_final": final_psi,
                    "epsilon": final_epsilon,
                    "estimand_variance": float(init_var),
                    "standard_error": float(init_se),
                    "estimand_variance_all_eic": float(init_var_all_eic),
                    "standard_error_all_eic": float(init_se_all_eic),
                    "gbar_clip": float(self.clip),
                    "targeting_gbar_floor": float(targeting_gbar_floor),
                    "targeting_gbar_floor_scale": (
                        float(self.targeting_gbar_floor_scale)
                        if self.targeting_gbar_floor_scale is not None
                        else float("nan")
                    ),
                    "clip_active_eic_included_count": int(init_filtered_eic["included_count"]),
                    "clip_active_eic_excluded_count": int(init_filtered_eic["excluded_count"]),
                    "clip_active_eic_excluded_rate": float(init_filtered_eic["excluded_rate"]),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "score_at_zero": float(init_state.score_at_zero if final_summary is None else final_summary["score_at_zero"]),
                    "score_at_solution": final_score_at_solution,
                    "objective_at_solution": final_objective_at_solution,
                    "eif_mean": final_eif_mean,
                    "solve_method": final_solve_method,
                    "converged": bool(final_stop_reason != "solver_failure"),
                    "psi_initial_stage": float(init_state.psi_current),
                    "eif_mean_initial_stage": float(init_state.eif_mean),
                    "sigma_initial_stage": float(init_state.eif_sigma),
                    "sigma_over_sqrtnlogn_initial": float(raw_init_tolerance),
                    "threshold_initial": float(threshold_initial),
                    "passes_initial_threshold": passes_initial_threshold,
                    "estimand_variance_initial_stage": float(init_var),
                    "standard_error_initial_stage": float(init_se),
                    "estimand_variance_all_eic_initial_stage": float(init_var_all_eic),
                    "standard_error_all_eic_initial_stage": float(init_se_all_eic),
                    "psi_one_step": float(one_step_stage["psi_star"]),
                    "eif_mean_one_step": float(one_step_stage["eif_mean"]),
                    "sigma_one_step": float(one_step_stage["sigma"]),
                    "sigma_over_sqrtnlogn_one_step": float(one_step_stage["sigma_over_sqrtnlogn"]),
                    "threshold_one_step": float(one_step_stage["threshold"]),
                    "epsilon_one_step": float(one_step_stage["epsilon"]),
                    "estimand_variance_one_step": float(one_step_stage["estimand_variance"]),
                    "standard_error_one_step": float(one_step_stage["standard_error"]),
                    "estimand_variance_all_eic_one_step": float(one_step_stage["estimand_variance_all_eic"]),
                    "standard_error_all_eic_one_step": float(one_step_stage["standard_error_all_eic"]),
                    "score_at_zero_one_step": float(one_step_stage["score_at_zero"]),
                    "score_at_solution_one_step": float(one_step_stage["score_at_solution"]),
                    "status_one_step": str(one_step_stage["status"]),
                    "psi_final": final_psi,
                    "eif_mean_final": final_eif_mean,
                    "sigma_final": float(final_sigma),
                    "sigma_over_sqrtnlogn_final": float(final_raw_tolerance),
                    "threshold_final": float(final_threshold),
                    "estimand_variance_final": float(final_var),
                    "standard_error_final": float(final_se),
                    "estimand_variance_all_eic_final": float(final_var_all_eic),
                    "standard_error_all_eic_final": float(final_se_all_eic),
                    "clip_active_eic_included_count_final": int(final_filtered_eic["included_count"]),
                    "clip_active_eic_excluded_count_final": int(final_filtered_eic["excluded_count"]),
                    "clip_active_eic_excluded_rate_final": float(final_filtered_eic["excluded_rate"]),
                    "epsilon_last": final_epsilon,
                    "continued_past_one_step": continued_past_one_step,
                    "used_iterative": continued_past_one_step,
                    "n_iterations": final_n_iterations,
                    "stop_reason": final_stop_reason,
                }
            ]
        )

        fit_row: dict[str, Any] = {
            "target": "entropy",
            "theta_targeting": np.asarray(final_epsilon_path, dtype=float),
            "theta_selected": np.asarray(initial_fit.theta_hat, dtype=float).copy(),
            "epsilon": final_epsilon,
            "psi_init": float(init_state.psi_current),
            "psi_star": final_psi,
            "entropy_init": float(init_state.psi_current),
            "entropy_star": final_psi,
            "entropy_one_step": float(one_step_stage["psi_star"]),
            "entropy_final": final_psi,
            "estimand_variance": float(init_var),
            "standard_error": float(init_se),
            "estimand_variance_all_eic": float(init_var_all_eic),
            "standard_error_all_eic": float(init_se_all_eic),
            "gbar_clip": float(self.clip),
            "targeting_gbar_floor": float(targeting_gbar_floor),
            "targeting_gbar_floor_scale": (
                float(self.targeting_gbar_floor_scale)
                if self.targeting_gbar_floor_scale is not None
                else float("nan")
            ),
            "clip_active_eic_included_count": int(init_filtered_eic["included_count"]),
            "clip_active_eic_excluded_count": int(init_filtered_eic["excluded_count"]),
            "clip_active_eic_excluded_rate": float(init_filtered_eic["excluded_rate"]),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "score_at_zero": float(init_state.score_at_zero if final_summary is None else final_summary["score_at_zero"]),
            "score_at_solution": final_score_at_solution,
            "objective_at_solution": final_objective_at_solution,
            "eif_mean": final_eif_mean,
            "solve_method": final_solve_method,
            "converged": bool(final_stop_reason != "solver_failure"),
            "continued_past_one_step": continued_past_one_step,
            "used_iterative": continued_past_one_step,
            "n_iterations": final_n_iterations,
            "stop_reason": final_stop_reason,
            "initial_stage": {
                "psi": float(init_state.psi_current),
                "eif_mean": float(init_state.eif_mean),
                "sigma": float(init_state.eif_sigma),
                "sigma_over_sqrtnlogn": float(raw_init_tolerance),
                "threshold": float(threshold_initial),
                "passes_threshold": passes_initial_threshold,
                "estimand_variance": float(init_var),
                "standard_error": float(init_se),
                "estimand_variance_all_eic": float(init_var_all_eic),
                "standard_error_all_eic": float(init_se_all_eic),
            },
            "one_step_stage": {
                "psi": float(one_step_stage["psi_star"]),
                "epsilon": float(one_step_stage["epsilon"]),
                "eif_mean": float(one_step_stage["eif_mean"]),
                "sigma": float(one_step_stage["sigma"]),
                "sigma_over_sqrtnlogn": float(one_step_stage["sigma_over_sqrtnlogn"]),
                "threshold": float(one_step_stage["threshold"]),
                "estimand_variance": float(one_step_stage["estimand_variance"]),
                "standard_error": float(one_step_stage["standard_error"]),
                "estimand_variance_all_eic": float(one_step_stage["estimand_variance_all_eic"]),
                "standard_error_all_eic": float(one_step_stage["standard_error_all_eic"]),
                "score_at_zero": float(one_step_stage["score_at_zero"]),
                "score_at_solution": float(one_step_stage["score_at_solution"]),
                "status": str(one_step_stage["status"]),
            },
            "final_stage": {
                "psi": final_psi,
                "eif_mean": final_eif_mean,
                "sigma": float(final_sigma),
                "sigma_over_sqrtnlogn": float(final_raw_tolerance),
                "threshold": float(final_threshold),
                "epsilon_last": final_epsilon,
                "estimand_variance": float(final_var),
                "standard_error": float(final_se),
                "estimand_variance_all_eic": float(final_var_all_eic),
                "standard_error_all_eic": float(final_se_all_eic),
            },
            "iteration_history": history_df.to_dict("records") if len(history_df) > 0 else [],
            "epsilon_path": final_epsilon_path,
            "eif_mean_path": final_eif_mean_path,
            "sigma_path": final_sigma_path,
            "psi_path": final_psi_path,
        }

        if store_arrays:
            fit_row.update(
                {
                    "grid_eval": final_state.grid_edges.copy(),
                    "grid_midpoints": final_state.grid_midpoints.copy(),
                    "delta_j": final_state.delta_j.copy(),
                    "estimated_density": final_state.density_grid.copy(),
                    "estimated_survival": final_state.survival_grid.copy(),
                    "targeted_survival_grid": final_state.survival_grid.copy(),
                    "density_before_grid": init_state.density_grid.copy(),
                    "raw_direction": final_state.raw_direction.copy()
                    if final_state.raw_direction is not None
                    else np.empty(0, dtype=float),
                    "centered_direction": final_state.centered_direction.copy()
                    if final_state.centered_direction is not None
                    else np.empty(0, dtype=float),
                    "eic_values": final_state.eif_values.copy()
                    if final_state.eif_values is not None
                    else np.empty(0, dtype=float),
                    "clip_active_eic_mask": final_filtered_eic["positivity_mask"].copy(),
                    "clip_active_eic_include_mask": final_filtered_eic["include_mask"].copy(),
                    "observed_gbar": final_filtered_eic["observed_gbar"].copy(),
                    "direction_grid_clip_active_mask": final_filtered_eic["direction_grid_mask"].copy(),
                }
            )

        return {
            "summary": summary,
            "pointwise_fits": [fit_row],
            "initial_fit": _serialize_initial_fit(initial_fit),
            "censoring_cache": _serialize_censoring_cache(censoring_cache),
            "metadata": {
                "api_version": "entropy_v1",
                "target": "entropy",
                "mode": mode,
                "one_step_eif_gate": self.one_step_eif_gate,
                "n_targets": 1,
                "n_observations": int(observed_data.shape[0]),
                "clip": self.clip,
                "survival_clip": self.survival_clip,
                "targeting_gbar_floor": float(targeting_gbar_floor),
                "targeting_gbar_floor_scale": self.targeting_gbar_floor_scale,
                "eps_bracket_start": self.eps_bracket_start,
                "eps_bracket_growth": self.eps_bracket_growth,
                "eps_bracket_max": self.eps_bracket_max,
                "eps_fallback_bounds": self.eps_fallback_bounds,
                "max_iter": self.max_iter,
                "min_abs_eps": self.min_abs_eps,
                "min_score_tol": self.min_score_tol,
                "gbar_clip": self.clip,
                "store_pointwise_arrays": store_arrays,
                "support": (0.0, 1.0),
                "decision_policy": "check the initial EIF score first; target only when the initial score exceeds sigma/(sqrt(n) log n)",
            },
        }

    def get_estimand_variance(
        self,
        targeted_fit: dict[str, Any],
        observed_data: Optional[pd.DataFrame] = None,
        which: str = "default",
    ) -> np.ndarray:
        _ = observed_data
        summary = targeted_fit["summary"]
        column = _select_variance_column(summary, which=which)
        return summary[column].to_numpy(dtype=float)


def right_censored_entropy_targeting_M_step(
    initial_estimator: RightCensoredInitEstimator,
    observed_data: pd.DataFrame,
    *,
    km: Optional[KaplanMeier] = None,
    mode: str = "iterative",
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
    targeting_gbar_floor_scale: Optional[float] = 1.0,
) -> dict[str, Any]:
    learner = RightCensoredEntropyTargetLearner(
        clip=clip,
        survival_clip=survival_clip,
        eps_bracket_start=eps_bracket_start,
        eps_bracket_growth=eps_bracket_growth,
        eps_bracket_max=eps_bracket_max,
        eps_fallback_bounds=eps_fallback_bounds,
        max_iter=max_iter,
        min_abs_eps=min_abs_eps,
        min_score_tol=min_score_tol,
        one_step_eif_gate=one_step_eif_gate,
        store_pointwise_arrays=store_pointwise_arrays,
        targeting_gbar_floor_scale=targeting_gbar_floor_scale,
    )
    return learner.run_m_step(
        initial_estimator=initial_estimator,
        observed_data=observed_data,
        km=km,
        mode=mode,
    )


def right_censored_entropy_estimand_variance(
    targeted_fit: dict[str, Any],
    observed_data: Optional[pd.DataFrame] = None,
    which: str = "default",
    **kwargs: Any,
) -> np.ndarray:
    learner = RightCensoredEntropyTargetLearner(**kwargs)
    return learner.get_estimand_variance(
        targeted_fit=targeted_fit,
        observed_data=observed_data,
        which=which,
    )
