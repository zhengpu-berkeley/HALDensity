from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _raw_gbar(cache: Any, t: np.ndarray | float) -> np.ndarray:
    return np.asarray(cache.km.predict(t), dtype=float)


def _clip_active_eic_mask(
    *,
    target_grid: Any,
    censoring_cache: Any,
    observed_data: pd.DataFrame,
    clip: float,
) -> dict[str, Any]:
    clip = float(clip)
    observed_t = observed_data["T"].to_numpy(dtype=float)
    delta = observed_data["Delta"].to_numpy(dtype=int)
    grid = np.asarray(target_grid.grid_midpoints, dtype=float)

    observed_gbar = _raw_gbar(censoring_cache, observed_t)
    grid_gbar = _raw_gbar(censoring_cache, grid)
    # Flag exactly the regions where the targeter replaces raw KM Gbar by the
    # computational clip. For typical sample sizes this mostly means Gbarhat=0.
    direction_grid_mask = grid_gbar < clip

    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    jump_gbar = np.asarray(censoring_cache.gbar_right, dtype=float)
    bad_jump_times = jump_times[jump_gbar < clip]
    if bad_jump_times.size > 0:
        direction_grid_mask = direction_grid_mask | np.any(
            grid[:, None] >= bad_jump_times[None, :],
            axis=1,
        )

    if bad_jump_times.size > 0:
        event_jump_mask = np.any(observed_t[:, None] >= bad_jump_times[None, :], axis=1)
    else:
        event_jump_mask = np.zeros_like(observed_t, dtype=bool)
    right_idx = np.searchsorted(grid, observed_t, side="left")
    right_idx = np.clip(right_idx, 0, len(grid) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(grid) - 1)
    event_direction_mask = direction_grid_mask[left_idx] | direction_grid_mask[right_idx]
    event_mask = (delta == 1) & ((observed_gbar < clip) | event_jump_mask | event_direction_mask)

    tail_mask = np.array(
        [bool(np.any(direction_grid_mask & (grid > t_i))) for t_i in observed_t],
        dtype=bool,
    )
    censored_mask = (delta == 0) & tail_mask
    mask = event_mask | censored_mask

    return {
        "mask": mask,
        "observed_gbar": observed_gbar,
        "direction_grid_mask": direction_grid_mask,
        "bad_jump_times": bad_jump_times,
        "clip": clip,
    }


def _estimate_positivity_filtered_eic_variance(
    *,
    eic_values: np.ndarray,
    target_grid: Any,
    censoring_cache: Any,
    observed_data: pd.DataFrame,
    clip: float,
) -> dict[str, Any]:
    eic_values = np.asarray(eic_values, dtype=float)
    mask_info = _clip_active_eic_mask(
        target_grid=target_grid,
        censoring_cache=censoring_cache,
        observed_data=observed_data,
        clip=clip,
    )
    positivity_mask = np.asarray(mask_info["mask"], dtype=bool)
    finite_mask = np.isfinite(eic_values)
    include_mask = finite_mask & ~positivity_mask
    filtered_values = eic_values[include_mask]
    n_included = int(filtered_values.size)

    if n_included <= 1:
        estimand_variance = float("nan")
        standard_error = float("nan")
    else:
        estimand_variance = float(np.var(filtered_values, ddof=1) / n_included)
        standard_error = float(np.sqrt(max(estimand_variance, 0.0)))

    n_total = int(eic_values.size)
    n_excluded = int(np.sum(positivity_mask | ~finite_mask))
    return {
        "estimand_variance": estimand_variance,
        "standard_error": standard_error,
        "included_count": n_included,
        "excluded_count": n_excluded,
        "excluded_rate": float(n_excluded / n_total) if n_total > 0 else float("nan"),
        "positivity_mask": positivity_mask,
        "include_mask": include_mask,
        "observed_gbar": np.asarray(mask_info["observed_gbar"], dtype=float),
        "direction_grid_mask": np.asarray(mask_info["direction_grid_mask"], dtype=bool),
        "bad_jump_times": np.asarray(mask_info["bad_jump_times"], dtype=float),
        "clip": float(clip),
    }


def _build_clipped_wald_interval_allow_nan(center: float, standard_error: float) -> tuple[float, float]:
    if not np.isfinite(standard_error):
        return float("nan"), float("nan")
    ci_lower = float(max(0.0, center - 1.96 * standard_error))
    ci_upper = float(min(1.0, center + 1.96 * standard_error))
    return ci_lower, ci_upper


def _build_unbounded_wald_interval_allow_nan(center: float, standard_error: float) -> tuple[float, float]:
    if not np.isfinite(standard_error):
        return float("nan"), float("nan")
    return float(center - 1.96 * standard_error), float(center + 1.96 * standard_error)
