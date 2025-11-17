from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple


def build_hal_design(
    values: Sequence[float] | np.ndarray,
    knots: Sequence[float] | np.ndarray,
    order: int,
    include_intercept: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Construct HAL basis (intercept + polynomials + truncated terms)."""
    if order < 0:
        raise ValueError("basis order must be >= 0")

    x = np.asarray(values, dtype=float).reshape(-1)
    knots_arr = np.asarray(knots, dtype=float).reshape(-1)
    n = x.size

    basis_list: list[np.ndarray] = []
    basis_names: list[str] = []

    if include_intercept:
        basis_list.append(np.ones(n, dtype=float))
        basis_names.append("Intercept")

    if order == 0:
        if knots_arr.size > 0:
            ramps = np.maximum(x[:, None] - knots_arr[None, :], 0.0)
            for idx, knot in enumerate(knots_arr):
                basis_list.append(ramps[:, idx])
                basis_names.append(f"(x - {knot:.6f})_+")
        return np.column_stack(basis_list), basis_names

    for power in range(1, order + 1):
        basis_list.append((x ** power).astype(float))
        basis_names.append(f"x^{power}")

    if knots_arr.size > 0:
        truncated = np.maximum(x[:, None] - knots_arr[None, :], 0.0) ** order
        for idx, knot in enumerate(knots_arr):
            basis_list.append(truncated[:, idx])
            basis_names.append(f"(x - {knot:.6f})_+^{order}")

    if not basis_list:
        return np.zeros((n, 0), dtype=float), []

    return np.column_stack(basis_list), basis_names


def normalized_hal_density(
    grid: Sequence[float] | np.ndarray,
    theta_hat: np.ndarray,
    basis_grid_points: np.ndarray,
    basis_order: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Evaluate and normalize the HAL density on `grid`."""
    grid_arr = np.asarray(grid, dtype=float).reshape(-1)
    theta_arr = np.asarray(theta_hat, dtype=float).ravel()

    basis_matrix, _ = build_hal_design(
        values=grid_arr,
        knots=basis_grid_points,
        order=basis_order,
        include_intercept=True,
    )
    if basis_matrix.shape[1] != theta_arr.size:
        raise ValueError("Theta length does not match basis width.")

    log_density = basis_matrix @ theta_arr
    max_log = float(np.max(log_density))
    density = np.exp(log_density - max_log)

    delta = _integration_widths(grid_arr)
    normalizer = float(np.sum(density * delta))
    if normalizer <= 0:
        raise ValueError("Normalization constant must be positive.")
    density /= normalizer

    return density, delta, max_log, normalizer


def _integration_widths(grid: np.ndarray) -> np.ndarray:
    """Compute integration widths (Δ_j) for a monotonically increasing grid."""
    delta = np.empty_like(grid)
    if grid.size <= 1:
        delta[:] = 1.0
        return delta

    diffs = np.diff(grid)
    delta[0] = diffs[0]
    delta[1:] = diffs
    return delta

