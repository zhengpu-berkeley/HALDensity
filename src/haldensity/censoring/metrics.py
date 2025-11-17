import numpy as np
import pandas as pd
from typing import Any, Callable, Optional
from haldensity.utils.density_computations import (
    generic_compute_survival_from_density,
)


def _interp_density(estimator, points: np.ndarray) -> np.ndarray:
    grid_points, density = estimator.get_density()
    # Linear interpolation with boundary fill
    left = float(density[0])
    right = float(density[-1])
    return np.interp(points, grid_points, density, left=left, right=right)


def _interp_survival(estimator, points: np.ndarray) -> np.ndarray:
    grid_points, density = estimator.get_density()
    survival, surv_grid = generic_compute_survival_from_density(grid_points, density)
    left = float(survival[0])
    right = float(survival[-1])
    return np.interp(points, surv_grid, survival, left=left, right=right)


def incomplete_loglik(estimator, val_df: pd.DataFrame, time_col: str = "T", delta_col: str = "Delta") -> float:
    """
    Incomplete-data log-likelihood:
        sum Delta * log f(T) + (1 - Delta) * log S(T)
    """
    t = np.asarray(val_df[time_col].values, dtype=float)
    d = np.asarray(val_df[delta_col].values, dtype=int)
    f = _interp_density(estimator, t)
    S = _interp_survival(estimator, t)
    f = np.maximum(f, 1e-12)
    S = np.maximum(S, 1e-12)
    ll = d * np.log(f) + (1 - d) * np.log(S)
    return float(np.sum(ll))


def mi_complete_loglik(
    estimator: Any,
    augmented_df: Optional[pd.DataFrame],
    value_col: str = "W1",
    weight_col: str = "weight",
    min_density: float = 1e-12,
) -> float:
    """
    MI-pooled complete-data log-likelihood proxy:
        Σ_j w_j log f_{θ}(x_j)

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator that exposes `get_density_at_points`.
    augmented_df : pd.DataFrame | None
        Pooled pseudo-complete data (e.g., output from `e_step_multiple_imputation`)
        containing at least `value_col` and, optionally, `weight_col`.
    value_col : str
        Column with observation locations in [0, 1].
    weight_col : str
        Column of replicate weights (defaults to 1 if missing).
    min_density : float
        Lower bound used to stabilize log evaluations.
    """
    if augmented_df is None or augmented_df.empty:
        return float("-inf")

    values = np.asarray(augmented_df[value_col].values, dtype=float).ravel()
    if weight_col in augmented_df.columns:
        weights = np.asarray(augmented_df[weight_col].values, dtype=float).ravel()
    else:
        weights = np.ones_like(values)

    densities = estimator.get_density_at_points(values)
    densities = np.maximum(densities, min_density)
    log_terms = np.log(densities)

    return float(np.sum(weights * log_terms))


def kl_divergence(true_pdf_fn: Callable[[np.ndarray], np.ndarray],
                  grid: np.ndarray,
                  est_density: np.ndarray,
                  eps: float = 1e-12) -> float:
    """
    Compute D_KL(true || est) ≈ Σ p_true(x) log(p_true(x) / p_est(x)) Δx
    """
    grid = np.asarray(grid, dtype=float)
    p_true = np.asarray(true_pdf_fn(grid), dtype=float)
    p_est = np.asarray(est_density, dtype=float)
    # Normalize to integrate to 1 numerically on the grid (safety)
    delta = np.diff(grid, prepend=grid[0])
    p_true = np.maximum(p_true, eps)
    p_est = np.maximum(p_est, eps)
    # Renormalize densities on this grid
    p_true = p_true / np.sum(p_true * delta)
    p_est = p_est / np.sum(p_est * delta)
    ratio = p_true / p_est
    return float(np.sum(p_true * np.log(ratio) * delta))


