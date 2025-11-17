import numpy as np
import pandas as pd
from typing import Callable
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


def mi_complete_loglik(augmented_df: pd.DataFrame, time_col: str = "W1", replicate_col: str = "mi_rep") -> float:
    """
    MI-pooled complete-data log-likelihood proxy:
      Average over replicates of the complete-data sum log f(x).
    Expect augmented_df to include per-replicate densities or be accompanied by an estimator per replicate.
    Here we assume uniform weights across pooled replicates and simply return 0 as placeholder unless
    the caller computes per-replicate log-likelihoods. This function is included for API completeness.
    """
    # Placeholder simple proxy: not used in core tests; can be extended as needed by caller.
    # Return 0 to avoid influencing selection if not explicitly used.
    return 0.0


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


