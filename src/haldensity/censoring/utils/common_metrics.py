"""Censoring-agnostic evaluation metrics.

Contains metrics that work across different censoring types.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def kl_divergence(
    true_pdf_fn: Callable[[np.ndarray], np.ndarray],
    grid: np.ndarray,
    est_density: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute KL divergence D_KL(true || estimated).

    Approximates the Kullback-Leibler divergence:
        D_KL(p || q) = integral p(x) * log(p(x) / q(x)) dx

    using a Riemann sum over the provided grid.

    Parameters
    ----------
    true_pdf_fn : Callable
        Function that takes array of points and returns true density values.
    grid : np.ndarray
        Grid points for integration.
    est_density : np.ndarray
        Estimated density values at grid points.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        KL divergence value. Lower is better.

    Examples
    --------
    >>> from scipy.stats import norm
    >>> grid = np.linspace(-3, 3, 100)
    >>> true_pdf = lambda x: norm.pdf(x)
    >>> est_density = norm.pdf(grid, loc=0.1)  # Slightly shifted
    >>> kl = kl_divergence(true_pdf, grid, est_density)
    """
    grid = np.asarray(grid, dtype=float)
    p_true = np.asarray(true_pdf_fn(grid), dtype=float)
    p_est = np.asarray(est_density, dtype=float)

    # Compute integration widths
    delta = np.diff(grid, prepend=grid[0])

    # Clip to avoid numerical issues
    p_true = np.maximum(p_true, eps)
    p_est = np.maximum(p_est, eps)

    # Renormalize densities on this grid
    p_true = p_true / np.sum(p_true * delta)
    p_est = p_est / np.sum(p_est * delta)

    ratio = p_true / p_est
    return float(np.sum(p_true * np.log(ratio) * delta))

