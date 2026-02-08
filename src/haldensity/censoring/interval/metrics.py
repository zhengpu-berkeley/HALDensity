"""Evaluation metrics for interval-censored density estimation."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _cdf_from_density_on_grid(grid_points: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a right-edge CDF representation from a piecewise-constant density on bins.

    Parameters
    ----------
    grid_points:
        1D array of bin edges of length J+1 in [0, 1].
    density:
        1D array of density values on bin midpoints of length J.

    Returns
    -------
    tuple
        (right_edges, cdf_at_right_edges) each of length J.
    """
    edges = np.asarray(grid_points, dtype=float).ravel()
    dens = np.asarray(density, dtype=float).ravel()
    if edges.size < 2:
        raise ValueError("grid_points must have length >= 2")
    if dens.size != edges.size - 1:
        raise ValueError("density must have length len(grid_points) - 1")

    delta = np.diff(edges)
    mass = np.maximum(dens, 0.0) * np.maximum(delta, 0.0)
    total = float(np.sum(mass))
    if total <= 0:
        # Degenerate; return a near-step at 1.
        right_edges = edges[1:].copy()
        cdf_right = np.linspace(0.0, 1.0, right_edges.size)
        cdf_right[-1] = 1.0
        return right_edges, cdf_right

    mass = mass / total
    cdf_right = np.cumsum(mass)
    cdf_right[-1] = 1.0
    return edges[1:].copy(), cdf_right


def incomplete_loglik_interval(
    estimator: Any,
    val_df: pd.DataFrame,
    L_col: str = "L",
    R_col: str = "R",
    min_mass: float = 1e-12,
) -> float:
    """Compute interval-censored incomplete-data log-likelihood.

    For interval-censored observations, the likelihood contribution is:
        log P(L < T* <= R) = log (F(R) - F(L))

    We approximate F via the estimator's discretized density grid (piecewise-constant
    on bins).
    """
    if L_col not in val_df.columns or R_col not in val_df.columns:
        raise ValueError(f"val_df must contain columns {L_col!r} and {R_col!r}")

    L = np.asarray(val_df[L_col].values, dtype=float).ravel()
    R = np.asarray(val_df[R_col].values, dtype=float).ravel()
    if L.shape != R.shape:
        raise ValueError("L and R must have the same shape")

    grid_mid, dens = estimator.get_density()

    # Prefer using estimator.grid_points if available (bin edges); otherwise assume
    # grid_mid are "points" and fabricate edges.
    edges = getattr(estimator, "grid_points", None)
    if edges is None:
        # Build edges from midpoints (approx).
        gm = np.asarray(grid_mid, dtype=float).ravel()
        if gm.size < 2:
            raise ValueError("Estimator grid too small to build a CDF")
        step = float(np.median(np.diff(gm)))
        edges = np.concatenate(([gm[0] - step / 2], gm + step / 2))
    else:
        edges = np.asarray(edges, dtype=float).ravel()

    right_edges, cdf_right = _cdf_from_density_on_grid(edges, np.asarray(dens, dtype=float).ravel())

    # CDF(t) by linear interpolation on right edges, with F(0)=0 and F(1)=1.
    cdf_L = np.interp(L, right_edges, cdf_right, left=0.0, right=1.0)
    cdf_R = np.interp(R, right_edges, cdf_right, left=0.0, right=1.0)
    mass = np.maximum(cdf_R - cdf_L, min_mass)
    return float(np.sum(np.log(mass)))


