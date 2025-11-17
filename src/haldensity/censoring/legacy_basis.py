import numpy as np
import pandas as pd
from typing import Tuple


def create_legacy_basis(
    data: pd.DataFrame,
    grid_points: np.ndarray,
) -> Tuple[np.ndarray, list[str]]:
    """
    Legacy basis used by original IPCW/EM script (legacy_em_ipcw_hal.py).
    NO INTERCEPT COLUMN - intercept is handled separately as theta[0].
    
    Returns basis matrix of shape (n_samples, len(grid_points)) where:
        φ_j(x) = max(x - ξ_j, 0) for each knot ξ_j
    
    The CVXPY formulation uses: theta[0] + basis @ theta[1:]
    """
    x = np.asarray(data["W1"].values, dtype=float)
    n = x.shape[0]
    knots = np.asarray(grid_points, dtype=float)
    m = knots.shape[0]
    basis = np.empty((n, m), dtype=float)
    for j, knot in enumerate(knots):
        basis[:, j] = np.maximum(x - knot, 0.0)
    names = [f"W1_basis_{j}" for j in range(m)]
    return basis, names


