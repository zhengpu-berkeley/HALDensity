import numpy as np
import pandas as pd
from typing import Tuple

def create_basis_functions(
    data_long_train: pd.DataFrame, 
    grid_points: np.ndarray, 
    order: int = 1,
    include_intercept: bool = True
) -> Tuple[np.ndarray, list[str]]:
    """
    Create truncated power basis functions for the data.
    
    For order=0 (Schumaker convention):
        {1, I(x >= ξ₁), I(x >= ξ₂), ..., I(x >= ξₘ)}
    
    For order≥1:
        {1, x, x², ..., x^k, (x-ξ₁)₊^k, (x-ξ₂)₊^k, ..., (x-ξₘ)₊^k}
        where (x-ξⱼ)₊^k = max(x-ξⱼ, 0)^k
    
    Args:
        data_long_train: DataFrame containing the data (uses 'W1' column)
        grid_points: Knot points ξ₁, ξ₂, ..., ξₘ for truncated power functions
        order: Polynomial and spline order (k)
        include_intercept: Whether to include intercept term
        
    Returns:
        Tuple of (basis_array, basis_names) where basis_array has shape (n, p)
        p = (k+1) + m for order≥1, or 1+m for order=0
    """
    x = data_long_train['W1'].values
    n, m = len(x), len(grid_points)
    basis_list, basis_names = [], []
    
    if order == 0:
        # Basis: {1, I(x >= ξ₁), I(x >= ξ₂), ..., I(x >= ξₘ)}
        if include_intercept:
            basis_list.append(np.ones(n, dtype=np.float32))
            basis_names.append("Intercept")
        
        # Vectorized indicator functions
        indicators = (x[:, np.newaxis] >= grid_points[np.newaxis, :]).astype(np.float32)
        for j in range(m):
            basis_list.append(indicators[:, j])
            basis_names.append(f"I(x >= {grid_points[j]:.6f})")
    else:
        # Basis: {1, x, x², ..., x^k, (x-ξ₁)₊^k, (x-ξ₂)₊^k, ..., (x-ξₘ)₊^k}
        for power in range(order + 1):
            if power == 0 and include_intercept:
                basis_list.append(np.ones(n, dtype=np.float32))
                basis_names.append("Intercept")
            elif power > 0:
                basis_list.append((x ** power).astype(np.float32))
                basis_names.append(f"x^{power}")
        
        # Truncated power terms
        for knot in grid_points:
            basis_list.append((np.maximum(x - knot, 0) ** order).astype(np.float32))
            basis_names.append(f"(x - {knot:.6f})_+^{order}")
    
    basis_array = np.column_stack(basis_list)
    return basis_array, basis_names

