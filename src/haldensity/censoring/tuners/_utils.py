"""Utility functions for censored density tuners.

Provides helper functions for IPCW weight computation and estimator fitting.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from haldensity.censoring._defaults import EM_DEFAULTS, TUNER_DEFAULTS


def create_s_c_predict(km: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Create S_C prediction wrapper that returns numpy array.
    
    Parameters
    ----------
    km : KaplanMeier
        Fitted Kaplan-Meier estimator.
        
    Returns
    -------
    Callable
        Function that predicts S_C(t) for array of times.
    """
    def s_c_predict(t: np.ndarray) -> np.ndarray:
        return np.atleast_1d(km.predict(t))
    return s_c_predict


def prepare_ipcw_data(
    data: pd.DataFrame,
    time_col: str = "T",
    delta_col: str = "Delta",
) -> Tuple[pd.DataFrame, np.ndarray, Any, Callable]:
    """Prepare data for IPCW estimation.
    
    Fits Kaplan-Meier, computes IPCW weights, and extracts uncensored observations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with time and event columns.
    time_col : str
        Name of time column.
    delta_col : str
        Name of event indicator column.
        
    Returns
    -------
    tuple
        (ipcw_df, ipcw_weights, km, s_c_predict)
        - ipcw_df: DataFrame with 'W1' column for uncensored observations
        - ipcw_weights: IPCW weights for uncensored observations
        - km: Fitted KaplanMeier estimator
        - s_c_predict: Function to predict S_C(t)
    """
    # Import here to avoid circular imports
    from haldensity.censoring.right.km import KaplanMeier
    from haldensity.censoring.right.weights import compute_ipcw_weights
    
    km = KaplanMeier().fit(data, time_col=time_col, delta_col=delta_col)
    s_c_predict = create_s_c_predict(km)
    
    T_vals = np.asarray(data[time_col].values, dtype=float)
    Delta_vals = np.asarray(data[delta_col].values, dtype=int)
    
    weights = compute_ipcw_weights(T_vals, Delta_vals, s_c_predict)
    uncensored_mask = Delta_vals == 1
    
    ipcw_df = pd.DataFrame({"W1": T_vals[uncensored_mask]})
    ipcw_weights = weights[uncensored_mask]
    
    return ipcw_df, ipcw_weights, km, s_c_predict


def fit_rc_init_estimator(
    data: pd.DataFrame,
    norm_constraint: float,
    basis_order: int,
    n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
    solver: str = TUNER_DEFAULTS.solver,
    use_secondary_solver: bool = TUNER_DEFAULTS.use_secondary_solver,
    tol: float = EM_DEFAULTS.tol,
    time_col: str = "T",
    delta_col: str = "Delta",
) -> Tuple[Any, Any, Callable]:
    """Fit a right-censored init estimator on data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with time and event columns.
    norm_constraint : float
        L1 norm constraint.
    basis_order : int
        Basis order.
    n_grid_points : int
        Number of grid points.
    solver : str
        CVXPY solver.
    use_secondary_solver : bool
        Whether to use fallback solvers.
    tol : float
        Tolerance for coefficient pruning.
    time_col : str
        Name of time column.
    delta_col : str
        Name of event indicator column.
        
    Returns
    -------
    tuple
        (estimator, km, s_c_predict)
    """
    from haldensity.censoring.right.estimators import RightCensoredInitEstimator
    
    ipcw_df, ipcw_weights, km, s_c_predict = prepare_ipcw_data(
        data, time_col=time_col, delta_col=delta_col
    )
    
    estimator = RightCensoredInitEstimator(
        tol=tol,
        norm_constraint=norm_constraint,
        n_grid_points=n_grid_points,
        basis_order=basis_order,
        solver=solver,
        use_secondary_solver=use_secondary_solver,
    )
    estimator.fit(ipcw_df, sample_weights=ipcw_weights)
    
    return estimator, km, s_c_predict


def fit_ic_init_estimator(
    data: pd.DataFrame,
    norm_constraint: float,
    basis_order: int,
    n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
    solver: str = TUNER_DEFAULTS.solver,
    use_secondary_solver: bool = TUNER_DEFAULTS.use_secondary_solver,
    tol: float = EM_DEFAULTS.tol,
    L_col: str = "L",
    R_col: str = "R",
) -> Any:
    """Fit an interval-censored init estimator on data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with interval columns.
    norm_constraint : float
        L1 norm constraint.
    basis_order : int
        Basis order.
    n_grid_points : int
        Number of grid points.
    solver : str
        CVXPY solver.
    use_secondary_solver : bool
        Whether to use fallback solvers.
    tol : float
        Tolerance for coefficient pruning.
    L_col : str
        Name of left interval column.
    R_col : str
        Name of right interval column.
        
    Returns
    -------
    Any
        Fitted estimator.
    """
    from haldensity.censoring.interval.estimators import IntervalCensoredInitEstimator
    
    estimator = IntervalCensoredInitEstimator(
        tol=tol,
        norm_constraint=norm_constraint,
        n_grid_points=n_grid_points,
        basis_order=basis_order,
        solver=solver,
        use_secondary_solver=use_secondary_solver,
        include_intercept_in_constraint=False,
    )
    estimator.fit(data, L_col=L_col, R_col=R_col)
    
    return estimator
