"""Evaluation metrics for right-censored density estimation.

Contains log-likelihood functions specific to right-censored data.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from haldensity.utils.density_computations import generic_compute_survival_from_density


def _interp_density(estimator: Any, points: np.ndarray) -> np.ndarray:
    """Interpolate density at given points using linear interpolation.

    Parameters
    ----------
    estimator : Any
        Fitted estimator with get_density() method.
    points : np.ndarray
        Points at which to evaluate density.

    Returns
    -------
    np.ndarray
        Interpolated density values.
    """
    grid_points, density = estimator.get_density()
    left = float(density[0])
    right = float(density[-1])
    return np.interp(points, grid_points, density, left=left, right=right)


def _interp_survival(estimator: Any, points: np.ndarray) -> np.ndarray:
    """Interpolate survival function at given points.

    Parameters
    ----------
    estimator : Any
        Fitted estimator with get_density() method.
    points : np.ndarray
        Points at which to evaluate survival.

    Returns
    -------
    np.ndarray
        Interpolated survival values S(t) = P(T > t).
    """
    grid_points, density = estimator.get_density()
    survival, surv_grid = generic_compute_survival_from_density(grid_points, density)
    left = float(survival[0])
    right = float(survival[-1])
    return np.interp(points, surv_grid, survival, left=left, right=right)


def incomplete_loglik(
    estimator: Any,
    val_df: pd.DataFrame,
    time_col: str = "T",
    delta_col: str = "Delta",
) -> float:
    """Compute incomplete-data log-likelihood for right-censored data.

    The incomplete-data log-likelihood is:
        L(theta) = sum_i [Delta_i * log f(T_i) + (1 - Delta_i) * log S(T_i)]

    where:
    - f(t) is the density
    - S(t) = P(T > t) = 1 - F(t) is the survival function
    - Delta_i = 1 for uncensored, 0 for censored

    Parameters
    ----------
    estimator : Any
        Fitted density estimator with get_density() method.
    val_df : pd.DataFrame
        Validation data with time and event columns.
    time_col : str
        Column name for observed time.
    delta_col : str
        Column name for event indicator.

    Returns
    -------
    float
        Sum of log-likelihood contributions.

    Notes
    -----
    This is the proper likelihood for right-censored data. For uncensored
    observations, we observe f(T). For censored observations, we only know
    T* > C, so we observe S(C) = P(T* > C).
    """
    t = np.asarray(val_df[time_col].values, dtype=float)
    d = np.asarray(val_df[delta_col].values, dtype=int)

    f = _interp_density(estimator, t)
    S = _interp_survival(estimator, t)

    # Clip to avoid log(0)
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
    """Compute MI-pooled complete-data log-likelihood proxy.

    This is the weighted sum of log-densities over the augmented
    (imputed) dataset from multiple imputation:
        L_complete(theta) = sum_j w_j * log f(x_j)

    Parameters
    ----------
    estimator : Any
        Fitted estimator with get_density_at_points() method.
    augmented_df : pd.DataFrame | None
        Pooled pseudo-complete data from e_step_multiple_imputation.
        Contains imputed values for censored observations.
    value_col : str
        Column with observation locations in [0, 1].
    weight_col : str
        Column of replicate weights (defaults to 1 if missing).
    min_density : float
        Lower bound for density to stabilize log evaluations.

    Returns
    -------
    float
        Weighted complete-data log-likelihood.
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


def ipcw_loglik(
    estimator: Any,
    val_df: pd.DataFrame,
    *,
    time_col: str = "T",
    delta_col: str = "Delta",
    S_c_predict: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    km: Optional[Any] = None,
    clip: float = 1e-6,
) -> float:
    """Compute the IPCW validation criterion for right-censored data.

    This evaluates

        sum_i Delta_i / Gbar_hat(T_i) * log f(T_i)

    using a censoring-survival estimate fit on the training fold. The fitting
    criterion itself remains unchanged; this function is intended only as an
    alternate cross-validation score.
    """
    if S_c_predict is None:
        if km is None or not hasattr(km, "predict"):
            raise ValueError("Provide either S_c_predict or a fitted km object.")
        S_c_predict = lambda t: np.atleast_1d(km.predict(t))

    t = np.asarray(val_df[time_col].values, dtype=float)
    d = np.asarray(val_df[delta_col].values, dtype=int)
    f = _interp_density(estimator, t)
    f = np.maximum(f, 1e-12)

    gbar = np.asarray(S_c_predict(t), dtype=float).ravel()
    gbar = np.maximum(gbar, float(clip))
    weights = d / gbar
    return float(np.sum(weights * np.log(f)))

