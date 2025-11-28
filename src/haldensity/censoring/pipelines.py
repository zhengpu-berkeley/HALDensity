"""Convenience workflow functions for censored data density estimation.

Provides simple interfaces for common estimation pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import pandas as pd

from .right.km import KaplanMeier
from .right.weights import compute_ipcw_weights
from .right.ipcw_estimator import RightCensoredIPCWEstimator
from .right.em_estimator import RightCensoredEMEstimator


def run_ipcw_hal_mle(
    data: pd.DataFrame,
    norm_constraint: float = 20.0,
    basis_order: int = 0,
    n_grid_points: int = 200,
    return_estimator: bool = False,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], tuple[Dict[str, Any], RightCensoredIPCWEstimator]]:
    """Fit the IPCW-HAL-MLE initializer and return standardized HAL results.

    This is a convenience function that handles the full IPCW workflow:
    1. Fit Kaplan-Meier for censoring survival
    2. Compute IPCW weights
    3. Fit weighted HAL-MLE on uncensored observations

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
    norm_constraint : float
        L1 norm constraint for theta.
    basis_order : int
        Order of the truncated power basis.
    n_grid_points : int
        Number of grid points for density evaluation.
    return_estimator : bool
        If True, return both results dict and fitted estimator.
    estimator_kwargs : dict | None
        Additional keyword arguments for the estimator.

    Returns
    -------
    dict or tuple
        If return_estimator=False: results dictionary from get_results().
        If return_estimator=True: (results_dict, fitted_estimator).

    Examples
    --------
    >>> results = run_ipcw_hal_mle(data, norm_constraint=50.0)
    >>> density = results["estimated_density"]
    >>> 
    >>> # Or get the estimator too
    >>> results, est = run_ipcw_hal_mle(data, return_estimator=True)
    >>> grid, density = est.get_density()
    """
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    w = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)

    mask_unc = data["Delta"].values == 1
    df_unc = pd.DataFrame({"W1": data.loc[mask_unc, "T"].values})
    w_unc = w[mask_unc]

    extra = dict(estimator_kwargs or {})
    extra.setdefault("norm_constraint", norm_constraint)
    extra.setdefault("basis_order", basis_order)
    extra.setdefault("n_grid_points", n_grid_points)

    est = RightCensoredIPCWEstimator(**extra).fit(df_unc, sample_weights=w_unc)
    results = est.get_results()

    return (results, est) if return_estimator else results


def run_em_ipcw_hal_mle(
    data: pd.DataFrame,
    norm_constraint: float = 20.0,
    basis_order: int = 0,
    n_grid_points: int = 200,
    m_imputations: int = 20,
    max_em_iter: int = 50,
    em_tol: float = 1e-3,
    use_sc_adjustment: bool = False,
    return_estimator: bool = False,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], tuple[Dict[str, Any], RightCensoredEMEstimator]]:
    """Run the full EM-IPCW-HAL-MLE workflow with multiple imputation.

    This is a convenience function that handles the full EM workflow:
    1. Fit IPCW-HAL-MLE initialization
    2. Run EM iterations with multiple imputation for censored observations

    The returned dictionary follows `BaseEstimator._get_common_results()` so
    targeting learners and diagnostics can treat censored fits exactly like
    uncensored estimators.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
    norm_constraint : float
        L1 norm constraint for theta.
    basis_order : int
        Order of the truncated power basis.
    n_grid_points : int
        Number of grid points for density evaluation.
    m_imputations : int
        Number of imputations per censored observation.
    max_em_iter : int
        Maximum EM iterations.
    em_tol : float
        Convergence tolerance for log-likelihood.
    use_sc_adjustment : bool
        Whether to adjust for censoring in E-step sampling.
    return_estimator : bool
        If True, return both results dict and fitted estimator.
    estimator_kwargs : dict | None
        Additional keyword arguments for the estimator.

    Returns
    -------
    dict or tuple
        If return_estimator=False: results dictionary from get_results().
        If return_estimator=True: (results_dict, fitted_estimator).

    Examples
    --------
    >>> results = run_em_ipcw_hal_mle(
    ...     data,
    ...     norm_constraint=50.0,
    ...     m_imputations=20,
    ...     max_em_iter=10,
    ... )
    >>> print(f"EM converged: {results['em_converged']}")
    >>> print(f"Iterations: {results['em_iterations']}")
    """
    extra = dict(estimator_kwargs or {})
    extra.setdefault("norm_constraint", norm_constraint)
    extra.setdefault("basis_order", basis_order)
    extra.setdefault("n_grid_points", n_grid_points)
    extra.setdefault("m_imputations", m_imputations)
    extra.setdefault("max_em_iter", max_em_iter)
    extra.setdefault("em_tol", em_tol)
    extra.setdefault("use_sc_adjustment", use_sc_adjustment)

    est = RightCensoredEMEstimator(**extra).fit(data)
    results = est.get_results()

    return (results, est) if return_estimator else results
