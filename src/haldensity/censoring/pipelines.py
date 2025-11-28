import pandas as pd
from typing import Dict, Any, Optional
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .em import EMIPCWEstimator


def run_ipcw_hal_mle(
    data: pd.DataFrame,
    norm_constraint: float = 20.0,
    basis_order: int = 0,
    n_grid_points: int = 200,
    return_estimator: bool = False,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any] | tuple[Dict[str, Any], WeightedCVXPYEstimator]:
    """
    Fit the IPCW–HAL–MLE initializer and return standardized HAL results.

    Parameters mirror the research notebooks but allow optional overrides via
    `estimator_kwargs` (e.g., solver="SCS"). Set `return_estimator=True` to
    receive both the results dictionary and the fitted estimator instance for
    downstream targeting or Optuna tuning.
    """
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    w = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)
    mask_unc = (data["Delta"].values == 1)
    df_unc = pd.DataFrame({"W1": data.loc[mask_unc, "T"].values})
    w_unc = w[mask_unc]
    extra = dict(estimator_kwargs or {})
    extra.setdefault("norm_constraint", norm_constraint)
    extra.setdefault("basis_order", basis_order)
    extra.setdefault("n_grid_points", n_grid_points)
    est = WeightedCVXPYEstimator(**extra).fit(df_unc, sample_weights=w_unc)
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
) -> Dict[str, Any] | tuple[Dict[str, Any], EMIPCWEstimator]:
    """
    Run the full EM–IPCW–HAL–MLE workflow with multiple imputation draws.

    The returned dictionary follows `BaseEstimator._get_common_results()` so
    targeting learners and diagnostics can treat censored fits exactly like the
    uncensored estimators. Pass overrides through `estimator_kwargs` (init/m-step
    solvers, RNG seed, EM tolerances, etc.) and set `return_estimator=True` if
    you also need the fitted estimator object for targeting or Optuna tuning.
    """
    extra = dict(estimator_kwargs or {})
    extra.setdefault("norm_constraint", norm_constraint)
    extra.setdefault("basis_order", basis_order)
    extra.setdefault("n_grid_points", n_grid_points)
    extra.setdefault("m_imputations", m_imputations)
    extra.setdefault("max_em_iter", max_em_iter)
    extra.setdefault("em_tol", em_tol)
    extra.setdefault("use_sc_adjustment", use_sc_adjustment)
    est = EMIPCWEstimator(**extra).fit(data)
    results = est.get_results()
    return (results, est) if return_estimator else results


