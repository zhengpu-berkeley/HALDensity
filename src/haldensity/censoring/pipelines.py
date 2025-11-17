import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .em_estimator import EMIPCWEstimator


def run_ipcw_hal_mle(
    data: pd.DataFrame,
    norm_constraint: float = 20.0,
    basis_order: int = 0,
    n_grid_points: int = 200,
) -> Dict[str, Any]:
    """
    IPCW–HAL–MLE initialization pipeline:
      - Fit KM for censoring S_c
      - Compute w = Δ / S_c(T)
      - Fit WeightedCVXPYEstimator on uncensored data with weights
    """
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    w = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)
    mask_unc = (data["Delta"].values == 1)
    df_unc = pd.DataFrame({"W1": data.loc[mask_unc, "T"].values})
    w_unc = w[mask_unc]
    est = WeightedCVXPYEstimator(
        norm_constraint=norm_constraint,
        basis_order=basis_order,
        n_grid_points=n_grid_points,
    ).fit(df_unc, sample_weights=w_unc)
    return est.get_results()


def run_em_ipcw_hal_mle(
    data: pd.DataFrame,
    norm_constraint: float = 20.0,
    basis_order: int = 0,
    n_grid_points: int = 200,
    m_imputations: int = 20,
    max_em_iter: int = 50,
    em_tol: float = 1e-3,
    use_sc_adjustment: bool = False,
) -> Dict[str, Any]:
    """
    EM–IPCW–HAL–MLE pipeline with multiple imputation.
    """
    est = EMIPCWEstimator(
        norm_constraint=norm_constraint,
        basis_order=basis_order,
        n_grid_points=n_grid_points,
        m_imputations=m_imputations,
        max_em_iter=max_em_iter,
        em_tol=em_tol,
        use_sc_adjustment=use_sc_adjustment,
    ).fit(data)
    return est.get_results()


