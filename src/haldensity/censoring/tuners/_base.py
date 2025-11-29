"""Shared utilities and mixins for hyperparameter tuners.

Contains IPCW fitting helper, default constants, and common evaluation logic.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np
import pandas as pd

from haldensity.censoring.core.models import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.ipcw_estimator import RightCensoredIPCWEstimator


# Mapping of estimator names to classes
ESTIMATORS = {
    "RightCensoredEMEstimator": "RightCensoredEMEstimator",
    "RightCensoredIPCWEstimator": "RightCensoredIPCWEstimator",
}


class IPCWFittingMixin:
    """Mixin providing IPCW estimator fitting functionality.

    Provides common methods for fitting IPCW estimators that are used
    across multiple tuner classes.
    """

    n_grid_points: int

    @staticmethod
    def create_s_c_predict(km: KaplanMeier) -> Callable[[np.ndarray], np.ndarray]:
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

    def fit_ipcw_estimator(
        self,
        train_df: pd.DataFrame,
        norm_constraint: float,
        basis_order: int,
        n_grid_points: Optional[int] = None,
        solver: str = "ECOS",
        use_secondary_solver: bool = True,
    ) -> tuple[RightCensoredIPCWEstimator, KaplanMeier]:
        """Fit IPCW estimator on training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with 'T' and 'Delta' columns.
        norm_constraint : float
            L1 norm constraint.
        basis_order : int
            Basis order.
        n_grid_points : int | None
            Number of grid points (uses self.n_grid_points if None).
        solver : str
            CVXPY solver.
        use_secondary_solver : bool
            Whether to use fallback solvers.

        Returns
        -------
        tuple
            (fitted_estimator, fitted_km)
        """
        if n_grid_points is None:
            n_grid_points = getattr(self, "n_grid_points", TUNER_DEFAULTS.n_grid_points)

        km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
        s_c_predict = self.create_s_c_predict(km)

        T_vals = np.asarray(train_df["T"].values, dtype=float)
        Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)

        weights = compute_ipcw_weights(T_vals, Delta_vals, s_c_predict)
        uncensored_mask = Delta_vals == 1
        ipcw_data = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        ipcw_weights = weights[uncensored_mask]

        estimator = RightCensoredIPCWEstimator(
            tol=EM_DEFAULTS.tol,
            norm_constraint=norm_constraint,
            n_grid_points=n_grid_points,
            basis_order=basis_order,
            solver=solver,
            use_secondary_solver=use_secondary_solver,
        )
        estimator.fit(ipcw_data, sample_weights=ipcw_weights)

        return estimator, km


def get_estimator_class(estimator_name: str) -> type:
    """Get estimator class by name.

    Parameters
    ----------
    estimator_name : str
        Name of the estimator class.

    Returns
    -------
    type
        The estimator class.

    Raises
    ------
    ValueError
        If estimator name is not recognized.
    """
    # Lazy import to avoid circular imports
    from haldensity.censoring.right.em_estimator import RightCensoredEMEstimator
    from haldensity.censoring.right.ipcw_estimator import RightCensoredIPCWEstimator

    class_map = {
        "EMIPCWEstimator": RightCensoredEMEstimator,
        "RightCensoredEMEstimator": RightCensoredEMEstimator,
        "RightCensoredIPCWEstimator": RightCensoredIPCWEstimator,
    }

    if estimator_name not in class_map:
        raise ValueError(
            f"Unsupported estimator '{estimator_name}'. "
            f"Available: {list(class_map.keys())}"
        )

    return class_map[estimator_name]

