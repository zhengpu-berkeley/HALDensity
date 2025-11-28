"""Protocols for censored data density estimation.

Defines interfaces for censoring survival estimators and density estimators
to enable future extensibility to left-censoring and interval-censoring.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class CensoringSurvivalEstimator(Protocol):
    """Protocol for estimating censoring survival function S_C(t) = P(C > t).

    Implementations should estimate the survival function of the censoring
    distribution from observed data (T, Delta) where:
    - T = min(T*, C) is the observed time
    - Delta = I(T* <= C) is the event indicator (1 = uncensored, 0 = censored)

    For right-censoring: KaplanMeier
    For left-censoring: Reverse Kaplan-Meier (future)
    For interval-censoring: Turnbull estimator (future)
    """

    def fit(
        self,
        data: pd.DataFrame,
        time_col: str = "T",
        delta_col: str = "Delta",
    ) -> "CensoringSurvivalEstimator":
        """Fit the censoring survival estimator.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with observed times and event indicators.
        time_col : str
            Column name for observed time.
        delta_col : str
            Column name for event indicator (1=uncensored, 0=censored).

        Returns
        -------
        self
        """
        ...

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict censoring survival probability S_C(t).

        Parameters
        ----------
        t : np.ndarray
            Times at which to evaluate survival probability.

        Returns
        -------
        np.ndarray
            Survival probabilities S_C(t) = P(C > t).
        """
        ...


@runtime_checkable
class DensityEstimatorProtocol(Protocol):
    """Protocol defining the interface required for density estimators.

    Any estimator that can be used as an initial estimator for EM refinement
    must implement this protocol.
    """

    theta_hat: np.ndarray
    _grid_points_hal: np.ndarray
    basis_order: int

    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the estimated density on the evaluation grid.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (grid_points, density_values)
        """
        ...

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate density at specific points.

        Parameters
        ----------
        points : np.ndarray
            Points at which to evaluate the density.

        Returns
        -------
        np.ndarray
            Density values at the given points.
        """
        ...


# Future: CensoringMechanism protocol for strategy pattern
# This would allow pluggable censoring types without changing estimator code
#
# class CensoringMechanism(Protocol):
#     """Strategy pattern for different censoring types."""
#     censoring_type: str  # "right", "left", "interval"
#
#     def fit_censoring_survival(self, data: pd.DataFrame) -> CensoringSurvivalEstimator:
#         """Fit appropriate censoring survival estimator."""
#         ...
#
#     def compute_weights(self, data: pd.DataFrame, S_c: CensoringSurvivalEstimator) -> np.ndarray:
#         """Compute IPCW weights for this censoring type."""
#         ...
#
#     def sample_imputations(
#         self,
#         censored_obs: pd.DataFrame,
#         density_fn: Callable,
#         rng: np.random.Generator,
#     ) -> np.ndarray:
#         """Sample imputed values for censored observations."""
#         ...
#
#     def incomplete_loglik(self, estimator, data: pd.DataFrame) -> float:
#         """Compute incomplete-data log-likelihood."""
#         ...

