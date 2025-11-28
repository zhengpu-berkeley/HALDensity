"""Kaplan-Meier estimator for right-censoring survival.

Estimates the censoring survival function S_C(t) = P(C > t) for right-censored data.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd


class KaplanMeier:
    """Kaplan-Meier estimator for censoring survival S_C(t) on [0, 1].

    For right-censored data where we observe (T, Delta):
    - T = min(T*, C) is the observed time
    - Delta = I(T* <= C) is the event indicator (1 = uncensored)

    We treat censoring events as (1 - Delta) to estimate S_C(t).

    Examples
    --------
    >>> km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    >>> survival_probs = km.predict(np.array([0.2, 0.5, 0.8]))
    """

    def __init__(self) -> None:
        self._times: Optional[np.ndarray] = None
        self._surv: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        time_col: str = "T",
        delta_col: str = "Delta",
    ) -> "KaplanMeier":
        """Fit Kaplan-Meier estimator for censoring survival S_C(t).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns for time and event indicator.
        time_col : str
            Column name for observed time Y = min(T, C) in [0, 1].
        delta_col : str
            Column name for event indicator Delta in {0, 1} (1 = uncensored).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If data is empty or has invalid shapes.
        """
        y = np.asarray(data[time_col].values, dtype=float).ravel()
        delta = np.asarray(data[delta_col].values, dtype=int).ravel()

        if y.ndim != 1 or delta.ndim != 1 or y.shape[0] != delta.shape[0]:
            raise ValueError("Invalid shapes for time and delta")
        if y.size == 0:
            raise ValueError("Empty data")

        # Censoring events: 1 - Delta (we're estimating survival of C, not T)
        cens_event = 1 - delta

        # Sort by time
        order = np.argsort(y)
        y_sorted = y[order]
        cens_sorted = cens_event[order]

        # Aggregate by unique times
        uniq_times, idx_start = np.unique(y_sorted, return_index=True)
        counts_at_time = np.diff(np.append(idx_start, len(y_sorted)))
        cens_at_time = np.add.reduceat(cens_sorted, idx_start)

        # Risk set sizes at the start of each unique time
        n = len(y_sorted)
        at_risk = n - np.cumsum(np.append(0, counts_at_time[:-1]))

        # KM for censoring survival: product over (1 - d_j / r_j)
        frac = np.clip(1.0 - (cens_at_time / np.maximum(at_risk, 1)), 0.0, 1.0)
        surv = np.cumprod(frac, dtype=float)

        self._times = uniq_times
        self._surv = surv
        return self

    def stepwise_survival_(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times, survival) step function arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Copies of the unique times and corresponding survival values.

        Raises
        ------
        ValueError
            If fit() has not been called.
        """
        if self._times is None or self._surv is None:
            raise ValueError("Call fit() first")
        return self._times.copy(), self._surv.copy()

    def predict(self, t: Union[float, np.ndarray, list]) -> Union[float, np.ndarray]:
        """Evaluate right-continuous step function S_C(t).

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to evaluate survival probability.

        Returns
        -------
        float or np.ndarray
            S_C(t) = P(C > t). Returns scalar if input is scalar.

        Raises
        ------
        ValueError
            If fit() has not been called.
        """
        if self._times is None or self._surv is None:
            raise ValueError("Call fit() first")

        ts = np.asarray(t, dtype=float)
        times, surv = self._times, self._surv

        # For values below min time => S=1; above last time => last survival
        idx = np.searchsorted(times, ts, side="right") - 1
        idx = np.clip(idx, -1, len(times) - 1)
        out = np.where(idx >= 0, surv[idx], 1.0)

        return out if isinstance(t, (list, np.ndarray)) else float(out)

