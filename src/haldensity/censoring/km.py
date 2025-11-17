import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple


class KaplanMeier:
    """
    Minimal Kaplan–Meier estimator for censoring survival S_c(t) on [0,1].
    We treat censoring events as (1 - Delta) on observed times Y = T ∧ C.
    """
    def __init__(self) -> None:
        self._times: Optional[np.ndarray] = None
        self._surv: Optional[np.ndarray] = None

    def fit(self, data: pd.DataFrame, time_col: str = "T", delta_col: str = "Delta") -> "KaplanMeier":
        """
        Fit KM for censoring survival S_c(t) using events = 1 - Delta at time T.
        Args:
            data: DataFrame with columns [time_col, delta_col]
            time_col: observed time Y = min(T, C) in [0,1]
            delta_col: indicator Delta∈{0,1} for event (uncensored)
        """
        y = np.asarray(data[time_col].values, dtype=float).ravel()
        delta = np.asarray(data[delta_col].values, dtype=int).ravel()
        if y.ndim != 1 or delta.ndim != 1 or y.shape[0] != delta.shape[0]:
            raise ValueError("Invalid shapes for time and delta")
        if y.size == 0:
            raise ValueError("Empty data")

        # Censoring events: 1 - Delta
        cens_event = 1 - delta

        # Aggregate by unique times
        order = np.argsort(y)
        y_sorted = y[order]
        cens_sorted = cens_event[order]

        uniq_times, idx_start = np.unique(y_sorted, return_index=True)
        # counts at each unique time
        counts_at_time = np.diff(np.append(idx_start, len(y_sorted)))
        cens_at_time = np.add.reduceat(cens_sorted, idx_start)

        # Risk set sizes at the start of each unique time
        n = len(y_sorted)
        at_risk = n - np.cumsum(np.append(0, counts_at_time[:-1]))

        # KM for censoring survival: product over (1 - d_j / r_j)
        # Clip to avoid negative/overflow if d_j > r_j due to degenerate data
        frac = np.clip(1.0 - (cens_at_time / np.maximum(at_risk, 1)), 0.0, 1.0)
        surv = np.cumprod(frac, dtype=float)

        self._times = uniq_times
        self._surv = surv
        return self

    def stepwise_survival_(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times, survival) step function arrays."""
        if self._times is None or self._surv is None:
            raise ValueError("Call fit() first")
        return self._times.copy(), self._surv.copy()

    def predict(self, t: Union[float, np.ndarray, list]) -> Union[float, np.ndarray]:
        """Right-continuous step function S_c(t)."""
        if self._times is None or self._surv is None:
            raise ValueError("Call fit() first")
        ts = np.asarray(t, dtype=float)
        times, surv = self._times, self._surv
        # For values below min time => S=1; above last time => last survival
        idx = np.searchsorted(times, ts, side="right") - 1
        idx = np.clip(idx, -1, len(times) - 1)
        out = np.where(idx >= 0, surv[idx], 1.0)
        return out if isinstance(t, (list, np.ndarray)) else float(out)


