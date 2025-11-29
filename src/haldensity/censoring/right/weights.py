"""IPCW weight computation for right-censored data.

Computes inverse probability of censoring weights: w_i = Delta_i / S_C(T_i).
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def compute_ipcw_weights(
    T: np.ndarray,
    Delta: np.ndarray,
    S_c_predict: Callable[[np.ndarray], np.ndarray],
    clip: float = 1e-6,
) -> np.ndarray:
    """Compute inverse probability of censoring weights.

    For right-censored data, the IPCW weight is:
        w_i = Delta_i / max(S_C(T_i), clip)

    where:
    - Delta_i = 1 for uncensored observations (T* <= C)
    - Delta_i = 0 for censored observations (T* > C)
    - S_C(t) = P(C > t) is the censoring survival function

    Parameters
    ----------
    T : np.ndarray
        Observed times Y = min(T*, C) in [0, 1].
    Delta : np.ndarray
        Event indicators (1 = uncensored, 0 = censored).
    S_c_predict : Callable
        Function that takes array of times and returns S_C(t).
    clip : float
        Lower bound for S_C(t) to prevent division by zero.

    Returns
    -------
    np.ndarray
        IPCW weights. Zero for censored observations.

    Examples
    --------
    >>> km = KaplanMeier().fit(data)
    >>> weights = compute_ipcw_weights(
    ...     T=data["T"].values,
    ...     Delta=data["Delta"].values,
    ...     S_c_predict=km.predict,
    ... )
    """
    T = np.asarray(T, dtype=float).ravel()
    Delta = np.asarray(Delta, dtype=float).ravel()

    if T.shape[0] != Delta.shape[0]:
        raise ValueError("T and Delta must have same length")

    Sc = np.asarray(S_c_predict(T), dtype=float)
    Sc = np.maximum(Sc, clip)
    w = Delta / Sc

    return w

