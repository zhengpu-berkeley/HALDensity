import numpy as np
from typing import Callable


def compute_ipcw_weights(
    T: np.ndarray,
    Delta: np.ndarray,
    S_c_predict: Callable[[np.ndarray], np.ndarray],
    clip: float = 1e-6,
) -> np.ndarray:
    """
    Compute inverse probability of censoring weights: w_i = Delta_i / max(S_c(T_i), clip)
    Args:
        T: observed times Y in [0,1]
        Delta: event indicators (1=uncensored, 0=censored)
        S_c_predict: function returning S_c(t) for array t
        clip: lower bound for stability
    """
    T = np.asarray(T, dtype=float).ravel()
    Delta = np.asarray(Delta, dtype=float).ravel()
    if T.shape[0] != Delta.shape[0]:
        raise ValueError("T and Delta must have same length")
    Sc = np.asarray(S_c_predict(T), dtype=float)
    Sc = np.maximum(Sc, clip)
    w = Delta / Sc
    return w


