"""Simulation helpers for interval-censored data.

These are intentionally lightweight utilities for generating (L, R) intervals
from latent event times T under common non-informative inspection schemes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


IntervalMechanism = Literal["inspection_uniform", "current_status_uniform", "width_uniform"]


def _as_rng(random_state: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(random_state)


def interval_censor_inspection_uniform(
    event_times: np.ndarray,
    *,
    n_inspections: int = 8,
    t_min: float = 0.0,
    t_max: float = 1.0,
    random_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inspection-time bracketing with inspection times independent of T.

    For each i, draw `n_inspections` inspection times uniformly on [t_min, t_max],
    sort them, add boundaries (t_min, t_max), and return the bracketing interval
    [L_i, R_i] such that L_i < T_i <= R_i (ties go to the right bin).
    """
    if n_inspections <= 0:
        raise ValueError("n_inspections must be positive")
    if not (t_min < t_max):
        raise ValueError("require t_min < t_max")

    T = np.asarray(event_times, dtype=float)
    n = int(T.shape[0])
    g = _as_rng(random_state, rng)

    inspections = g.uniform(t_min, t_max, size=(n, n_inspections))
    inspections.sort(axis=1)

    C = np.concatenate(
        [
            np.full((n, 1), t_min, dtype=float),
            inspections,
            np.full((n, 1), t_max, dtype=float),
        ],
        axis=1,
    )

    # Bracket index j with C_j < T <= C_{j+1}. Use <= with side="right" behavior.
    idx = np.sum(C <= T[:, None], axis=1) - 1
    idx = np.clip(idx, 0, C.shape[1] - 2)

    L = C[np.arange(n), idx]
    R = C[np.arange(n), idx + 1]
    return pd.DataFrame({"L": L, "R": R})


def interval_censor_current_status_uniform(
    event_times: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    random_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Current-status interval censoring with independent uniform inspection time.

    Draw one inspection time C ⟂ T uniformly on [t_min, t_max]. Observe:
    - [t_min, C] if T <= C
    - [C, t_max] otherwise
    """
    if not (t_min < t_max):
        raise ValueError("require t_min < t_max")

    T = np.asarray(event_times, dtype=float)
    n = int(T.shape[0])
    g = _as_rng(random_state, rng)

    C = g.uniform(t_min, t_max, size=n)
    is_event = T <= C
    L = np.where(is_event, t_min, C)
    R = np.where(is_event, C, t_max)
    return pd.DataFrame({"L": L, "R": R})


def interval_censor_width_uniform(
    event_times: np.ndarray,
    *,
    width_low: float = 0.05,
    width_high: float = 0.15,
    min_width: float = 0.01,
    t_min: float = 0.0,
    t_max: float = 1.0,
    random_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Legacy mechanism: construct an interval around T (informative).

    This is retained for backward compatibility with older examples/tests where
    [L, R] was created directly from the latent event time.
    """
    if min_width <= 0:
        raise ValueError("min_width must be positive")
    if width_low <= 0 or width_high <= 0 or width_high < width_low:
        raise ValueError("require 0 < width_low <= width_high")
    if not (t_min < t_max):
        raise ValueError("require t_min < t_max")

    T = np.asarray(event_times, dtype=float)
    n = int(T.shape[0])
    g = _as_rng(random_state, rng)

    widths = g.uniform(width_low, width_high, size=n)
    widths = np.maximum(widths, min_width)
    left_shifts = g.uniform(0.0, 1.0, size=n) * widths

    L = np.clip(T - left_shifts, t_min, t_max)
    R = np.clip(L + widths, t_min, t_max)

    R = np.maximum(R, T)
    R = np.minimum(t_max, np.maximum(R, L + min_width))

    invalid = R <= L
    if np.any(invalid):
        L[invalid] = np.minimum(L[invalid], t_max - min_width)
        L[invalid] = np.maximum(L[invalid], t_min)
        R[invalid] = np.minimum(t_max, L[invalid] + min_width)
        R[invalid] = np.maximum(R[invalid], T[invalid])

    return pd.DataFrame({"L": L, "R": R})


def interval_censor(
    event_times: np.ndarray,
    *,
    mechanism: IntervalMechanism = "inspection_uniform",
    **kwargs,
) -> pd.DataFrame:
    """Dispatch helper for interval censoring mechanisms."""
    if mechanism == "inspection_uniform":
        return interval_censor_inspection_uniform(event_times, **kwargs)
    if mechanism == "current_status_uniform":
        return interval_censor_current_status_uniform(event_times, **kwargs)
    if mechanism == "width_uniform":
        return interval_censor_width_uniform(event_times, **kwargs)
    raise ValueError(f"Unsupported interval censoring mechanism: {mechanism}")

