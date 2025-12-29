"""Interval-censored density estimation.

This subpackage implements HAL-based density estimation when each observation is an
interval [L, R] known to contain the true event time T*.

Key design (mirrors right-censored implementation):
- Stage 1: midpoint-imputed HAL-MLE initializer (deterministic).
- Stage 2: parametric EM refinement with multiple imputation in the E-step:
  sample T* ~ f_theta truncated to [L, R], and fit a weighted HAL-MLE in the M-step
  while keeping the knot structure fixed.
"""

from .midpoint_estimator import IntervalCensoredMidpointEstimator
from .em_stage import IntervalCensoredEMStage
from .em_estimator import IntervalCensoredEMEstimator
from .metrics import incomplete_loglik_interval

__all__ = [
    "IntervalCensoredMidpointEstimator",
    "IntervalCensoredEMStage",
    "IntervalCensoredEMEstimator",
    "incomplete_loglik_interval",
]


