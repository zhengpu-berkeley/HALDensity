"""Interval-censored density estimation module.

Provides HAL-MLE and EM-based estimators for interval-censored data.
"""

from .estimators import (
    IntervalCensoredInitEstimator,
    IntervalCensoredMidpointEstimator,  # Alias for backward compatibility
    IntervalCensoredEMEstimator,
    IntervalCensoredEMStage,
)
from .metrics import incomplete_loglik_interval

__all__ = [
    # Estimators (new names)
    "IntervalCensoredInitEstimator",
    "IntervalCensoredEMEstimator",
    "IntervalCensoredEMStage",
    # Backward compatibility alias
    "IntervalCensoredMidpointEstimator",
    # Metrics
    "incomplete_loglik_interval",
]
