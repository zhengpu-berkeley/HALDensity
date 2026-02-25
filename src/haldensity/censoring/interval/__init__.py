"""Interval-censored density estimation module.

Provides HAL-MLE and EM-based estimators for interval-censored data.
"""

from .estimators import (
    IntervalCensoredInitEstimator,
    IntervalCensoredEMEstimator,
    IntervalCensoredEMStage,
)
from .metrics import incomplete_loglik_interval
from .simulate import (
    interval_censor,
    interval_censor_current_status_uniform,
    interval_censor_inspection_uniform,
    interval_censor_width_uniform,
)

__all__ = [
    # Estimators
    "IntervalCensoredInitEstimator",
    "IntervalCensoredEMEstimator",
    "IntervalCensoredEMStage",
    # Metrics
    "incomplete_loglik_interval",
    # Simulation helpers
    "interval_censor",
    "interval_censor_inspection_uniform",
    "interval_censor_current_status_uniform",
    "interval_censor_width_uniform",
]
