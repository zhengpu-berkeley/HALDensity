"""Hyperparameter tuners for censored density estimation.

Provides tuners for both right-censored and interval-censored data.
"""

from ._base import TuningResult, OverSmoothInitRecord, OverSmoothEMRecord

from .right_tuners import (
    RightCensoredInitTuner,
    RightCensoredEMTuner,
    RightCensoredJointTuner,
    # Backward compatibility aliases
    RightCensoredOptunaHyperparameterTuner,
    RightCensoredEMStageOverSmoothTuner,
)

from .interval_tuners import (
    IntervalCensoredInitTuner,
    IntervalCensoredEMTuner,
    IntervalCensoredJointTuner,
    # Backward compatibility aliases
    IntervalCensoredOptunaHyperparameterTuner,
    IntervalCensoredEMStageOverSmoothTuner,
)

__all__ = [
    # Result types
    "TuningResult",
    "OverSmoothInitRecord",
    "OverSmoothEMRecord",
    # Right-censored tuners (new names)
    "RightCensoredInitTuner",
    "RightCensoredEMTuner",
    "RightCensoredJointTuner",
    # Interval-censored tuners (new names)
    "IntervalCensoredInitTuner",
    "IntervalCensoredEMTuner",
    "IntervalCensoredJointTuner",
    # Backward compatibility aliases
    "RightCensoredOptunaHyperparameterTuner",
    "RightCensoredEMStageOverSmoothTuner",
    "IntervalCensoredOptunaHyperparameterTuner",
    "IntervalCensoredEMStageOverSmoothTuner",
]
