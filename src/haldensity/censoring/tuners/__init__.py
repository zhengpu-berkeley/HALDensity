"""Hyperparameter tuners for censored density estimation.

Provides tuners for both right-censored and interval-censored data.
"""

from ._base import TuningResult, OverSmoothInitRecord, OverSmoothEMRecord, CVOversmoothRecord

from .right_tuners import (
    RightCensoredInitTuner,
    RightCensoredEMTuner,
    RightCensoredJointTuner,
    RightCensoredCVOversmoothEMTuner,
    RightCensoredObservedFISTATuner,
    RightCensoredObservedFPGDTuner,
    RightCensoredObservedL1Tuner,
)

from .interval_tuners import (
    IntervalCensoredInitTuner,
    IntervalCensoredEMTuner,
    IntervalCensoredJointTuner,
    IntervalCensoredCVOversmoothEMTuner,
    IntervalCensoredFISTATuner,
    IntervalCensoredProjectedGDTuner,
)

__all__ = [
    # Result types
    "TuningResult",
    "OverSmoothInitRecord",
    "OverSmoothEMRecord",
    "CVOversmoothRecord",
    # Right-censored tuners
    "RightCensoredInitTuner",
    "RightCensoredEMTuner",
    "RightCensoredJointTuner",
    "RightCensoredCVOversmoothEMTuner",
    "RightCensoredObservedFISTATuner",
    "RightCensoredObservedFPGDTuner",
    "RightCensoredObservedL1Tuner",
    # Interval-censored tuners
    "IntervalCensoredInitTuner",
    "IntervalCensoredEMTuner",
    "IntervalCensoredJointTuner",
    "IntervalCensoredCVOversmoothEMTuner",
    "IntervalCensoredFISTATuner",
    "IntervalCensoredProjectedGDTuner",
]
