"""Censored data density estimation module.

Provides HAL-based density estimation for censored data:
- Right-censoring: IPCW-weighted HAL-MLE and EM refinement
- Interval-censoring: Midpoint initialization and EM refinement

Estimators
----------
RightCensoredInitEstimator : Stage 1 IPCW-weighted HAL for right-censored data
RightCensoredEMEstimator : Combined Stage 1 + Stage 2 EM for right-censored data
RightCensoredEMStage : Standalone EM refinement stage

IntervalCensoredInitEstimator : Stage 1 midpoint-imputed HAL for interval-censored data
IntervalCensoredEMEstimator : Combined Stage 1 + Stage 2 EM for interval-censored data
IntervalCensoredEMStage : Standalone EM refinement stage

Tuners
------
RightCensoredInitTuner : Stage 1 tuner (Optuna CV)
RightCensoredEMTuner : Stage 2 tuner (oversmooth or direct no-oversmooth mode)
RightCensoredJointTuner : Convenience wrapper (Stage 1 + Stage 2)

IntervalCensoredInitTuner : Stage 1 tuner for interval-censored data
IntervalCensoredEMTuner : Stage 2 tuner for interval-censored data
IntervalCensoredJointTuner : Convenience wrapper for interval-censored data

Utilities
---------
KaplanMeier : Kaplan-Meier estimator for censoring survival
compute_ipcw_weights : Compute IPCW weights
incomplete_loglik : Incomplete-data log-likelihood for right-censored data
incomplete_loglik_interval : Incomplete-data log-likelihood for interval-censored data
kl_divergence : KL divergence between true and estimated density
TuningResult : NamedTuple for tuner output
"""

# Right-censoring module
from .right import (
    KaplanMeier,
    compute_ipcw_weights,
    RightCensoredInitEstimator,
    RightCensoredEMEstimator,
    RightCensoredEMStage,
    incomplete_loglik,
    mi_complete_loglik,
)

# Interval-censoring module
from .interval import (
    IntervalCensoredInitEstimator,
    IntervalCensoredEMEstimator,
    IntervalCensoredEMStage,
    incomplete_loglik_interval,
    interval_censor,
    interval_censor_current_status_uniform,
    interval_censor_inspection_uniform,
    interval_censor_width_uniform,
)

# Core models and defaults
from ._defaults import EMStageResult

# Shared base estimator
from ._base_mle import WeightedHALMLEEstimator

# Tuners
from .tuners import (
    TuningResult,
    RightCensoredInitTuner,
    RightCensoredEMTuner,
    RightCensoredJointTuner,
    RightCensoredCVOversmoothEMTuner,
    IntervalCensoredInitTuner,
    IntervalCensoredEMTuner,
    IntervalCensoredJointTuner,
    IntervalCensoredCVOversmoothEMTuner,
)

# Shared utilities
from .utils.common_metrics import kl_divergence

# Submodules
from . import right
from . import interval
from . import tuners
from . import utils

__all__ = [
    # Utilities
    "KaplanMeier",
    "compute_ipcw_weights",
    "interval_censor",
    "interval_censor_inspection_uniform",
    "interval_censor_current_status_uniform",
    "interval_censor_width_uniform",
    # Metrics
    "incomplete_loglik",
    "incomplete_loglik_interval",
    "mi_complete_loglik",
    "kl_divergence",
    # Result types
    "TuningResult",
    "EMStageResult",
    # Shared base estimator
    "WeightedHALMLEEstimator",
    # Right-censored estimators
    "RightCensoredInitEstimator",
    "RightCensoredEMEstimator",
    "RightCensoredEMStage",
    # Interval-censored estimators
    "IntervalCensoredInitEstimator",
    "IntervalCensoredEMEstimator",
    "IntervalCensoredEMStage",
    # Right-censored tuners
    "RightCensoredInitTuner",
    "RightCensoredEMTuner",
    "RightCensoredJointTuner",
    "RightCensoredCVOversmoothEMTuner",
    # Interval-censored tuners
    "IntervalCensoredInitTuner",
    "IntervalCensoredEMTuner",
    "IntervalCensoredJointTuner",
    "IntervalCensoredCVOversmoothEMTuner",
    # Submodules
    "right",
    "interval",
    "tuners",
    "utils",
]
