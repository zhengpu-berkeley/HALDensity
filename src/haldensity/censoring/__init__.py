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
RightCensoredEMTuner : Stage 2 tuner (oversmooth or CV mode)
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
    RightCensoredIPCWEstimator,  # Backward compatibility alias
    RightCensoredEMEstimator,
    RightCensoredEMStage,
    incomplete_loglik,
    mi_complete_loglik,
)

# Interval-censoring module
from .interval import (
    IntervalCensoredInitEstimator,
    IntervalCensoredMidpointEstimator,  # Backward compatibility alias
    IntervalCensoredEMEstimator,
    IntervalCensoredEMStage,
    incomplete_loglik_interval,
)

# Core models
from .core.models import RightCensoredEMStageResult

# Tuners
from .tuners import (
    # Result types
    TuningResult,
    # Right-censored tuners (new names)
    RightCensoredInitTuner,
    RightCensoredEMTuner,
    RightCensoredJointTuner,
    # Interval-censored tuners (new names)
    IntervalCensoredInitTuner,
    IntervalCensoredEMTuner,
    IntervalCensoredJointTuner,
    # Backward compatibility aliases
    RightCensoredOptunaHyperparameterTuner,
    RightCensoredEMStageOverSmoothTuner,
    IntervalCensoredOptunaHyperparameterTuner,
    IntervalCensoredEMStageOverSmoothTuner,
)

# Shared utilities
from .utils.common_metrics import kl_divergence

# Submodules
from . import pipelines
from . import right
from . import interval
from . import core
from . import tuners
from . import utils

__all__ = [
    # Utilities
    "KaplanMeier",
    "compute_ipcw_weights",
    # Metrics
    "incomplete_loglik",
    "incomplete_loglik_interval",
    "mi_complete_loglik",
    "kl_divergence",
    # Result types
    "TuningResult",
    "RightCensoredEMStageResult",
    # Right-censored estimators (new names)
    "RightCensoredInitEstimator",
    "RightCensoredEMEstimator",
    "RightCensoredEMStage",
    # Right-censored estimators (backward compatibility)
    "RightCensoredIPCWEstimator",
    # Interval-censored estimators (new names)
    "IntervalCensoredInitEstimator",
    "IntervalCensoredEMEstimator",
    "IntervalCensoredEMStage",
    # Interval-censored estimators (backward compatibility)
    "IntervalCensoredMidpointEstimator",
    # Right-censored tuners (new names)
    "RightCensoredInitTuner",
    "RightCensoredEMTuner",
    "RightCensoredJointTuner",
    # Interval-censored tuners (new names)
    "IntervalCensoredInitTuner",
    "IntervalCensoredEMTuner",
    "IntervalCensoredJointTuner",
    # Tuners (backward compatibility)
    "RightCensoredOptunaHyperparameterTuner",
    "RightCensoredEMStageOverSmoothTuner",
    "IntervalCensoredOptunaHyperparameterTuner",
    "IntervalCensoredEMStageOverSmoothTuner",
    # Submodules
    "pipelines",
    "right",
    "interval",
    "core",
    "tuners",
    "utils",
]
