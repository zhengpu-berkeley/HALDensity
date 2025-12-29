"""Censored data density estimation module.

This module provides tools for HAL-based density estimation with right-censored data.
This version supports:
- Right-censoring
- Interval-censoring (midpoint init + EM with truncated-to-interval imputation)

Estimators
----------
RightCensoredIPCWEstimator : IPCW-weighted HAL density estimator for right-censored data
RightCensoredEMEstimator : Combined IPCW initialization + EM refinement
RightCensoredEMStage : Standalone EM refinement stage (works with any initial estimator)

Tuners
------
RightCensoredOptunaHyperparameterTuner : Joint CV tuner for IPCW or EM estimators
RightCensoredTwoStageTuner : Two-stage tuner (fast IPCW + focused EM)
RightCensoredEMStageTuner : Standalone tuner for EM refinement

Utilities
---------
KaplanMeier : Kaplan-Meier estimator for censoring survival
compute_ipcw_weights : Compute IPCW weights
incomplete_loglik : Incomplete-data log-likelihood for right-censored data
mi_complete_loglik : MI-pooled complete-data log-likelihood
kl_divergence : KL divergence between true and estimated density

Submodules
----------
right : Right-censoring specific implementations
core : Core protocols and models
tuners : Hyperparameter tuning classes
utils : Shared utilities
pipelines : Convenience workflow functions

Examples
--------
>>> from haldensity.censoring import (
...     KaplanMeier,
...     compute_ipcw_weights,
...     RightCensoredIPCWEstimator,
...     RightCensoredEMEstimator,
... )
>>> 
>>> # Fit Kaplan-Meier for censoring survival
>>> km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
>>> 
>>> # Compute IPCW weights
>>> weights = compute_ipcw_weights(data["T"], data["Delta"], km.predict)
>>> 
>>> # Fit IPCW estimator
>>> uncensored = data[data["Delta"] == 1]
>>> ipcw_est = RightCensoredIPCWEstimator(norm_constraint=50.0).fit(
...     pd.DataFrame({"W1": uncensored["T"]}),
...     sample_weights=weights[data["Delta"] == 1],
... )
>>> 
>>> # Or use EM estimator directly
>>> em_est = RightCensoredEMEstimator(norm_constraint=50.0, m_imputations=20).fit(data)
"""

# Right-censoring module
from .right import (
    KaplanMeier,
    compute_ipcw_weights,
    RightCensoredIPCWEstimator,
    RightCensoredEMEstimator,
    RightCensoredEMStage,
    incomplete_loglik,
    mi_complete_loglik,
)

# Interval-censoring module
from .interval import (
    IntervalCensoredMidpointEstimator,
    IntervalCensoredEMStage,
    IntervalCensoredEMEstimator,
    incomplete_loglik_interval,
)

# Core models
from .core.models import RightCensoredEMStageResult

# Tuners
from .tuners import (
    RightCensoredOptunaHyperparameterTuner,
    RightCensoredTwoStageTuner,
    RightCensoredEMStageTuner,
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
    "mi_complete_loglik",
    "kl_divergence",
    # Estimators (explicit names - preferred)
    "RightCensoredIPCWEstimator",
    "RightCensoredEMEstimator",
    "RightCensoredEMStage",
    "RightCensoredEMStageResult",
    # Interval-censoring estimators
    "IntervalCensoredMidpointEstimator",
    "IntervalCensoredEMEstimator",
    "IntervalCensoredEMStage",
    # Tuners
    "RightCensoredOptunaHyperparameterTuner",
    "RightCensoredTwoStageTuner",
    "RightCensoredEMStageTuner",
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
    # Interval metrics
    "incomplete_loglik_interval",
]
