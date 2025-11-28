"""Censored data density estimation module.

This module provides tools for HAL-based density estimation with right-censored data.
Future versions will support left-censoring and interval-censoring.

Estimators
----------
WeightedCVXPYEstimator : IPCW-weighted HAL density estimator (alias for RightCensoredIPCWEstimator)
EMIPCWEstimator : Combined IPCW initialization + EM refinement (alias for RightCensoredEMEstimator)
EMStage : Standalone EM refinement stage (works with any initial estimator)

Tuners
------
CensoredOptunaHyperparameterTuner : Joint CV tuner for IPCW or EM estimators
TwoStageCensoredTuner : Two-stage tuner (fast IPCW + focused EM)
EMStageTuner : Standalone tuner for EM refinement

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
...     WeightedCVXPYEstimator,
...     EMIPCWEstimator,
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
>>> ipcw_est = WeightedCVXPYEstimator(norm_constraint=50.0).fit(
...     pd.DataFrame({"W1": uncensored["T"]}),
...     sample_weights=weights[data["Delta"] == 1],
... )
>>> 
>>> # Or use EM estimator directly
>>> em_est = EMIPCWEstimator(norm_constraint=50.0, m_imputations=20).fit(data)
"""

# Re-export from right-censoring module with backward-compatible aliases
from .right import (
    KaplanMeier,
    compute_ipcw_weights,
    RightCensoredIPCWEstimator,
    RightCensoredEMEstimator,
    EMStage,
    incomplete_loglik,
    mi_complete_loglik,
)

# Backward-compatible aliases
WeightedCVXPYEstimator = RightCensoredIPCWEstimator
EMIPCWEstimator = RightCensoredEMEstimator

# Core models
from .core.models import EMStageResult

# Tuners
from .tuners import (
    CensoredOptunaHyperparameterTuner,
    TwoStageCensoredTuner,
    EMStageTuner,
)

# Shared utilities
from .utils.common_metrics import kl_divergence

# Submodules
from . import pipelines
from . import right
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
    # Estimators (with aliases)
    "WeightedCVXPYEstimator",
    "RightCensoredIPCWEstimator",
    "EMIPCWEstimator",
    "RightCensoredEMEstimator",
    "EMStage",
    "EMStageResult",
    # Tuners
    "CensoredOptunaHyperparameterTuner",
    "TwoStageCensoredTuner",
    "EMStageTuner",
    # Submodules
    "pipelines",
    "right",
    "core",
    "tuners",
    "utils",
]
