"""Censored data density estimation module.

This module provides tools for HAL-based density estimation with right-censored data:

Estimators:
- WeightedCVXPYEstimator: IPCW-weighted HAL density estimator
- EMIPCWEstimator: Combined IPCW initialization + EM refinement
- EMStage: Standalone EM refinement stage (works with any initial estimator)

Tuners:
- CensoredOptunaHyperparameterTuner: Joint CV tuner for IPCW or EM estimators
- TwoStageCensoredTuner: Two-stage tuner (fast IPCW + focused EM)
- EMStageTuner: Standalone tuner for EM refinement

Utilities:
- KaplanMeier: Kaplan-Meier estimator for censoring survival
- compute_ipcw_weights: Compute IPCW weights
- incomplete_loglik, mi_complete_loglik, kl_divergence: Evaluation metrics
- pipelines: Convenience functions for common workflows
"""

from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .metrics import (
    incomplete_loglik,
    mi_complete_loglik,
    kl_divergence,
)
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .em import EMStage, EMStageResult, EMIPCWEstimator
from .optuna_cv import (
    CensoredOptunaHyperparameterTuner,
    TwoStageCensoredTuner,
    EMStageTuner,
)
from . import pipelines

__all__ = [
    # Utilities
    "KaplanMeier",
    "compute_ipcw_weights",
    # Metrics
    "incomplete_loglik",
    "mi_complete_loglik",
    "kl_divergence",
    # Estimators
    "WeightedCVXPYEstimator",
    "EMStage",
    "EMStageResult",
    "EMIPCWEstimator",
    # Tuners
    "CensoredOptunaHyperparameterTuner",
    "TwoStageCensoredTuner",
    "EMStageTuner",
    # Pipelines
    "pipelines",
]
