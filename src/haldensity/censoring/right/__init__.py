"""Right-censoring density estimation module.

Provides HAL-MLE and EM-based estimators for right-censored data.
"""

from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .estimators import (
    RightCensoredInitEstimator,
    RightCensoredEMEstimator,
    RightCensoredEMStage,
)
from .observed_mle import (
    build_right_censored_knot_grid,
    RightCensoredObservedFISTAEstimator,
    RightCensoredObservedProjectedGDEstimator,
    RightCensoredObservedFPGDEstimator,
    RightCensoredObservedL1MLE,
    right_censored_observed_loglik_and_gradient,
)
from .metrics import incomplete_loglik, mi_complete_loglik, ipcw_loglik
from .comparison import (
    fit_right_censored_cv_ipcw_l1mle_plugin,
    fit_right_censored_cv_ipcw_l1mle_undersmoothed_plugin,
)

__all__ = [
    # Utilities
    "KaplanMeier",
    "compute_ipcw_weights",
    "build_right_censored_knot_grid",
    # Estimators
    "RightCensoredInitEstimator",
    "RightCensoredEMEstimator",
    "RightCensoredEMStage",
    "RightCensoredObservedFISTAEstimator",
    "RightCensoredObservedProjectedGDEstimator",
    "RightCensoredObservedFPGDEstimator",
    "RightCensoredObservedL1MLE",
    # Metrics
    "incomplete_loglik",
    "mi_complete_loglik",
    "ipcw_loglik",
    "right_censored_observed_loglik_and_gradient",
    # CV-IPCW-L1MLE plug-in survival APIs
    "fit_right_censored_cv_ipcw_l1mle_plugin",
    "fit_right_censored_cv_ipcw_l1mle_undersmoothed_plugin",
]
