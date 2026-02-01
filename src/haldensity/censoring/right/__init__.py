"""Right-censoring density estimation module.

Provides HAL-MLE and EM-based estimators for right-censored data.
"""

from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .estimators import (
    RightCensoredInitEstimator,
    RightCensoredIPCWEstimator,  # Alias for backward compatibility
    RightCensoredEMEstimator,
    RightCensoredEMStage,
)
from .metrics import incomplete_loglik, mi_complete_loglik

__all__ = [
    # Utilities
    "KaplanMeier",
    "compute_ipcw_weights",
    # Estimators (new names)
    "RightCensoredInitEstimator",
    "RightCensoredEMEstimator",
    "RightCensoredEMStage",
    # Backward compatibility alias
    "RightCensoredIPCWEstimator",
    # Metrics
    "incomplete_loglik",
    "mi_complete_loglik",
]
