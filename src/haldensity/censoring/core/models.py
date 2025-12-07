"""Pydantic models and dataclasses for censored data density estimation.

Contains result containers and default configuration models.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class EMDefaults(BaseModel):
    """Default parameters for EM algorithm.

    These defaults are shared across RightCensoredEMStage, EMIPCWEstimator, and tuners.
    """

    model_config = ConfigDict(frozen=True)

    tol: float = 1e-4
    m_imputations: int = 100
    max_em_iter: int = 10
    em_tol: float = 1e-1
    e_step_n_grid: int = 200
    use_sc_adjustment: bool = False
    m_step_solver: str = "ECOS"
    init_solver: str = "ECOS"


class TunerDefaults(BaseModel):
    """Default parameters for hyperparameter tuners.

    These defaults are shared across all Optuna-based tuners.
    """

    model_config = ConfigDict(frozen=True)

    cv_folds: int = 5
    random_state: int = 42
    n_grid_points: int = 200
    solver: str = "ECOS"
    use_secondary_solver: bool = True

    # Conservative adjustment settings (to reduce CV oversmoothing)
    # Finds smallest norm_constraint where CV_LL >= max_CV_LL - k% * SD
    use_conservative_adjustment: bool = True
    conservative_k_percent: float = 0.05  # 5% of SD as threshold
    conservative_max_steps: int = 50  # Maximum search steps
    conservative_step_pct: float = 0.02  # 2% step size per step


# Default instances for convenience
EM_DEFAULTS = EMDefaults()
TUNER_DEFAULTS = TunerDefaults()


class RightCensoredEMStageResult:
    """Container for RightCensoredEMStage results.

    Attributes
    ----------
    final_estimator : Any
        The fitted estimator after EM refinement.
    theta_path : list[np.ndarray]
        History of theta values across EM iterations.
    em_iterations : int
        Number of EM iterations performed.
    em_converged : bool
        Whether EM converged before reaching max iterations.
    final_augmented_data : pd.DataFrame | None
        The final augmented (imputed) dataset from the last E-step.
    """

    def __init__(
        self,
        final_estimator,
        theta_path: list[np.ndarray],
        em_iterations: int,
        em_converged: bool,
        final_augmented_data: Optional[pd.DataFrame] = None,
    ):
        self.final_estimator = final_estimator
        self.theta_path = theta_path
        self.em_iterations = em_iterations
        self.em_converged = em_converged
        self.final_augmented_data = final_augmented_data

    def __repr__(self) -> str:
        return (
            f"RightCensoredEMStageResult(em_iterations={self.em_iterations}, "
            f"em_converged={self.em_converged}, "
            f"theta_path_len={len(self.theta_path)})"
        )

