"""EM-based density estimator for interval-censored data."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.censoring.core.models import EM_DEFAULTS
from haldensity.censoring.interval.midpoint_estimator import IntervalCensoredMidpointEstimator
from haldensity.censoring.interval.em_stage import IntervalCensoredEMStage

logger = logging.getLogger(__name__)


class IntervalCensoredEMEstimator(BaseEstimator):
    """Midpoint-initialized parametric EM for interval-censored data on [0, 1]."""

    def __init__(
        self,
        tol: float = EM_DEFAULTS.tol,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        m_imputations: int = EM_DEFAULTS.m_imputations,
        max_em_iter: int = EM_DEFAULTS.max_em_iter,
        em_tol: float = EM_DEFAULTS.em_tol,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        verbose: bool = False,
        init_solver: str = EM_DEFAULTS.init_solver,
        m_step_solver: str = EM_DEFAULTS.m_step_solver,
        init_norm_constraint: Optional[float] = None,
        m_step_norm_constraint: Optional[float] = None,
        e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        rng_seed: int = 0,
        L_col: str = "L",
        R_col: str = "R",
    ):
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = float(norm_constraint)
        self.n_grid_points = int(n_grid_points)
        self.m_imputations = int(m_imputations)
        self.max_em_iter = int(max_em_iter)
        self.em_tol = float(em_tol)
        self.verbose = bool(verbose)
        self.init_solver = str(init_solver)
        self.m_step_solver = str(m_step_solver)
        self.init_norm_constraint = (
            float(init_norm_constraint) if init_norm_constraint is not None else self.norm_constraint
        )
        self.m_step_norm_constraint = (
            float(m_step_norm_constraint) if m_step_norm_constraint is not None else self.norm_constraint
        )
        self.e_step_n_grid = int(e_step_n_grid)
        self.rng_seed = int(rng_seed)
        self.L_col = str(L_col)
        self.R_col = str(R_col)

        # Fitted state
        self.theta_path_: list[np.ndarray] = []
        self.em_iterations_: int = 0
        self.em_converged_: bool = False
        self.uncensored_augmented_: Optional[pd.DataFrame] = None
        self._current_estimator: Optional[BaseEstimator] = None
        self._em_stage_result = None

    def _init_midpoint(self, data: pd.DataFrame) -> IntervalCensoredMidpointEstimator:
        return IntervalCensoredMidpointEstimator(
            tol=self.tol,
            norm_constraint=self.init_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.init_solver,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            include_intercept_in_constraint=False,
            use_secondary_solver=False,
        ).fit(data, L_col=self.L_col, R_col=self.R_col)

    def fit(self, data: pd.DataFrame) -> "IntervalCensoredEMEstimator":
        if self.L_col not in data.columns or self.R_col not in data.columns:
            raise ValueError(f"data must contain columns {self.L_col!r} and {self.R_col!r}")

        if self.verbose:
            logger.info("Initializing midpoint HAL-MLE...")
        init_est = self._init_midpoint(data)

        em_stage = IntervalCensoredEMStage(
            m_imputations=self.m_imputations,
            max_em_iter=self.max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=self.m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            tol=self.tol,
            m_step_solver=self.m_step_solver,
            include_intercept_in_constraint=True,
            verbose=self.verbose,
            rng_seed=self.rng_seed,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            e_step_n_grid=self.e_step_n_grid,
            L_col=self.L_col,
            R_col=self.R_col,
        )

        em_result = em_stage.run(initial_estimator=init_est, data=data)
        self._em_stage_result = em_result

        self.theta_path_ = em_result.theta_path
        self.em_iterations_ = em_result.em_iterations
        self.em_converged_ = em_result.em_converged
        self.uncensored_augmented_ = em_result.final_augmented_data

        final_est = em_result.final_estimator
        self._current_estimator = final_est

        # Copy final estimator state to self
        if final_est.theta_hat is None or final_est._grid_points_hal is None:
            raise RuntimeError("EM stage failed: final estimator missing theta/grid")

        self.theta_hat = final_est.theta_hat.copy()
        self._grid_points_hal = final_est._grid_points_hal.copy()
        self.grid_midpoints = final_est.grid_midpoints.copy() if final_est.grid_midpoints is not None else None
        self.delta_j = final_est.delta_j.copy() if final_est.delta_j is not None else None
        self.grid_points = final_est.grid_points.copy() if final_est.grid_points is not None else None
        self.grid_points_hal_selected = (
            final_est.grid_points_hal_selected.copy()
            if final_est.grid_points_hal_selected is not None
            else None
        )
        self.basis_names = final_est.basis_names
        self.fitted_theta_dict = final_est.fitted_theta_dict
        self.is_fitted = True
        return self

    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update(
            {
                "theta_path": [theta.tolist() for theta in self.theta_path_],
                "em_iterations": self.em_iterations_,
                "em_converged": self.em_converged_,
            }
        )
        return base

    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density()
        return super().get_density()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density_at_points(points)
        return super().get_density_at_points(points)


