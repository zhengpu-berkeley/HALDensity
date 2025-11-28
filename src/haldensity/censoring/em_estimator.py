from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from haldensity.estimation.base_estimator import BaseEstimator
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .em_stage import EMStage, EMStageResult


class EMIPCWEstimator(BaseEstimator):
    """EM with multiple imputation for right-censored data on [0, 1].
    
    This estimator combines:
    1. IPCW-HAL-MLE initialization on uncensored observations
    2. EM refinement via EMStage with multiple imputation for censored observations
    """

    def __init__(
        self,
        tol: float = 1e-4,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        m_imputations: int = 20,
        max_em_iter: int = 50,
        em_tol: float = 1e-3,
        use_sc_adjustment: bool = False,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        verbose: bool = False,
        init_solver: str = "SCS",
        m_step_solver: str = "ECOS",
        init_norm_constraint: Optional[float] = None,
        m_step_norm_constraint: Optional[float] = None,
        e_step_n_grid: int = 1000,
        rng_seed: int = 0,
    ):
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.m_imputations = m_imputations
        self.max_em_iter = max_em_iter
        self.em_tol = em_tol
        self.use_sc_adjustment = use_sc_adjustment
        self.verbose = verbose
        self.init_solver = init_solver
        self.m_step_solver = m_step_solver
        self.m_step_solver_sequence: list[str] = []
        for cand in (m_step_solver, "CLARABEL", "ECOS", "SCS"):
            if cand not in self.m_step_solver_sequence:
                self.m_step_solver_sequence.append(cand)
        self.init_norm_constraint = init_norm_constraint if init_norm_constraint is not None else norm_constraint
        self.m_step_norm_constraint = (
            m_step_norm_constraint if m_step_norm_constraint is not None else norm_constraint
        )
        self.e_step_n_grid = e_step_n_grid
        self.rng_seed = rng_seed
        self.km_: Optional[KaplanMeier] = None
        self.uncensored_augmented_: Optional[pd.DataFrame] = None
        self.theta_path_: list[np.ndarray] = []
        self.em_iterations_: int = 0
        self.em_converged_: bool = False
        self._current_estimator: Optional[BaseEstimator] = None
        self._em_stage_result: Optional[EMStageResult] = None

    def _init_ipcw(self, data: pd.DataFrame) -> WeightedCVXPYEstimator:
        """Fit initial IPCW-weighted HAL estimator on uncensored observations."""
        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
        self.km_ = km
        T_vals = np.asarray(data["T"].values, dtype=float)
        Delta_vals = np.asarray(data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(
            T=T_vals,
            Delta=Delta_vals,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )
        uncensored_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        w_unc = weights[uncensored_mask]
        return WeightedCVXPYEstimator(
            tol=self.tol,
            norm_constraint=self.init_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            use_secondary_solver=False,
            solver=self.init_solver,
            include_intercept_in_constraint=True,
        ).fit(df_unc, sample_weights=w_unc)

    def fit(self, data: pd.DataFrame) -> "EMIPCWEstimator":
        """Fit the EM-IPCW-HAL estimator.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
            
        Returns
        -------
        self
        """
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")

        # Step 1: Fit initial IPCW estimator
        if self.verbose:
            print("Initializing IPCW-HAL-MLE...")
        init_est = self._init_ipcw(data)
        self._debug_estimator_state("init-ipcw", init_est)

        # Step 2: Run EM stage
        em_stage = EMStage(
            m_imputations=self.m_imputations,
            max_em_iter=self.max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=self.m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            use_sc_adjustment=self.use_sc_adjustment,
            e_step_n_grid=self.e_step_n_grid,
            tol=self.tol,
            m_step_solver=self.m_step_solver,
            m_step_solver_sequence=self.m_step_solver_sequence,
            include_intercept_in_constraint=True,
            verbose=self.verbose,
            rng_seed=self.rng_seed,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
        )

        def _s_c_predict_wrapper(x: np.ndarray) -> np.ndarray:
            if self.km_ is not None:
                return np.atleast_1d(self.km_.predict(x))
            return np.ones_like(x)

        em_result = em_stage.run(
            initial_estimator=init_est,
            data=data,
            S_c_predict=_s_c_predict_wrapper,
        )
        self._em_stage_result = em_result

        # Store EM results
        if init_est.theta_hat is not None:
            self.theta_path_ = [init_est.theta_hat.copy()] + em_result.theta_path[1:]
        else:
            self.theta_path_ = em_result.theta_path
        self.em_iterations_ = em_result.em_iterations
        self.em_converged_ = em_result.em_converged
        self.uncensored_augmented_ = em_result.final_augmented_data

        # Copy final estimator state to self
        final_est = em_result.final_estimator
        self._current_estimator = final_est
        
        # Validate final estimator has required attributes
        if final_est.theta_hat is None:
            raise RuntimeError("EM stage failed: final estimator has no theta_hat")
        if final_est._grid_points_hal is None:
            raise RuntimeError("EM stage failed: final estimator has no _grid_points_hal")
        if final_est.grid_midpoints is None:
            raise RuntimeError("EM stage failed: final estimator has no grid_midpoints")
        if final_est.delta_j is None:
            raise RuntimeError("EM stage failed: final estimator has no delta_j")
        if final_est.grid_points is None:
            raise RuntimeError("EM stage failed: final estimator has no grid_points")
        if final_est.grid_points_hal_selected is None:
            raise RuntimeError("EM stage failed: final estimator has no grid_points_hal_selected")
            
        self.theta_hat = final_est.theta_hat.copy()
        self._grid_points_hal = final_est._grid_points_hal.copy()
        self.grid_midpoints = final_est.grid_midpoints.copy()
        self.delta_j = final_est.delta_j.copy()
        self.grid_points = final_est.grid_points.copy()
        self.grid_points_hal_selected = final_est.grid_points_hal_selected.copy()
        self.basis_names = final_est.basis_names
        self.fitted_theta_dict = final_est.fitted_theta_dict
        self.is_fitted = True

        # Compute pruned theta for compatibility
        self.theta_full_ = self.theta_hat.copy()
        self.basis_full_ = self._grid_points_hal.copy()
        self.theta_subset_, self.active_knots_ = self._prune_theta(
            self.theta_full_,
            self.basis_full_,
        )

        return self

    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({
            "theta_path": [theta.tolist() for theta in self.theta_path_],
            "has_km": self.km_ is not None,
            "em_iterations": self.em_iterations_,
            "em_converged": self.em_converged_,
        })
        return base

    def get_density(self):
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density()
        return super().get_density()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density_at_points(points)
        return super().get_density_at_points(points)

    def _prune_theta(
        self,
        theta: np.ndarray,
        knots: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols
        if knot_start > theta.size:
            knot_start = theta.size
        truncated = theta[knot_start:]
        if truncated.size == 0:
            pruned = theta[:knot_start].copy()
            return pruned, np.array([], dtype=float)
        mask = np.abs(truncated) > self.tol
        if not np.any(mask):
            pruned = np.zeros(knot_start + truncated.size, dtype=float)
            pruned[:knot_start] = theta[:knot_start]
            pruned[knot_start:] = truncated
            return pruned, knots.copy()
        active_knots = knots[mask].copy()
        pruned = np.zeros(knot_start + mask.sum(), dtype=float)
        pruned[:knot_start] = theta[:knot_start]
        pruned[knot_start:] = truncated[mask]
        return pruned, active_knots

    def _debug_estimator_state(self, label: str, estimator: BaseEstimator) -> None:
        if not self.verbose:
            return
        try:
            theta = estimator.theta_hat
            if theta is None:
                print(f"[DEBUG][{label}] theta is None")
                return
            theta = np.asarray(theta, dtype=float)
            non_finite = np.logical_not(np.isfinite(theta)).sum()
            l1 = float(np.sum(np.abs(theta)))
            knots = getattr(estimator, "_grid_points_hal", None)
            n_knots = len(knots) if knots is not None else 0
            sel = getattr(estimator, "grid_points_hal_selected", None)
            n_sel = len(sel) if sel is not None else 0
            print(
                f"[DEBUG][{label}] theta stats: shape={theta.shape}, "
                f"min={theta.min():.4f}, max={theta.max():.4f}, "
                f"mean={theta.mean():.4f}, l1={l1:.4f}, non_finite={non_finite}, "
                f"knots={n_knots}, selected_knots={n_sel}"
            )
            grid, density = estimator.get_density()
            delta = BaseEstimator._integration_widths(grid)
            integral = float(np.sum(density * delta))
            print(
                f"[DEBUG][{label}] density stats: min={density.min():.4e}, "
                f"max={density.max():.4e}, integral={integral:.6f}"
            )
        except Exception as exc:
            print(f"[DEBUG][{label}] unable to compute diagnostics: {exc}")
