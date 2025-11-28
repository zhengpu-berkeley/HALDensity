"""EM-based density estimation for right-censored data.

This module provides:
1. EMStage: Standalone EM refinement stage that works with any initial estimator
2. EMStageResult: Container for EMStage results
3. EMIPCWEstimator: Combined IPCW initialization + EM refinement estimator
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Any, Callable, Optional, Protocol

from haldensity.estimation.base_estimator import BaseEstimator
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .sampling import e_step_multiple_imputation
from .metrics import incomplete_loglik


class DensityEstimatorProtocol(Protocol):
    """Protocol defining the interface required for an initial estimator."""

    theta_hat: np.ndarray
    _grid_points_hal: np.ndarray
    basis_order: int

    def get_density(self) -> tuple[np.ndarray, np.ndarray]: ...
    def get_density_at_points(self, points: np.ndarray) -> np.ndarray: ...


class EMStageResult:
    """Container for EMStage results."""

    def __init__(
        self,
        final_estimator: WeightedCVXPYEstimator,
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


class EMStage:
    """
    Standalone EM stage for right-censored density estimation.

    This class implements the Expectation-Maximization algorithm with multiple
    imputation for refining an initial density estimate. It can work with any
    estimator that satisfies the DensityEstimatorProtocol interface.

    Parameters
    ----------
    m_imputations : int
        Number of imputations per censored observation in E-step.
    max_em_iter : int
        Maximum number of EM iterations.
    em_tol : float
        Convergence tolerance for incomplete-data log-likelihood.
    norm_constraint : float
        L1 norm constraint for M-step HAL fitting.
    n_grid_points : int
        Number of grid points for density evaluation.
    use_sc_adjustment : bool
        Whether to adjust for censoring survival in E-step sampling.
    e_step_n_grid : int
        Number of grid points for E-step sampling.
    tol : float
        Tolerance for pruning small coefficients.
    m_step_solver : str
        CVXPY solver for M-step.
    m_step_solver_sequence : list[str] | None
        Fallback solver sequence for M-step.
    include_intercept_in_constraint : bool
        Whether to include intercept in norm constraint.
    verbose : bool
        Whether to print progress information.
    rng_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        m_imputations: int = 20,
        max_em_iter: int = 50,
        em_tol: float = 1e-3,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        use_sc_adjustment: bool = False,
        e_step_n_grid: int = 1000,
        tol: float = 1e-4,
        m_step_solver: str = "ECOS",
        m_step_solver_sequence: Optional[list[str]] = None,
        include_intercept_in_constraint: bool = True,
        verbose: bool = False,
        rng_seed: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
    ):
        self.m_imputations = m_imputations
        self.max_em_iter = max_em_iter
        self.em_tol = em_tol
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.use_sc_adjustment = use_sc_adjustment
        self.e_step_n_grid = e_step_n_grid
        self.tol = tol
        self.m_step_solver = m_step_solver
        self.include_intercept_in_constraint = include_intercept_in_constraint
        self.verbose = verbose
        self.rng = np.random.default_rng(rng_seed)
        self.log_dir = log_dir
        self.log_frequency = log_frequency

        # Build solver waterfall
        if m_step_solver_sequence is not None:
            self.m_step_solver_sequence = list(m_step_solver_sequence)
        else:
            self.m_step_solver_sequence = []
            for cand in (m_step_solver, "CLARABEL", "ECOS", "SCS"):
                if cand not in self.m_step_solver_sequence:
                    self.m_step_solver_sequence.append(cand)

    def run(
        self,
        initial_estimator: Any,
        data: pd.DataFrame,
        S_c_predict: Callable[[np.ndarray], np.ndarray],
    ) -> EMStageResult:
        """
        Run EM iterations starting from an initial estimator.

        Parameters
        ----------
        initial_estimator : DensityEstimatorProtocol
            A fitted estimator with theta_hat, _grid_points_hal, basis_order,
            and get_density_at_points() method.
        data : pd.DataFrame
            DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
        S_c_predict : Callable
            Function to predict censoring survival probability S_c(t).

        Returns
        -------
        EMStageResult
            Container with the refined estimator, theta path, and convergence info.
        """
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")

        # Extract required attributes from initial estimator
        theta_full = np.asarray(initial_estimator.theta_hat, dtype=float).copy()
        basis_grid_points = np.asarray(initial_estimator._grid_points_hal, dtype=float).copy()
        basis_order = int(initial_estimator.basis_order)

        theta_path: list[np.ndarray] = [theta_full.copy()]
        current_estimator = initial_estimator
        final_augmented: Optional[pd.DataFrame] = None

        # Compute initial log-likelihood
        prev_ll = incomplete_loglik(current_estimator, data, time_col="T", delta_col="Delta")
        if self.verbose:
            print(f"EMStage: Initial incomplete-data log-likelihood: {prev_ll:.4f}")

        em_converged = False
        em_iterations = 0

        for em_iter in range(self.max_em_iter):
            em_iterations = em_iter + 1

            # E-step: multiple imputation
            t0 = time.time()
            pooled = e_step_multiple_imputation(
                data=data,
                theta_hat=theta_full,
                basis_grid_points=basis_grid_points,
                basis_order=basis_order,
                S_c_predict=S_c_predict,
                m_imputations=self.m_imputations,
                n_grid=self.e_step_n_grid,
                use_sc_adjustment=self.use_sc_adjustment,
                rng=self.rng,
            )
            e_time = time.time() - t0
            final_augmented = pooled

            if self.verbose:
                self._debug_pooled_data(em_iter + 1, pooled)

            # M-step: fit weighted HAL on pooled data
            t1 = time.time()
            mstep_est = self._fit_m_step(
                pooled_df=pooled,
                grid_override=basis_grid_points,
                warm_theta=theta_full,
                basis_order=basis_order,
            )
            m_time = time.time() - t1

            # Update state
            current_estimator = mstep_est
            if mstep_est.theta_hat is None or mstep_est._grid_points_hal is None:
                raise RuntimeError("M-step estimator fitting failed - theta or grid points is None")
            theta_full = mstep_est.theta_hat.copy()
            basis_grid_points = mstep_est._grid_points_hal.copy()
            theta_path.append(theta_full.copy())

            if self.verbose:
                self._debug_estimator_state(f"M-step-{em_iter + 1}", mstep_est)

            # Check convergence
            curr_ll = incomplete_loglik(mstep_est, data, time_col="T", delta_col="Delta")
            ll_diff = np.abs(curr_ll - prev_ll)

            if self.verbose:
                print(
                    f"EMStage Iter {em_iter + 1}: LL={curr_ll:.4f}, Δ={ll_diff:.6f}, "
                    f"E-step={e_time:.3f}s, M-step={m_time:.3f}s"
                )

            if ll_diff < self.em_tol:
                if self.verbose:
                    print(
                        f"\nEMStage: Converged at iteration {em_iter + 1}: "
                        f"LL diff {ll_diff:.6f} < tol {self.em_tol}"
                    )
                em_converged = True
                break

            prev_ll = curr_ll

        return EMStageResult(
            final_estimator=current_estimator,
            theta_path=theta_path,
            em_iterations=em_iterations,
            em_converged=em_converged,
            final_augmented_data=final_augmented,
        )

    def _fit_m_step(
        self,
        pooled_df: pd.DataFrame,
        grid_override: np.ndarray,
        warm_theta: np.ndarray,
        basis_order: int,
    ) -> WeightedCVXPYEstimator:
        """Fit weighted HAL estimator on pooled imputed data."""
        weights = pooled_df["weight"].values.astype(float)
        df_values = pd.DataFrame({"W1": pooled_df["W1"].values})

        return WeightedCVXPYEstimator(
            tol=self.tol,
            norm_constraint=self.norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=basis_order,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            solver=self.m_step_solver,
            use_secondary_solver=True,
            solver_waterfall=self.m_step_solver_sequence,
            include_intercept_in_constraint=self.include_intercept_in_constraint,
        ).fit(
            df_values,
            sample_weights=weights,
            grid_points_override=grid_override,
            warm_start_theta=warm_theta if len(warm_theta) > 0 else None,
        )

    def _debug_estimator_state(self, label: str, estimator: BaseEstimator) -> None:
        """Print debug information about estimator state."""
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

    def _debug_pooled_data(self, em_iter: int, pooled_df: pd.DataFrame) -> None:
        """Print debug information about pooled imputed data."""
        if not self.verbose or pooled_df is None or pooled_df.empty:
            return
        weights = pooled_df["weight"].values.astype(float)
        values = pooled_df["W1"].values.astype(float)
        unc_mask = np.isclose(weights, 1.0)
        unc_min = float(values[unc_mask].min()) if np.any(unc_mask) else float("nan")
        cen_mask = ~unc_mask
        cen_min = float(values[cen_mask].min()) if np.any(cen_mask) else float("nan")
        print(
            f"[DEBUG][E-step-{em_iter}] pooled samples={len(values)}, "
            f"weight_sum={weights.sum():.4f}, weight_range=({weights.min():.4e}, {weights.max():.4e}), "
            f"value_range=({values.min():.4f}, {values.max():.4f}), "
            f"unc_min={unc_min:.4f}, cens_min={cen_min:.4f}"
        )


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

