"""Shared weighted HAL-MLE estimator for censored data.

This module provides the core weighted HAL-MLE fitting logic used by both
right-censored and interval-censored estimators.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
import pandas as pd
import cvxpy as cp

from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.utils.basis import create_basis_functions


logger = logging.getLogger(__name__)


class WeightedHALMLEEstimator(BaseEstimator):
    """CVXPY-based HAL density estimator with per-sample weights.

    This is the shared implementation for weighted HAL-MLE fitting, used by:
    - RightCensoredInitEstimator (with IPCW weights)
    - IntervalCensoredInitEstimator (with uniform/imputation weights)
    - M-step in EM algorithms for both censoring types

    The optimization problem is:
        min_theta  -sum_i w_i * phi(x_i)^T theta + n_eff * log(Z(theta))
        s.t.       ||theta[k:]||_1 <= norm_constraint

    where:
    - w_i are sample weights
    - n_eff = sum(w_i) is the effective sample size
    - Z(theta) is the normalizing constant

    Parameters
    ----------
    tol : float
        Tolerance for pruning small theta coefficients.
    norm_constraint : float
        L1 norm constraint for theta coefficients.
    n_grid_points : int
        Number of grid points for density evaluation.
    basis_order : int
        Order of the truncated power basis (0 = step functions).
    solver : str
        CVXPY solver to use.
    log_dir : str | None
        Directory for logging.
    log_frequency : int
        Frequency of logging (-1 = no logging).
    use_secondary_solver : bool
        Whether to try fallback solvers on failure.
    solver_waterfall : list[str]
        Fallback solver sequence.
    max_threads : int | None
        Maximum threads for MOSEK solver.
    include_intercept_in_constraint : bool
        Whether to include intercept in norm constraint.
    """

    def __init__(
        self,
        tol: float = 1e-4,
        norm_constraint: float = 3.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        solver: str = "ECOS",
        log_dir: Optional[str] = None,
        log_frequency: int = 10,
        use_secondary_solver: bool = False,
        solver_waterfall: list[str] = ["MOSEK", "CLARABEL", "ECOS", "SCS"],
        max_threads: Optional[int] = None,
        include_intercept_in_constraint: bool = False,
    ):
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir or "./local/logs/cvxpy.log",
            log_frequency=log_frequency,
        )
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.solver = solver
        self.use_secondary_solver = use_secondary_solver
        self.solver_waterfall = list(solver_waterfall)
        self.max_threads = max_threads
        self.include_intercept_in_constraint = include_intercept_in_constraint

        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
        self._norm_shift: Optional[float] = None
        self._norm_Z: Optional[float] = None
        self._density_midpoints: Optional[np.ndarray] = None

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
        grid_points_override: Optional[np.ndarray] = None,
        warm_start_theta: Optional[np.ndarray] = None,
        skip_coefficient_pruning: bool = False,
        **kwargs: Any,
    ) -> "WeightedHALMLEEstimator":
        """Fit the weighted HAL density estimator.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with column 'W1' containing the observations.
        sample_weights : np.ndarray | None
            Sample weights for each observation. If None, looks for
            'ipcw_weight' column or uses uniform weights.
        grid_points_override : np.ndarray | None
            Override the knot locations for the basis.
        warm_start_theta : np.ndarray | None
            Initial theta values for warm starting.
        skip_coefficient_pruning : bool
            If True, skip the pruning step that zeros out small coefficients.
            This is used in the M-step of EM to keep the knot structure fixed.
            Default is False.

        Returns
        -------
        self
        """
        if "W1" not in data.columns:
            raise ValueError("data must contain column 'W1'")

        x = np.asarray(data["W1"].values, dtype=float).ravel()
        n_samples = x.shape[0]

        # Get weights
        if sample_weights is None:
            if "ipcw_weight" in data.columns:
                w = np.asarray(data["ipcw_weight"].values, dtype=float).ravel()
            else:
                w = np.ones(n_samples, dtype=float)
        else:
            w = np.asarray(sample_weights, dtype=float).ravel()

        if w.shape[0] != n_samples:
            raise ValueError("sample_weights length must match number of rows in data")

        w_sum = float(np.sum(w))
        if w_sum <= 0:
            raise ValueError("Sum of weights must be positive")

        # Set up grid points for basis
        # IMPORTANT: if grid_points_override is provided (even if empty), respect it.
        # This allows "parametric-only" fits (e.g., {1, x, x^2, ...} with no knot terms),
        # which is required for EM edge cases where no knots are selected at initialization.
        if grid_points_override is not None:
            grid_points_hal = np.sort(np.unique(np.asarray(grid_points_override, dtype=float)))
        else:
            grid_points_hal = np.unique(
                np.concatenate(([0.0], data["W1"].dropna().values.astype(float), [1.0]))
            )
        self._grid_points_hal = grid_points_hal

        # Build basis for data points
        df_x = pd.DataFrame({"W1": x})
        basis_array, basis_names = create_basis_functions(
            df_x,
            grid_points_hal,
            order=self.basis_order,
            include_intercept=True,
        )
        self.basis_names = basis_names
        b_ik = basis_array

        # Build basis for evaluation grid
        grid_eval = np.linspace(0.0, 1.0, self.n_grid_points)
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        df_mid = pd.DataFrame({"W1": grid_midpoints})
        basis_grid_array, _ = create_basis_functions(
            df_mid,
            grid_points_hal,
            order=self.basis_order,
            include_intercept=True,
        )
        b_jk = basis_grid_array

        # Set up optimization problem
        K = b_ik.shape[1]
        theta = cp.Variable(K)

        first_term = -cp.sum(cp.multiply(w, b_ik @ theta))
        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = np.log(delta_j)
        log_terms = log_delta_j + b_jk @ theta
        log_Z = cp.log_sum_exp(log_terms)
        second_term = w_sum * log_Z

        loss = first_term + second_term

        if self.include_intercept_in_constraint:
            constraints = [cp.norm1(theta) <= self.norm_constraint]
        else:
            start_idx = 1
            if start_idx >= K:
                constraints = []
            else:
                constraints = [cp.norm1(theta[start_idx:]) <= self.norm_constraint]

        problem = cp.Problem(cp.Minimize(loss), constraints)

        # Warm start
        warm_args = False
        if warm_start_theta is not None and len(warm_start_theta) == K:
            theta.value = warm_start_theta
            warm_args = True

        def _solve_with_kwargs(solver_name: str, warm: bool) -> None:
            solve_kwargs: dict[str, Any] = {"solver": solver_name}
            if solver_name.upper() == "MOSEK" and self.max_threads is not None:
                solve_kwargs["mosek_params"] = {"MSK_IPAR_NUM_THREADS": int(self.max_threads)}
            if warm:
                solve_kwargs["warm_start"] = True
            problem.solve(**solve_kwargs)

        # Solve with fallback
        try:
            _solve_with_kwargs(self.solver, warm_args)
        except Exception as exc:
            if not self.use_secondary_solver:
                raise RuntimeError(f"CVXPY optimization failed: {exc}")
            success = False
            last_error: Optional[Exception] = None
            for solver in self.solver_waterfall:
                try:
                    if warm_start_theta is not None and len(warm_start_theta) == K:
                        theta.value = warm_start_theta
                        _solve_with_kwargs(solver, True)
                    else:
                        _solve_with_kwargs(solver, False)
                    success = True
                    break
                except Exception as e2:
                    last_error = e2
            if not success:
                raise RuntimeError(
                    f"CVXPY optimization failed with all solvers in waterfall; last error: {last_error}"
                )

        if constraints:
            self.lambda_val_lag = problem.constraints[0].dual_value
        if theta.value is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")

        self.optimized_theta_raw = theta.value.copy()
        self.theta_hat = theta.value.copy()

        # Prune small coefficients (skip if doing parametric EM M-step)
        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols
        if self.theta_hat.size < knot_start:
            knot_start = self.theta_hat.size

        if skip_coefficient_pruning:
            # For parametric EM: keep all coefficients, no selection/thresholding
            # The grid_points_hal_selected is set to match the override grid
            self.grid_points_hal_selected = grid_points_hal.copy()
        else:
            # Standard IPCW fitting: prune small coefficients
            self.theta_hat[knot_start:] = np.where(
                np.abs(self.theta_hat[knot_start:]) > self.tol,
                self.theta_hat[knot_start:],
                0,
            )

            non_zero_knot_indices = np.where(self.theta_hat[knot_start:] != 0)[0]
            if non_zero_knot_indices.size > 0:
                self.grid_points_hal_selected = grid_points_hal[non_zero_knot_indices].copy()
            else:
                self.grid_points_hal_selected = np.array([])

        # Compute normalized density
        output_grid = np.linspace(0.0, 1.0, self.n_grid_points)
        output_grid_mid = (output_grid[:-1] + output_grid[1:]) / 2
        delta_out = output_grid[1:] - output_grid[:-1]

        density_out, _, max_log, norm_const = BaseEstimator.normalized_hal_density(
            output_grid_mid,
            self.theta_hat,
            grid_points_hal,
            self.basis_order,
        )
        self._norm_shift = max_log
        self._norm_Z = norm_const
        self._density_midpoints = density_out

        self.grid_midpoints = output_grid_mid
        self.delta_j = delta_out
        self.grid_points = output_grid

        self.is_fitted = True
        self.fitted_theta_dict = {
            name: float(value) for name, value in zip(self.basis_names, self.theta_hat)
        }
        return self

    def _normalized_density(self, points: np.ndarray) -> np.ndarray:
        """Evaluate normalized density at points."""
        if self._norm_shift is None or self._norm_Z is None:
            raise RuntimeError("Estimator must be fitted before requesting density")
        if self._grid_points_hal is None:
            raise RuntimeError("Estimator must be fitted before requesting density")

        df_pts = pd.DataFrame({"W1": points})
        basis_eval, _ = create_basis_functions(
            df_pts,
            self._grid_points_hal,
            order=self.basis_order,
            include_intercept=True,
        )
        log_eval = basis_eval @ self.theta_hat
        shifted = np.clip(log_eval - self._norm_shift, -700, 700)
        return np.exp(shifted) / self._norm_Z

    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the estimated density on the evaluation grid.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (grid_midpoints, density_values)
        """
        if not self.is_fitted or self._density_midpoints is None:
            raise ValueError("Estimator must be fitted before getting density.")
        if self.grid_midpoints is None:
            raise ValueError("Estimator must be fitted before getting density.")
        return self.grid_midpoints, self._density_midpoints.copy()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate density at specific points.

        Parameters
        ----------
        points : np.ndarray
            Points at which to evaluate the density.

        Returns
        -------
        np.ndarray
            Density values.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density.")
        pts = np.asarray(points, dtype=float).ravel()
        return self._normalized_density(pts)

    def get_results(self) -> dict:
        """Return standardized results plus estimator-specific fields."""
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({
            "lambda_val_lag": self.lambda_val_lag,
            "optimized_theta_raw": self.optimized_theta_raw,
        })
        return base
