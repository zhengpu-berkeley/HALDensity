"""Midpoint-imputed HAL-MLE initializer for interval-censored data."""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import cvxpy as cp

from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.utils.basis import create_basis_functions


class IntervalCensoredMidpointEstimator(BaseEstimator):
    """HAL density estimator using midpoint imputation for interval-censored data.

    Input data format
    -----------------
    Expects a DataFrame with columns:
    - `L`: left endpoint
    - `R`: right endpoint

    The initializer computes midpoints W1 = (L+R)/2 and fits the same convex HAL-MLE
    objective used elsewhere in the project. This is used as Stage-1 in the
    interval-censor pipeline and for Stage-1 cross-validation.
    """

    def __init__(
        self,
        tol: float = 1e-4,
        norm_constraint: float = 3.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        solver: str = "ECOS",
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        use_secondary_solver: bool = False,
        solver_waterfall: list[str] = ["MOSEK", "CLARABEL", "ECOS", "SCS"],
        max_threads: Optional[int] = None,
        include_intercept_in_constraint: bool = False,
    ):
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = float(norm_constraint)
        self.n_grid_points = int(n_grid_points)
        self.solver = str(solver)
        self.use_secondary_solver = bool(use_secondary_solver)
        self.solver_waterfall = list(solver_waterfall)
        self.max_threads = max_threads
        self.include_intercept_in_constraint = bool(include_intercept_in_constraint)

        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
        self._norm_shift: Optional[float] = None
        self._norm_Z: Optional[float] = None
        self._density_midpoints: Optional[np.ndarray] = None

    @staticmethod
    def _midpoint_impute(df: pd.DataFrame, L_col: str = "L", R_col: str = "R") -> np.ndarray:
        if L_col not in df.columns or R_col not in df.columns:
            raise ValueError(f"data must contain columns {L_col!r} and {R_col!r}")
        L = np.asarray(df[L_col].values, dtype=float).ravel()
        R = np.asarray(df[R_col].values, dtype=float).ravel()
        return 0.5 * (L + R)

    def fit(
        self,
        data: pd.DataFrame,
        *,
        L_col: str = "L",
        R_col: str = "R",
        grid_points_override: Optional[np.ndarray] = None,
        warm_start_theta: Optional[np.ndarray] = None,
        skip_coefficient_pruning: bool = False,
    ) -> "IntervalCensoredMidpointEstimator":
        """Fit the midpoint-imputed HAL-MLE.

        Parameters
        ----------
        data:
            DataFrame with columns (L, R).
        grid_points_override:
            Optional fixed knot locations for the HAL basis (used in EM M-step).
        warm_start_theta:
            Optional warm start for CVXPY solver.
        skip_coefficient_pruning:
            If True, skip pruning to keep knot structure fixed (used in EM M-step).
        """
        x = self._midpoint_impute(data, L_col=L_col, R_col=R_col)
        n_samples = int(x.shape[0])
        if n_samples == 0:
            raise ValueError("data must be non-empty")

        # Knots for basis
        if grid_points_override is not None and len(grid_points_override) > 0:
            grid_points_hal = np.sort(np.unique(np.asarray(grid_points_override, dtype=float)))
        else:
            grid_points_hal = np.unique(np.concatenate(([0.0], x.astype(float), [1.0])))
        self._grid_points_hal = grid_points_hal

        # Basis for observed points
        df_x = pd.DataFrame({"W1": x})
        b_ik, basis_names = create_basis_functions(
            df_x, grid_points_hal, order=self.basis_order, include_intercept=True
        )
        self.basis_names = basis_names

        # Basis for evaluation grid midpoints
        grid_eval = np.linspace(0.0, 1.0, self.n_grid_points)
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        df_mid = pd.DataFrame({"W1": grid_midpoints})
        b_jk, _ = create_basis_functions(
            df_mid, grid_points_hal, order=self.basis_order, include_intercept=True
        )

        K = int(b_ik.shape[1])
        theta = cp.Variable(K)

        # Unweighted likelihood for midpoint pseudo-observations
        first_term = -cp.sum(b_ik @ theta)

        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = np.log(delta_j)
        log_terms = log_delta_j + b_jk @ theta
        log_Z = cp.log_sum_exp(log_terms)
        second_term = float(n_samples) * log_Z

        loss = first_term + second_term

        if self.include_intercept_in_constraint:
            constraints = [cp.norm1(theta) <= self.norm_constraint]
        else:
            start_idx = 1
            constraints = [cp.norm1(theta[start_idx:]) <= self.norm_constraint] if start_idx < K else []

        problem = cp.Problem(cp.Minimize(loss), constraints)

        warm_args = False
        if warm_start_theta is not None and len(warm_start_theta) == K:
            theta.value = np.asarray(warm_start_theta, dtype=float).ravel()
            warm_args = True

        def _solve_with_kwargs(solver_name: str, warm: bool) -> None:
            solve_kwargs: dict[str, object] = {"solver": solver_name}
            if solver_name.upper() == "MOSEK" and self.max_threads is not None:
                solve_kwargs["mosek_params"] = {"MSK_IPAR_NUM_THREADS": int(self.max_threads)}
            if warm:
                solve_kwargs["warm_start"] = True
            problem.solve(**solve_kwargs)

        try:
            _solve_with_kwargs(self.solver, warm_args)
        except Exception as exc:
            if not self.use_secondary_solver:
                raise RuntimeError(f"CVXPY optimization failed: {exc}")
            last_error: Optional[Exception] = None
            success = False
            for solver in self.solver_waterfall:
                try:
                    if warm_start_theta is not None and len(warm_start_theta) == K:
                        theta.value = np.asarray(warm_start_theta, dtype=float).ravel()
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
            self.lambda_val_lag = float(problem.constraints[0].dual_value)
        if theta.value is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")

        self.optimized_theta_raw = np.asarray(theta.value, dtype=float).copy()
        self.theta_hat = self.optimized_theta_raw.copy()

        # Prune to select knots unless parametric M-step.
        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols
        if self.theta_hat.size < knot_start:
            knot_start = self.theta_hat.size

        if skip_coefficient_pruning:
            self.grid_points_hal_selected = grid_points_hal.copy()
        else:
            self.theta_hat[knot_start:] = np.where(
                np.abs(self.theta_hat[knot_start:]) > self.tol,
                self.theta_hat[knot_start:],
                0.0,
            )
            non_zero = np.where(self.theta_hat[knot_start:] != 0)[0]
            self.grid_points_hal_selected = grid_points_hal[non_zero].copy() if non_zero.size > 0 else np.array([])

        # Normalized density on output grid
        output_grid = np.linspace(0.0, 1.0, self.n_grid_points)
        output_mid = (output_grid[:-1] + output_grid[1:]) / 2
        delta_out = output_grid[1:] - output_grid[:-1]
        density_out, _, max_log, norm_const = BaseEstimator.normalized_hal_density(
            output_mid, self.theta_hat, grid_points_hal, self.basis_order
        )
        self._norm_shift = max_log
        self._norm_Z = norm_const
        self._density_midpoints = density_out

        self.grid_midpoints = output_mid
        self.delta_j = delta_out
        self.grid_points = output_grid
        self.is_fitted = True

        self.fitted_theta_dict = {
            name: float(value) for name, value in zip(self.basis_names, self.theta_hat)
        }
        return self

    def _normalized_density(self, points: np.ndarray) -> np.ndarray:
        if self._norm_shift is None or self._norm_Z is None:
            raise RuntimeError("Estimator must be fitted before requesting density")
        pts = np.asarray(points, dtype=float).ravel()
        df_pts = pd.DataFrame({"W1": pts})
        basis_eval, _ = create_basis_functions(
            df_pts, self._grid_points_hal, order=self.basis_order, include_intercept=True
        )
        log_eval = basis_eval @ self.theta_hat
        shifted = np.clip(log_eval - self._norm_shift, -700, 700)
        return np.exp(shifted) / self._norm_Z

    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted or self._density_midpoints is None:
            raise ValueError("Estimator must be fitted before getting density.")
        return self.grid_midpoints, self._density_midpoints.copy()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density.")
        return self._normalized_density(points)

    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({"lambda_val_lag": self.lambda_val_lag, "optimized_theta_raw": self.optimized_theta_raw})
        return base


