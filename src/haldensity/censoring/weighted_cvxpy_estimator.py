from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional
from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.utils.basis import create_basis_functions


class WeightedCVXPYEstimator(BaseEstimator):
    """CVXPY-based HAL density estimator with per-sample weights (truncated basis)."""

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
        self.solver_waterfall = solver_waterfall
        self.max_threads = max_threads
        self.include_intercept_in_constraint = include_intercept_in_constraint

        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
        self._norm_shift: Optional[float] = None
        self._norm_Z: Optional[float] = None
        self._density_midpoints: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
        grid_points_override: Optional[np.ndarray] = None,
        warm_start_theta: Optional[np.ndarray] = None,
    ) -> "WeightedCVXPYEstimator":
        if "W1" not in data.columns:
            raise ValueError("data must contain column 'W1'")
        x = np.asarray(data["W1"].values, dtype=float).ravel()
        n_samples = x.shape[0]

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

        if grid_points_override is not None and len(grid_points_override) > 0:
            grid_points_hal = np.sort(np.unique(np.asarray(grid_points_override, dtype=float)))
        else:
            grid_points_hal = np.unique(
                np.concatenate(([0.0], data["W1"].dropna().values.astype(float), [1.0]))
            )
        self._grid_points_hal = grid_points_hal

        df_x = pd.DataFrame({"W1": x})
        basis_array, basis_names = create_basis_functions(
            df_x,
            grid_points_hal,
            order=self.basis_order,
            include_intercept=True,
        )
        self.basis_names = basis_names
        b_ik = basis_array

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
            poly_cols = self.basis_order if self.basis_order > 0 else 0
            start_idx = 1 + poly_cols
            if start_idx >= K:
                constraints = []
            else:
                constraints = [cp.norm1(theta[start_idx:]) <= self.norm_constraint]
        problem = cp.Problem(cp.Minimize(loss), constraints)

        warm_args = False
        if warm_start_theta is not None and len(warm_start_theta) == K:
            theta.value = warm_start_theta
            warm_args = True

        def _solve_with_kwargs(solver_name: str, warm: bool) -> None:
            solve_kwargs = {"solver": solver_name}
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

        self.lambda_val_lag = problem.constraints[0].dual_value
        if theta.value is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")
        self.optimized_theta_raw = theta.value.copy()

        self.theta_hat = theta.value.copy()
        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols
        if self.theta_hat.size < knot_start:
            knot_start = self.theta_hat.size
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

        grid_eval_dense = np.linspace(0.0, 1.0, self.n_grid_points)
        grid_mid = (grid_eval_dense[:-1] + grid_eval_dense[1:]) / 2
        density_mid, delta_mid, max_log, norm_const = BaseEstimator.normalized_hal_density(
            grid_mid,
            self.theta_hat,
            grid_points_hal,
            self.basis_order,
        )
        self._norm_shift = max_log
        self._norm_Z = norm_const
        self._density_midpoints = density_mid
        self.grid_midpoints = grid_mid
        self.delta_j = delta_mid
        self.grid_points = np.linspace(0.0, 1.0, self.n_grid_points)

        self.is_fitted = True
        self.fitted_theta_dict = {name: float(value) for name, value in zip(self.basis_names, self.theta_hat)}
        return self

    def _normalized_density(self, points: np.ndarray) -> np.ndarray:
        if self._norm_shift is None or self._norm_Z is None:
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
        if not self.is_fitted or self._density_midpoints is None:
            raise ValueError("Estimator must be fitted before getting density.")
        return self.grid_midpoints, self._density_midpoints.copy()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density.")
        pts = np.asarray(points, dtype=float).ravel()
        return self._normalized_density(pts)

    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({
            "lambda_val_lag": self.lambda_val_lag,
            "optimized_theta_raw": self.optimized_theta_raw,
        })
        return base
