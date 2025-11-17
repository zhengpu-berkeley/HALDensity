import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional
from haldensity.utils.basis import create_basis_functions
from haldensity.estimation.base_estimator import BaseEstimator
from .legacy_basis import create_legacy_basis


class WeightedCVXPYEstimator(BaseEstimator):
    """
    CVXPY-based HAL density estimator with per-sample weights.
    Objective (weighted):
        minimize  -sum_i w_i * (b_i @ theta) + (sum_i w_i) * log_sum_exp(log delta_j + (b_j @ theta))
        s.t. ||theta[1:]||_1 <= norm_constraint
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
        legacy_mode: bool = False,
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
        self.legacy_mode = legacy_mode
        self.include_intercept_in_constraint = include_intercept_in_constraint

        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None

    def fit(
        self,
        data: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
        grid_points_override: Optional[np.ndarray] = None,
        warm_start_theta: Optional[np.ndarray] = None,
    ) -> "WeightedCVXPYEstimator":
        """
        Fit weighted HAL estimator. Data must contain 'W1'. If sample_weights is None,
        looks for column 'ipcw_weight' aligned with data rows.
        """
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
            grid_points_hal = np.unique(np.concatenate(([0.0], x, [1.0])))
        self._grid_points_hal = grid_points_hal

        if self.legacy_mode:
            basis_array, basis_names = create_legacy_basis(data, grid_points_hal)
            # In legacy mode, add intercept to basis_names
            self.basis_names = ["Intercept"] + basis_names
        else:
            basis_array, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
            self.basis_names = basis_names
        b_ik = basis_array

        # normalization/eval grid
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        data_grid = pd.DataFrame({"W1": grid_midpoints})
        if self.legacy_mode:
            basis_grid_array, _ = create_legacy_basis(data_grid, grid_points_hal)
        else:
            basis_grid_array, _ = create_basis_functions(data_grid, grid_points_hal, order=self.basis_order)
        b_jk = basis_grid_array

        # In legacy mode: basis has no intercept, so K = len(grid_points) + 1
        # theta[0] is intercept, theta[1:] are basis coefficients
        if self.legacy_mode:
            K = b_ik.shape[1] + 1
            theta = cp.Variable(K)
            first_term = -cp.sum(cp.multiply(w, theta[0] + b_ik @ theta[1:]))
            delta_j = grid_eval[1:] - grid_eval[:-1]
            log_delta_j = np.log(delta_j)
            log_terms = log_delta_j + theta[0] + b_jk @ theta[1:]
            log_Z = cp.log_sum_exp(log_terms)
            second_term = w_sum * log_Z
        else:
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
            constraints = [cp.norm1(theta[1:]) <= self.norm_constraint]
        objective = cp.Minimize(loss)
        problem = cp.Problem(objective, constraints)

        if warm_start_theta is not None and len(warm_start_theta) == K:
            theta.value = warm_start_theta
            warm_args = True
        else:
            warm_start_theta = None
            warm_args = False
        try:
            solve_kwargs = {"solver": self.solver}
            if str(self.solver).upper() == "MOSEK" and self.max_threads is not None:
                solve_kwargs["mosek_params"] = {"MSK_IPAR_NUM_THREADS": int(self.max_threads)}
            if warm_args:
                solve_kwargs["warm_start"] = True
            problem.solve(**solve_kwargs)
        except Exception as e:
            if not self.use_secondary_solver:
                raise RuntimeError(f"CVXPY optimization failed: {e}")
            if self.do_log:
                self.logger.info(
                    f"{self.solver} solver failed (basis_order={self.basis_order}, norm_constraint={self.norm_constraint}). "
                    f"Trying solvers: {self.solver_waterfall}"
                )
            success = False
            last_error: Optional[Exception] = None
            for solver in self.solver_waterfall:
                try:
                    solve_kwargs = {"solver": solver}
                    if str(solver).upper() == "MOSEK" and self.max_threads is not None:
                        solve_kwargs["mosek_params"] = {"MSK_IPAR_NUM_THREADS": int(self.max_threads)}
                    if warm_start_theta is not None:
                        theta.value = warm_start_theta
                        solve_kwargs["warm_start"] = True
                    problem.solve(**solve_kwargs)
                    if self.do_log:
                        self.logger.info(f"{solver} succeeded as primary solver")
                    success = True
                    break
                except Exception as e2:
                    last_error = e2
                    if self.do_log:
                        self.logger.info(f"{solver} failed: {e2}")
            if not success:
                raise RuntimeError(
                    f"CVXPY optimization failed with all solvers in waterfall; last error: {last_error}"
                )

        self.lambda_val_lag = problem.constraints[0].dual_value
        if theta.value is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")
        self.optimized_theta_raw = theta.value.copy()

        self.theta_hat = theta.value.copy()
        if self.legacy_mode:
            # In legacy mode: theta[0] = intercept, theta[1:] = basis coefficients
            # Prune theta[1:] based on threshold
            self.theta_hat[1:] = np.where(
                np.abs(self.theta_hat[1:]) > self.tol,
                self.theta_hat[1:],
                0,
            )
            non_zero_knot_indices = np.where(self.theta_hat[1:] != 0)[0]
            if non_zero_knot_indices.size > 0:
                self.grid_points_hal_selected = grid_points_hal[non_zero_knot_indices]
            else:
                self.grid_points_hal_selected = np.array([])
        else:
            self.theta_hat[self.basis_order + 1:] = np.where(
                np.abs(self.theta_hat[self.basis_order + 1:]) > self.tol,
                self.theta_hat[self.basis_order + 1:],
                0,
            )
            knot_coeffs_start_index = self.basis_order + 1
            non_zero_knot_indices = np.where(self.theta_hat[knot_coeffs_start_index:] != 0)[0]
            if non_zero_knot_indices.size > 0:
                self.grid_points_hal_selected = grid_points_hal[non_zero_knot_indices]
            else:
                self.grid_points_hal_selected = np.array([])

        grid_eval = np.sort(np.unique(np.concatenate((grid_eval, self.grid_points_hal_selected))))
        self.grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        self.delta_j = grid_eval[1:] - grid_eval[:-1]
        self.grid_points = np.linspace(0, 1, self.n_grid_points)

        self.is_fitted = True
        assert len(self.basis_names) == len(self.theta_hat)
        self.fitted_theta_dict = {name: float(value) for name, value in zip(self.basis_names, self.theta_hat)}
        return self

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        """Override to handle legacy mode correctly."""
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density.")
        
        if self.legacy_mode:
            # Use legacy density calculation
            from haldensity.censoring.sampling import _calculate_legacy_density
            return _calculate_legacy_density(points, self.theta_hat, self._grid_points_hal)
        else:
            # Use standard BaseEstimator method
            return super().get_density_at_points(points)
    
    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({
            "lambda_val_lag": self.lambda_val_lag,
            "optimized_theta_raw": self.optimized_theta_raw,
        })
        return base


