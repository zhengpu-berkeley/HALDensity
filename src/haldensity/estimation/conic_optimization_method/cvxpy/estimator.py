import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Tuple
from src.haldensity.utils.basis import create_basis_functions
from src.haldensity.estimation.base_estimator import BaseEstimator


class CVXPYEstimator(BaseEstimator):
    """
    Density estimator using CVXPY for convex optimization.
    
    This class implements the original NPMLE approach using cumulative
    indicator basis functions with CVXPY for solving the convex optimization problem:
    
    min_θ  −ℓ(θ)  subject to  ‖θ[1:]‖₁ ≤ norm_constraint
    
    where ℓ is the HAL-basis log-likelihood.
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
    ):
        """
        Initialize CVXPY estimator.
        
        Args:
            tol: tol for pruning small theta coefficients
            norm_constraint: One-norm constraint for theta[1:]
            n_grid_points: Number of grid points for density evaluation
            solver: CVXPY solver to use (default: "ECOS" for high reliability)
        """
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir or "./local/logs/cvxpy.log",
            log_frequency=log_frequency,
        )
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.solver = solver
        
        # Internal, non-standard attributes for inspection
        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
        self.use_secondary_solver = use_secondary_solver
        self.solver_waterfall = solver_waterfall
        
    def fit(self, data: pd.DataFrame) -> 'CVXPYEstimator':
        """
        Fit the CVXPY density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            
        Returns:
            Self for method chaining
        """
        n_samples = len(data)
        # grid_points_hal = np.unique(np.concatenate(([0], data['W1'].dropna(), [1])))  # Ensure grid includes 0 and 1
        grid_points_hal = np.unique(data['W1'].dropna())
        self._grid_points_hal = grid_points_hal  # Store the original full set of knots
        
        # Build basis for the data points
        basis_array, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
        b_ik = basis_array  # shape: (n_samples, K)
        self.basis_names = basis_names
        
        # Create evaluation grid (equally spaced points) and corresponding basis functions
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        data_grid = pd.DataFrame({'W1': grid_midpoints})
        basis_grid_array, _ = create_basis_functions(data_grid, grid_points_hal, order=self.basis_order)
        b_jk = basis_grid_array  # shape: (len(grid_midpoints), K)
        
        # K: number of basis functions
        if self.basis_order == 0:
            K = 1 + len(grid_points_hal)  # {1, I(x >= ξ₁), ..., I(x >= ξₘ)}
        else:
            K = (self.basis_order + 1) + len(grid_points_hal)  # {1, x, ..., x^k, (x-ξ₁)₊^k, ..., (x-ξₘ)₊^k}
        
        theta = cp.Variable(K)
        
        # Data log-density at the data points and at the evaluation grid
        # Since our basis includes the intercept, we use the full basis @ theta
        first_term = -cp.sum(b_ik @ theta)
        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = np.log(delta_j)
        log_density_grid = b_jk @ theta
        log_terms = log_delta_j + log_density_grid
        log_Z = cp.log_sum_exp(log_terms)
        second_term = n_samples * log_Z
        
        loss = first_term + second_term
        constraints = [cp.norm1(theta[1:]) <= self.norm_constraint]
        objective = cp.Minimize(loss)
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=self.solver)
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
                    problem.solve(solver=solver)
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
        
        # Check if optimization was successful
        if theta.value is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")
         # Store raw optimized value for inspection
        self.optimized_theta_raw = theta.value.copy()
        
        # --- SET STANDARDIZED ATTRIBUTES ---
        # Create and set the official, pruned theta_hat
        self.theta_hat = theta.value.copy()
        # Prune small coefficients (excluding polynomial terms)
        self.theta_hat[self.basis_order + 1:] = np.where(
            np.abs(self.theta_hat[self.basis_order + 1:]) > self.tol, 
            self.theta_hat[self.basis_order + 1:], 
            0
        )
        
        # Determine the selected grid points based on non-zero coefficients in theta_hat
        knot_coeffs_start_index = self.basis_order + 1
        non_zero_knot_indices = np.where(self.theta_hat[knot_coeffs_start_index:] != 0)[0]
        
        if non_zero_knot_indices.size > 0:
            self.grid_points_hal_selected = grid_points_hal[non_zero_knot_indices]
        else:
            self.grid_points_hal_selected = np.array([])
        # --- End Standardization ---
            
        grid_eval = np.sort(np.unique(np.concatenate((grid_eval, self.grid_points_hal_selected))))
        self.grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        self.delta_j = grid_eval[1:] - grid_eval[:-1]
        # Public evaluation grid for density queries
        self.grid_points = np.linspace(0, 1, self.n_grid_points)
        
        self.is_fitted = True

        # Store the fitted theta as a dictionary for inspection
        assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
        self.fitted_theta_dict = {name: float(value) for name, value in zip(self.basis_names, self.theta_hat)}

        return self
    
    def get_results(self) -> dict:
        """Return standardized common results plus CVX-specific fields."""
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        base = self._get_common_results()
        base.update({
            "lambda_val_lag": self.lambda_val_lag,
            "optimized_theta_raw": self.optimized_theta_raw,
        })
        return base
