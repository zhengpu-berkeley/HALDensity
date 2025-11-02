import numpy as np
import pandas as pd
from typing import Optional, Tuple
from src.haldensity.utils.basis import create_basis_functions
from src.haldensity.estimation.base_estimator import BaseEstimator


class ProjectedGDEstimator(BaseEstimator):
    """
    Density estimator using Projected Gradient Descent (with L1 ball projection).
    
    This class implements projected gradient descent for solving the constrained
    density estimation problem:
    
    min_θ  −ℓ(θ)  subject to  ‖θ[1:]‖₁ ≤ norm_constraint
    
    where ℓ is the HAL-basis log-likelihood and θ[0] is the intercept (unconstrained).
    """
    
    def __init__(
        self,
        lam: float = 0.25,
        norm_constraint: Optional[float] = None,
        learning_rate: float = 1e-2,
        n_iterations: int = 30000,
        tol: float = 1e-4,
        n_grid_points: int = 200,
        basis_order: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1
    ):
        """
        Initialize Projected GD estimator.
        
        Args:
            lam: L1 penalty coefficient for proximal update (if norm_constraint is None)
            norm_constraint: L1 ball radius for projection (if provided, uses constrained form)
            learning_rate: Gradient descent step size
            n_iterations: Maximum number of iterations
            tol: tol for pruning small coefficients
            n_grid_points: Number of grid points for density evaluation
            basis_order: Order of the truncated power basis
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
        """
        # Initialize base class (lam used for proximal penalty when no constraint)
        super().__init__(
            lam=lam,
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency
        )
        
        # ProjectedGD-specific parameters
        self.norm_constraint = norm_constraint
        self.learning_rate = learning_rate
        self.tol = tol
        self.n_grid_points = n_grid_points
        
        # Will be set during fitting
        self.theta_final: Optional[np.ndarray] = None
        self.theta_pruned: Optional[np.ndarray] = None
        self.grid_points_hal_selected: Optional[np.ndarray] = None

    
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'ProjectedGDEstimator':
        """
        Fit the Projected GD density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            warm_start_coefficients: Optional initial coefficients for warm starting
            validation_data: Optional validation data for tracking performance
            validation_frequency: Frequency for validation logging (default -1 means no validation)
            
        Returns:
            Self for method chaining
        """
        n_samples = len(data)
        
        # Use unique observed data points as grid for basis functions
        grid_points_hal = np.unique(data['W1'].dropna())
        
        # Build basis for the data points
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
        self.basis_names = basis_names

        # Create evaluation grid and corresponding basis functions
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        data_grid = pd.DataFrame({'W1': grid_midpoints})
        b_jk, _ = create_basis_functions(data_grid, grid_points_hal, order=self.basis_order)
        
        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = np.log(delta_j)
        
        # Initialise θ
        K = b_ik.shape[1]
        if warm_start_coefficients is None:
            theta = np.zeros(K, dtype=float)
        elif len(warm_start_coefficients) == K:
            theta = np.asarray(warm_start_coefficients, dtype=float).copy()
        else:
            if self.do_log:
                self.logger.info(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}")
            theta = np.zeros(K, dtype=float)

        if self.do_log:
            self.logger.info(f"Starting ProjectedGD with K={K} parameters")
            self.logger.info(f"ProjectedGD: norm_constraint={self.norm_constraint}, lr={self.learning_rate}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")

        sum_phi_data = np.sum(b_ik, axis=0)

        for it in range(self.n_iterations):
            # Gradient of smooth part
            grad_term1 = -sum_phi_data
            log_density_grid = b_jk @ theta
            max_log = np.max(log_delta_j + log_density_grid)
            weights_unnorm = np.exp((log_delta_j + log_density_grid) - max_log)
            Z = np.sum(weights_unnorm)
            weights = weights_unnorm / Z
            grad_term2 = n_samples * (b_jk.T @ weights)
            gradient = grad_term1 + grad_term2

            # Objective value for monitoring
            loss_val = -np.sum(b_ik @ theta) + n_samples * (max_log + np.log(Z))
            if self._check_objective_explosion(loss_val, it):
                break

            # Gradient step
            theta -= self.learning_rate * gradient

            # Intercept correction for exact normalization
            logZ = np.log(np.sum(np.exp(b_jk @ theta) * delta_j))
            if not np.isfinite(logZ):
                if self.do_log:
                    self.logger.warning(f"logZ became {logZ} at iteration {it}, stopping optimization")
                break
            theta[0] -= logZ

            # Sparsity update: projection (constraint) or proximal soft-thresholding (penalty)
            if self.norm_constraint is not None:
                def _project_onto_l1_ball(v: np.ndarray, z: float) -> np.ndarray:
                    if np.abs(v).sum() <= z:
                        return v
                    u = np.sort(np.abs(v))[::-1]
                    sv = np.cumsum(u)
                    rho = np.where(u > (sv - z) / np.arange(1, len(u) + 1))[0]
                    if len(rho) == 0:
                        tau = 0.0
                    else:
                        rho_idx = rho[-1]
                        tau = (sv[rho_idx] - z) / (rho_idx + 1)
                    return np.sign(v) * np.maximum(np.abs(v) - tau, 0)
                theta[1:] = _project_onto_l1_ball(theta[1:], z=self.norm_constraint)
            else:
                # proximal soft-thresholding with step size*lam
                thresh = self.learning_rate * self.lam
                v = theta[1:]
                theta[1:] = np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)

            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and it % validation_frequency == 0:
                self.theta_hat = theta.copy()
                self.grid_midpoints = grid_midpoints
                self.delta_j = delta_j
                self._grid_points_hal = grid_points_hal
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {it}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")

            if it % self.log_frequency == 0:
                l1_norm = float(np.sum(np.abs(theta[1:])))
                num_selected_knots = int(np.sum(np.abs(theta[1:]) > self.tol))
                if self.do_log:
                    self.logger.info(f"Iter {it}: loss={loss_val:.4f}, ‖θ[1:]‖₁={l1_norm:.3f}, num_selected_knots={num_selected_knots}")
        
        # After optimization, we can prune small coefficients
        self.theta_final = theta.copy()
        self.theta_pruned = np.copy(self.theta_final)
        self.theta_hat = self.theta_final
        self.theta_pruned[1:] = np.where(
            np.abs(self.theta_final[1:]) > self.tol, 
            self.theta_final[1:], 
            0
        )
        
        # Identify nonzero indices and map them correctly to grid points
        non_zero_indices = np.nonzero(self.theta_pruned)[0]
        
        # For basis functions, we need to separate:
        # - Index 0: intercept (always included)
        # - Indices 1 to basis_order: polynomial terms (don't correspond to grid points)
        # - Indices (basis_order+1) onwards: truncated power basis (correspond to grid_points_hal)
        
        if self.basis_order == 0:
            # All non-zero indices (except intercept) correspond to grid points
            truncated_power_indices = non_zero_indices[non_zero_indices > 0]
            grid_point_indices = truncated_power_indices - 1  # Map to grid_points_hal indices
        else:
            # Only indices > basis_order correspond to grid points
            truncated_power_indices = non_zero_indices[non_zero_indices > self.basis_order]
            grid_point_indices = truncated_power_indices - (self.basis_order + 1)  # Map to grid_points_hal indices
        
        # Select grid points corresponding to non-zero truncated power coefficients
        self.grid_points_hal_selected = (
            grid_points_hal[grid_point_indices] 
            if len(grid_point_indices) > 0 
            else np.array([])
        )
        
        # Incorporate the grid points corresponding to nonzero coefficients into the evaluation grid
        grid_eval = np.sort(np.unique(np.concatenate((grid_eval, self.grid_points_hal_selected))))
        self.grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        self.delta_j = grid_eval[1:] - grid_eval[:-1]
        
        # Store grid points for density computation
        self._grid_points_hal = grid_points_hal
        self._non_zero_indices = non_zero_indices
        
        # Choose public evaluation grid for API
        self.grid_points = np.linspace(0, 1, self.n_grid_points)

        # Final log
        final_selected_knots = len(self.grid_points_hal_selected) if self.grid_points_hal_selected is not None else 0
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={it}")
        
        self.is_fitted = True

        # Store the fitted theta as a dictionary for inspection
        assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
        self.fitted_theta_dict = {name: float(value) for name, value in zip(self.basis_names, self.theta_hat.tolist())}

        return self
    
    def get_results(self) -> dict:
        """
        Get comprehensive results from the fitting process.
        
        Returns:
            dictionary containing all relevant results
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        
        return self._get_common_results()
