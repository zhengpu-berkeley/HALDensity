import numpy as np
import pandas as pd
from typing import Optional
from haldensity.utils.basis import create_basis_functions
from haldensity.estimation.base_estimator import BaseEstimator


class FISTAEstimator(BaseEstimator):
    """
    Density estimator using FISTA (accelerated proximal gradient descent).
    
    This class implements the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    for solving the regularized density estimation problem:
    
    min_θ  −ℓ(θ) + λ‖θ[1:]‖₁
    
    where ℓ is the HAL-basis log-likelihood and θ[0] is the intercept (unpenalized).
    """
    
    def __init__(
        self,
        lam: float = 3.0,
        L: float = 2000.0,
        n_iterations: int = 8000,
        tol: float = 1e-6,
        n_grid_points: int = 200,
        basis_order: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1
    ):
        """
        Initialize FISTA estimator.
        
        Args:
            lam: L1 regularization parameter
            L: Lipschitz constant estimate for step size (step = 1/L)
            n_iterations: Maximum number of iterations
            tol: Convergence tolerance
            n_grid_points: Number of grid points for density evaluation
            basis_order: Order of the truncated power basis
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
        """
        # Initialize base class
        super().__init__(
            lam=lam,
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency
        )
        
        # FISTA-specific parameters
        self.L = L
        self.n_grid_points = n_grid_points
        
        # Will be set during fitting
        self.grid_points_hal_selected: Optional[np.ndarray] = None
        
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'FISTAEstimator':
        """
        Fit the FISTA density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            warm_start_coefficients: Optional initial coefficients for warm starting
            validation_data: Optional validation data for tracking performance
            validation_frequency: Frequency for validation logging (default -1 means no validation)
            
        Returns:
            Self for method chaining
        """
        n_samples = len(data)
        
        # 1) HAL basis on observed W1
        grid_points_hal = np.unique(data['W1'].dropna())
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)  # (n, K)
        self.basis_names = basis_names
        
        # 2) midpoint grid for the log‐normaliser
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = np.log(delta_j)
        df_mid = pd.DataFrame({'W1': midpoints})
        b_jk, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)  # (m, K)
        
        # 3) initialise θ⁽⁰⁾ and θ⁽⁻¹⁾ = θ⁽⁰⁾
        K = b_ik.shape[1]  # number of parameters
        
        if warm_start_coefficients is None:
            theta_old = np.zeros(K)
            theta_cur = np.zeros(K)
        elif len(warm_start_coefficients) == K:
            theta_old = np.array(warm_start_coefficients, dtype=float).copy()
            theta_cur = np.array(warm_start_coefficients, dtype=float).copy()
        else:
            if self.do_log:
                self.logger.info(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}")
            theta_old = np.zeros(K)
            theta_cur = np.zeros(K)
        
        step = 1.0 / self.L
        
        if self.do_log:
            self.logger.info(f"Starting FISTA with K={K} parameters")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"FISTA: lam={self.lam}, L={self.L}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        for k in range(1, self.n_iterations + 1):
            # 4a) form momentum point v = θ^(k-1) + ((k-2)/(k+1)) * (θ^(k-1) - θ^(k-2))
            momentum = (k - 2) / (k + 1)
            v = theta_cur + momentum * (theta_cur - theta_old)
            
            # 4b) compute gradient ∇(−ℓ) at v (numpy)
            grad_term1 = -np.sum(b_ik, axis=0)
            log_f_grid = b_jk @ v
            max_log_f = np.max(log_f_grid)
            f_grid_unnorm = np.exp(log_f_grid - max_log_f)
            weights_unnorm = f_grid_unnorm * delta_j
            Z = np.sum(weights_unnorm)
            weights = weights_unnorm / Z
            grad_term2 = n_samples * (weights[:, None] * b_jk).sum(axis=0)
            grad_v = grad_term1 + grad_term2

            # Early stopping if objective function explodes
            obj_current = -np.sum(b_ik @ v) + n_samples * (max_log_f + np.log(np.sum(np.exp(log_f_grid - max_log_f) * delta_j)))
            if self._check_objective_explosion(float(obj_current), k):
                break
            
            # 4c) gradient step + soft-threshold
            theta_next = v - step * grad_v
            # soft-threshold on θ[1:] (exclude intercept)
            u = theta_next[1:]
            shrunk = np.maximum(np.abs(u) - self.lam * step, 0.0)
            theta_next[1:] = np.sign(u) * shrunk
            
            # Intercept correction for exact normalization
            logZ = np.log(np.sum(np.exp(b_jk @ theta_next) * delta_j))
            if not np.isfinite(logZ):
                if self.do_log:
                    self.logger.warning(f"logZ became {logZ} at iteration {k}, stopping optimization")
                break
            theta_next[0] -= logZ  # Subtract from intercept only
            
            # 4d) check convergence on max parameter change
            change = float(np.max(np.abs(theta_next - theta_cur)))
            if change < self.tol:
                theta_old, theta_cur = theta_cur, theta_next
                if self.do_log:
                    self.logger.info(f"Converged at iteration {k}")
                break
            
            # rotate for next iteration
            theta_old, theta_cur = theta_cur, theta_next
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and k % validation_frequency == 0:
                # Update parameters for validation
                self.theta_hat = theta_cur
                self.grid_midpoints = midpoints
                self.delta_j = delta_j
                self._grid_points_hal = grid_points_hal
                
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {k}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
            
            if k % self.log_frequency == 0:
                l1norm = float(np.sum(np.abs(theta_cur[1:])))
                num_selected_knots = int(np.sum(np.abs(theta_cur[1:]) > self.tol))
                loss_val = float(obj_current)
                
                if self.do_log:
                    self.logger.info(f"Iter {k:4d}: loss={loss_val:.4f}, change={change:.2e}, ‖θ[1:]‖₁={l1norm:.3f}, num_selected_knots={num_selected_knots}")
        
        # Store results
        self.theta_hat = theta_cur
        self.grid_midpoints = midpoints
        self.delta_j = delta_j
        
        # Store grid points for density computation
        self._grid_points_hal = grid_points_hal
        
        # select non-zero knots (only for truncated power terms, not polynomial terms)
        if self.basis_order == 0:
            # For order 0: theta = [intercept, step_functions...]
            truncated_power_coeffs = self.theta_hat[1:]
        else:
            # For order k≥1: theta = [intercept, x, x^2, ..., x^k, (x-ξ₁)₊^k, ...]
            truncated_power_coeffs = self.theta_hat[1 + self.basis_order:]
        
        mask = np.abs(truncated_power_coeffs) > self.tol
        self.grid_points_hal_selected = grid_points_hal[mask]
        
        # Final log
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={k}")
        
        # Choose an evaluation grid for public density API
        self.grid_points = np.linspace(0, 1, self.n_grid_points)
        self.is_fitted = True

        # Store the fitted theta as a dictionary for inspection
        assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
        self.fitted_theta_dict = {name: float(value) for name, value in zip(self.basis_names, self.theta_hat.tolist())}

        return self
    
    # Rely on BaseEstimator.get_density
    
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