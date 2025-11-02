import numpy as np
import pandas as pd
from scipy.special import logsumexp
from typing import Optional
from src.haldensity.utils.basis import create_basis_functions
from src.haldensity.estimation.base_estimator import BaseEstimator


class ProximalAdaGradEstimator(BaseEstimator):
    """
    HAL density estimator via proximal gradient + AdaGrad-style
    per-coordinate scaling (uses the diagonal Hessian n·Var_θ(φ)).
    
    This class implements the proximal AdaGrad algorithm for solving 
    the regularized density estimation problem:
    
    min_θ  −ℓ(θ) + λ‖θ[1:]‖₁
    
    where ℓ is the HAL-basis log-likelihood and θ[0] is the intercept (unpenalized).
    
    Key advantages over LBFGS:
    - Memory efficient: O(K) instead of O(K × memory_size)
    - Per-coordinate adaptive learning rates based on variance
    - No matrix inversions or line searches needed
    """
    
    def __init__(
        self,
        lam: float = 3.0,
        n_iterations: int = 2500,
        tol: float = 1e-6,
        alpha: float = 1e-3,      # Global learning rate
        eps: float = 1e-8,       # Stabilizer for denominator
        rho: float = 1.0,        # EMA smoothing (1.0 = no smoothing, 0.9 for noisy problems)
        min_hessian: float = 1e-4,  # Minimum diagonal Hessian value for early iterations
        n_grid_points: int = 200,
        basis_order: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        safeguard_iterations: int = 5,  # Number of initial iterations with minimum Hessian enforcement
        min_intercept_hessian: float = 1.0,  # Minimum Hessian value for intercept term
        relative_objective_tolerance: float = 1e-5  # Tolerance for relative objective change
    ):
        """
        Initialize Proximal AdaGrad estimator.
        
        Args:
            lam: L1 regularization parameter
            n_iterations: Maximum number of iterations
            tol: Convergence tolerance
            alpha: Global learning rate (start with 1.0, halve if oscillations)
            eps: Stabilizer for denominator (prevents division by zero)
            rho: EMA smoothing factor for diagonal Hessian (1.0 = no smoothing)
            min_hessian: Minimum diagonal Hessian value for early iterations
            n_grid_points: Number of grid points for density evaluation
            basis_order: Order of truncated power basis functions
            log_dir: Directory for logging
            log_frequency: Frequency of logging
            safeguard_iterations: Number of initial iterations with minimum Hessian enforcement
            min_intercept_hessian: Minimum Hessian value for intercept term
            relative_objective_tolerance: Tolerance for relative objective change
        """
        super().__init__(
            lam=lam,
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency
        )
        
        self.alpha = alpha
        self.eps = eps
        self.rho = rho
        self.min_hessian = min_hessian
        self.n_grid_points = n_grid_points
        self.safeguard_iterations = safeguard_iterations
        self.min_intercept_hessian = min_intercept_hessian
        self.relative_objective_tolerance = relative_objective_tolerance
        
        # For tracking diagonal Hessian evolution
        self._H_diag_prev: Optional[np.ndarray] = None
    
    @staticmethod
    def _soft_thresh(z: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Vectorized soft-thresholding operator.
        
        Args:
            z: Input array
            tau: Threshold array (same shape as z, τ ≥ 0)
            
        Returns:
            Soft-thresholded array: sign(z) * max(|z| - τ, 0)
        """
        return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)
    
    def _hessian_diag(self, theta: np.ndarray, phi_grid_full: np.ndarray,
                      delta_j: np.ndarray, n_samples: int, iteration: int) -> np.ndarray:
        """
        Compute diagonal of Hessian: H_diag = n * Var_θ(φ)
        
        Args:
            theta: Current parameter vector
            phi_grid_full: Basis matrix at grid midpoints (m, K)
            delta_j: Grid widths for integration (m,)
            n_samples: Number of data samples
            iteration: Current iteration number (for early iteration safeguards)
            
        Returns:
            Diagonal Hessian array (K,)
        """
        # Compute density weights on midpoint grid
        log_f = phi_grid_full @ theta
        max_l = np.max(log_f)
        f = np.exp(log_f - max_l)               # (m,)
        w = f * delta_j                         # Unnormalized weights
        w /= w.sum()                            # Normalized weights (m,)
        
        # Compute E_θ[φ] and Var_θ(φ)
        E_phi = np.sum(w[:, None] * phi_grid_full, axis=0)          # (K,)
        centered = phi_grid_full - E_phi[None, :]                   # (m, K)
        var_phi = np.sum(w[:, None] * centered**2, axis=0)          # (K,)
        
        # Scale by sample size
        H_diag = n_samples * var_phi
        
        # Apply EMA smoothing if requested
        if self.rho < 1.0 and self._H_diag_prev is not None:
            H_diag = self.rho * self._H_diag_prev + (1 - self.rho) * H_diag
        
        # Safeguard for early iterations
        if iteration < self.safeguard_iterations:
            H_diag = np.maximum(H_diag, self.min_hessian)
        
        # Special handling for intercept: start with reasonable scale
        H_diag[0] = max(H_diag[0], self.min_intercept_hessian)
        
        # Store for next iteration if using EMA
        if self.rho < 1.0:
            self._H_diag_prev = H_diag.copy()
        
        return H_diag
    
    def _objective_function(self, theta: np.ndarray, phi_data_full: np.ndarray, 
                           phi_grid_full: np.ndarray, delta_j: np.ndarray) -> float:
        """
        Compute the penalized objective function F(θ).
        
        Args:
            theta: parameter vector [θ₀, θ₁, ..., θₖ]
            phi_data_full: full basis matrix at data points (N, K) with intercept
            phi_grid_full: full basis matrix at grid midpoints (m, K) with intercept
            delta_j: grid widths for integration (m,)
            
        Returns:
            Objective function value
        """
        N = phi_data_full.shape[0]
        
        # First term: -∑ᵢ log f(xᵢ) = -∑ᵢ (φᵢᵀθ)
        log_f_data = phi_data_full @ theta  # shape (N,)
        term1 = -np.sum(log_f_data.astype(np.float64))
        
        # Second term: N * log(∫ f(x) dx) using Riemann sum
        log_f_grid = phi_grid_full @ theta  # shape (m,)
        # Use logsumexp for numerical stability
        max_log_f = np.max(log_f_grid)
        log_integral = max_log_f + np.log(np.sum(np.exp(log_f_grid - max_log_f) * delta_j))
        term2 = N * log_integral
        
        # Penalty term (only on θ[1:], intercept is unpenalized)
        penalty = self.lam * np.sum(np.abs(theta[1:]))
        
        return term1 + term2 + penalty
    
    def _compute_gradient(self, theta: np.ndarray, phi_data_full: np.ndarray, 
                         phi_grid_full: np.ndarray, delta_j: np.ndarray, 
                         n_samples: int) -> np.ndarray:
        """
        Compute gradient of the objective function.
        
        Args:
            theta: current parameter vector
            phi_data_full: basis matrix at data points with intercept (N, K)
            phi_grid_full: basis matrix at grid points with intercept (m, K)
            delta_j: grid widths for integration
            n_samples: number of data samples
            
        Returns:
            Gradient vector
        """
        # First term: -∑ᵢ φ(xᵢ) 
        grad_term1 = -np.sum(phi_data_full, axis=0)
        
        # Second term: N * E[φ(x)] where expectation is under current density
        # Compute weights for current density estimate
        log_f_grid = phi_grid_full @ theta
        max_log_f = np.max(log_f_grid)
        f_grid_unnorm = np.exp(log_f_grid - max_log_f)
        weights_unnorm = f_grid_unnorm * delta_j
        Z = np.sum(weights_unnorm)
        weights = weights_unnorm / Z
        
        # Weighted expectation
        grad_term2 = n_samples * np.sum(phi_grid_full * weights[:, None], axis=0)
        
        return grad_term1 + grad_term2
    
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'ProximalAdaGradEstimator':
        """
        Fit the Proximal AdaGrad density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            warm_start_coefficients: Optional initial coefficients for warm starting
            validation_data: Optional validation data for tracking performance
            validation_frequency: Frequency for validation logging
            
        Returns:
            Self for method chaining
        """
        n_samples = len(data)
        
        # 1) HAL basis on observed W1 (same as other methods)
        grid_points_hal = np.unique(data['W1'].dropna())
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
        self.basis_names = basis_names

        # 2) Midpoint grid for the log-normalizer
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        delta_j = grid_eval[1:] - grid_eval[:-1]
        df_mid = pd.DataFrame({'W1': midpoints})
        b_jk, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)
        
        # Basis arrays already include intercept
        phi_data_full = b_ik  # (N, K)
        phi_grid_full = b_jk  # (m, K)
        
        # 3) Initialize θ = [θ₀, θ₁, ..., θₖ₋₁]
        K = phi_data_full.shape[1]        
        if warm_start_coefficients is None:
            theta = np.zeros(K)
        elif len(warm_start_coefficients) == K:
            theta = warm_start_coefficients.copy()
        else:
            if self.do_log:
                self.logger.info(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}")
            theta = np.zeros(K)
                
        # Initialize for EMA smoothing
        self._H_diag_prev = None
        
        # Initialization log similar to LBFGS
        if self.do_log:
            self.logger.info(f"ProximalAdaGrad: lam={self.lam}, alpha={self.alpha}, eps={self.eps}, rho={self.rho}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        # Track objective for relative convergence
        obj_prev = None
        
        # Main proximal AdaGrad loop
        for iter_k in range(self.n_iterations):
            # a) Compute gradient
            g = self._compute_gradient(theta, phi_data_full, phi_grid_full, delta_j, n_samples)
            
            # b) Compute diagonal Hessian with safeguards
            H_diag = self._hessian_diag(theta, phi_grid_full, delta_j, n_samples, iter_k)
            h = np.sqrt(H_diag) + self.eps  # Denominator vector
            
            # c) Proximal AdaGrad step
            # Intercept: simple gradient step (no penalty)
            theta0_new = theta[0] - self.alpha * g[0] / h[0]
            
            # Other coefficients: proximal step with soft thresholding
            z = theta[1:] - self.alpha * g[1:] / h[1:]  # Raw gradient step
            tau = self.alpha * self.lam / h[1:]         # Threshold vector
            theta_rest_new = self._soft_thresh(z, tau)  # Soft threshold
            
            # Combine
            theta_new = np.concatenate(([theta0_new], theta_rest_new))
            
            # Intercept correction for exact normalization
            # logZ = np.log(np.sum(np.exp(phi_grid_full @ theta_new) * delta_j))
            logZ = logsumexp(phi_grid_full @ theta_new, b=delta_j, axis=0)
            if self.do_log and iter_k % self.log_frequency == 0:
                self.logger.info(f"Iteration {iter_k}: logZ={logZ:.6f}")
            if not np.isfinite(logZ):
                if self.do_log:
                    self.logger.warning(f"logZ became {logZ} at iteration {iter_k}, stopping optimization")
                break
            theta_new[0] -= logZ  # Subtract from intercept only
            
            # d) Convergence check and logging
            change = np.max(np.abs(theta_new - theta))
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and iter_k % validation_frequency == 0:
                # Temporarily update parameters for validation
                old_fitted = self.is_fitted
                old_theta = getattr(self, 'theta_hat', None)
                old_midpoints = getattr(self, 'grid_midpoints', None)
                old_delta_j = getattr(self, 'delta_j', None)
                old_grid_points_hal = getattr(self, '_grid_points_hal', None)
                
                self.theta_hat = theta_new
                self.grid_midpoints = midpoints
                self.delta_j = delta_j
                self._grid_points_hal = grid_points_hal
                self.is_fitted = True  # Temporarily set to fitted for validation
                
                validation_pts = np.array(validation_data['W1'].values)
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {iter_k}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
                
                # Restore previous fitted state
                self.is_fitted = old_fitted
                self.theta_hat = old_theta
                self.grid_midpoints = old_midpoints
                self.delta_j = old_delta_j
                self._grid_points_hal = old_grid_points_hal
            
            # Compute objective for logging and relative convergence check
            if iter_k % self.log_frequency == 0 or iter_k < 10:
                obj = self._objective_function(theta, phi_data_full, phi_grid_full, delta_j)
                
                # Early stopping if objective function explodes
                if self._check_objective_explosion(obj, iter_k):
                    break
                
                l1_norm = np.sum(np.abs(theta_new[1:]))
                h_min, h_max = np.min(h[1:]), np.max(h[1:])
                num_knots = np.sum(np.abs(theta_new[1:]) > self.tol)
                
                if self.do_log and iter_k % self.log_frequency == 0:
                    self.logger.info(f"Iter {iter_k:3d}: obj={obj:.6f}, change={change:.2e}, α={self.alpha:.3f}, ‖θ[1:]‖₁={l1_norm:.3f}, h_range=[{h_min:.2e}, {h_max:.2e}], num_knots={num_knots}")
                
                # Relative objective convergence check
                if obj_prev is not None:
                    rel_obj_change = abs(obj - obj_prev) / max(1, abs(obj_prev))
                    if rel_obj_change < self.relative_objective_tolerance and change < self.tol:
                        if self.do_log:
                            self.logger.info(f"Converged at iteration {iter_k} (relative objective change)")
                        break
                obj_prev = obj
            
            # Standard convergence check
            if change < self.tol:
                if self.do_log:
                    self.logger.info(f"Converged at iteration {iter_k}")
                break
            
            theta = theta_new
        
        # Store results (same as other estimators)
        self.theta_hat = theta
        self.grid_midpoints = midpoints
        self.delta_j = delta_j
        self._grid_points_hal = grid_points_hal
        
        # Select non-zero knots (only for truncated power terms)
        if self.basis_order == 0:
            # For step functions, skip intercept
            truncated_power_coeffs = self.theta_hat[1:]
        else:
            # For higher order, skip intercept and polynomial terms
            n_polynomial = self.basis_order + 1
            truncated_power_coeffs = self.theta_hat[n_polynomial:]
        
        mask = np.abs(truncated_power_coeffs) > self.tol
        self.grid_points_hal_selected = grid_points_hal[mask]
        
        # Public evaluation grid for density queries
        self.grid_points = np.linspace(0, 1, self.n_grid_points)
        
        # Final log
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={iter_k}")
        
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
            raise ValueError("Estimator must be fitted before getting results")
        
        return self._get_common_results()