import numpy as np
import pandas as pd
from typing import Optional
from src.haldensity.utils.basis import create_basis_functions
from src.haldensity.estimation.base_estimator import BaseEstimator


class ProximalGDEstimator(BaseEstimator):
    """
    Density estimator using ISTA (proximal gradient descent).
    
    This class implements the Iterative Shrinkage-Thresholding Algorithm (ISTA)
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
        """Initialize ISTA (proximal gradient descent) estimator."""
        super().__init__(
            lam=lam,
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.L = L
        self.n_grid_points = n_grid_points
        self.grid_points_hal_selected: Optional[np.ndarray] = None
        
    def fit(self, data: pd.DataFrame, warm_start_coefficients: Optional[np.ndarray] = None,
            validation_data: Optional[pd.DataFrame] = None,
            validation_frequency: int = -1) -> 'ProximalGDEstimator':
        """
        Fit the ISTA density estimator.
        
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
        
        # 3) initialise θ
        K = b_ik.shape[1]  # number of parameters
        
        if warm_start_coefficients is None:
            theta = np.zeros(K)
        elif len(warm_start_coefficients) == K:
            theta = np.array(warm_start_coefficients, dtype=float).copy()
        else:
            if self.do_log:
                self.logger.info(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}")
            theta = np.zeros(K)
        
        step = 1.0 / self.L
        
        if self.do_log:
            self.logger.info(f"Starting ProximalGD (ISTA) with K={K} parameters")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"ProximalGD: lam={self.lam}, L={self.L}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        # 4) ISTA loop (manual GD + prox)
        for it in range(self.n_iterations):
            # compute neg‐log‐lik (without penalty)
            log_f_data = b_ik @ theta           # (n,)
            term1 = -np.sum(log_f_data)
            log_f_grid = b_jk @ theta           # (m,)
            # use logsumexp for stability
            max_log = np.max(log_f_grid + log_delta_j)
            loss_nopen = term1 + n_samples * (max_log + np.log(np.sum(np.exp((log_f_grid + log_delta_j) - max_log))))
            
            # Early stopping if objective function explodes
            if self._check_objective_explosion(float(loss_nopen), it):
                break
            
            # gradient step (∇(−ℓ))
            grad_term1 = -np.sum(b_ik, axis=0)
            # weights under current density
            max_log_f = np.max(log_f_grid)
            f_grid_unnorm = np.exp(log_f_grid - max_log_f)
            weights_unnorm = f_grid_unnorm * delta_j
            Z = np.sum(weights_unnorm)
            weights = weights_unnorm / Z
            grad_term2 = n_samples * (weights[:, None] * b_jk).sum(axis=0)
            grad = grad_term1 + grad_term2
            
            theta_next = theta - step * grad
            # proximal: soft‐threshold θ[1:]
            v = theta_next[1:]
            shrunk = np.maximum(np.abs(v) - self.lam * step, 0.0)
            theta_next[1:] = np.sign(v) * shrunk
            
            # Intercept correction for exact normalization
            logZ = np.log(np.sum(np.exp(b_jk @ theta_next) * delta_j))
            if not np.isfinite(logZ):
                if self.do_log:
                    self.logger.warning(f"logZ became {logZ} at iteration {it}, stopping optimization")
                break
            theta_next[0] -= logZ  # Subtract from intercept only
            
            # stopping rule: max parameter change small
            change = float(np.max(np.abs(theta_next - theta)))
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and it % validation_frequency == 0:
                # Update parameters for validation
                self.theta_hat = theta_next
                self.grid_midpoints = midpoints
                self.delta_j = delta_j
                self._grid_points_hal = grid_points_hal
                
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {it}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
            
            if it % self.log_frequency == 0:
                l1norm = float(np.sum(np.abs(theta_next[1:])))
                num_selected_knots = int(np.sum(np.abs(theta_next[1:]) > self.tol))
                loss_val = float(loss_nopen)
                
                if self.do_log:
                    self.logger.info(f"Iter {it:4d}: loss={loss_val:.4f}, change={change:.2e}, ‖θ[1:]‖₁={l1norm:.3f}, num_selected_knots={num_selected_knots}")
            
            if change < self.tol:
                if self.do_log:
                    self.logger.info(f"Converged at iteration {it}")
                theta = theta_next
                break
            
            theta = theta_next
        
        # Store results
        self.theta_hat = theta
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
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={it}")
        
        # Choose an evaluation grid for public density API
        self.grid_points = np.linspace(0, 1, self.n_grid_points)
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