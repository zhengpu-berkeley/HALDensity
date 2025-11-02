import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.special import logsumexp
from src.haldensity.utils.basis import create_basis_functions
from src.haldensity.estimation.base_estimator import BaseEstimator


class ProximalNewtonLBFGSEstimator(BaseEstimator):
    """
    Density estimator using Proximal Newton method with simplified L-BFGS approximation.
    
    This is the "reduced" or "more convenient" version of the L-BFGS estimator that uses a 
    diagonal approximation of the L-BFGS Hessian for computational efficiency and simplicity.
    
    Mathematical Framework:
    ======================
    Solves the regularized HAL density estimation problem:
    
    min_θ  F(θ) = −ℓ(θ) + λ‖θ[1:]‖₁
    
    where:
    - ℓ(θ) is the HAL-basis log-likelihood: ℓ(θ) = ∑ᵢ log f(xᵢ) - N log(∫ f(x) dx)
    - f(x) = exp(φ(x)ᵀθ) with φ(x) being HAL basis functions
    - θ = [θ₀, θ₁, ..., θₖ] with θ₀ being the intercept (unpenalized)
    - λ > 0 is the L1 regularization parameter
    
    Key Algorithmic Simplification:
    ==============================
    Instead of using the full L-BFGS two-loop recursion, this implementation uses a 
    **diagonal approximation** of the Hessian:
    
    H⁻¹ ≈ (1/γₖ) * I
    
    where γₖ is the L-BFGS scaling factor computed from the most recent (s,y) pair:
    γₖ = yₖᵀyₖ / sₖᵀyₖ
    
    This allows for:
    1. **Vectorized proximal updates** instead of coordinate descent
    2. **Simpler implementation** without two-loop recursion
    3. **Lower computational cost** per iteration
    4. **Same memory efficiency** as full L-BFGS (O(mK) storage)
    
    Algorithm Overview:
    ==================
    1. Compute gradient ∇F(θₖ) of the smooth + non-smooth objective
    2. Use scalar approximation: H⁻¹ ≈ (1/γₖ)I for all diagonal entries
    3. Solve proximal subproblem via vectorized soft-thresholding:
       - Intercept: θ₀ ← θ₀ - (1/γₖ)∇F(θₖ)[0]
       - Penalized: θⱼ ← soft_thresh(θⱼ - (1/γₖ)∇F(θₖ)[j], λ/γₖ)
    4. Apply backtracking line search with Armijo condition
    5. Update L-BFGS memory with (sₖ, yₖ) pairs and compute new γₖ₊₁
    6. Enforce normalization constraint through intercept adjustment
    
    When to Use This Version:
    ========================
    ✓ **Recommended when:**
    - You want simpler, more interpretable L-BFGS implementation
    - Computational efficiency per iteration is important
    - Problem size is moderate to large (K > 100)
    - You don't need the full quasi-Newton approximation quality
    
    ✗ **Consider full L-BFGS when:**
    - Maximum convergence rate is critical
    - You have computational budget for two-loop recursion
    - Problem exhibits strong curvature that benefits from better Hessian approximation
    
    Comparison with ProximalNewtonLBFGSFullEstimator:
    ===============================================
    | Aspect                    | This (Reduced)        | Full L-BFGS          |
    |---------------------------|----------------------|----------------------|
    | Hessian approximation     | Diagonal H⁻¹≈(1/γ)I   | Two-loop recursion   |
    | Proximal update           | Vectorized           | Applied to direction |
    | Per-iteration cost        | O(K)                 | O(mK)               |
    | Implementation complexity | Simple               | More complex         |
    | Convergence rate          | Good                 | Potentially better   |
    | Memory usage             | O(mK)                | O(mK)               |
    
    Convergence Properties:
    ======================
    - Maintains global convergence under Armijo line search
    - Preserves sparsity-inducing properties of L1 regularization
    - Typically achieves good practical convergence rates
    - May require more iterations than full L-BFGS but with faster iterations
    
    References:
    ==========
    - Nocedal, J. & Wright, S. J. (2006). Numerical Optimization, 2nd ed. Springer.
    - Benkeser, D. et al. (2016). The highly adaptive lasso estimator. ICML.
    - Beck, A. & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm.
    """
    
    def __init__(
        self,
        lam: float = 3.0,
        n_iterations: int = 100,
        tol: float = 1e-6,
        n_grid_points: int = 200,
        line_search_c: float = 1e-4,
        max_line_search_steps: int = 20,
        lbfgs_memory: int = 5,  # number of (s,y) pairs to keep
        basis_order: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        non_desc_clip_alpha: bool = True,
        # Exposed hyperparameters
        lbfgs_initial_gamma: float = 1.0,  # Initial L-BFGS scaling factor
        lbfgs_update_tolerance: float = 1e-12,  # Tolerance for L-BFGS updates
        lbfgs_gamma_clip_range: Tuple[float, float] = (1e-6, 1e3),  # Clipping range for gamma_k
        line_search_beta: float = 0.5,  # Step size reduction factor for line search
        non_descent_step_size: float = 0.1  # Fallback step size for non-descent directions
    ):
        """
        Initialize Proximal Newton L-BFGS estimator with diagonal Hessian approximation.
        
        This is the "reduced" L-BFGS implementation that uses a scalar approximation
        H⁻¹ ≈ (1/γₖ)I instead of the full two-loop recursion for computational efficiency.
        
        Core Algorithm Parameters:
        ========================
        lam : float, default=3.0
            L1 regularization parameter (λ). Controls the sparsity of the solution.
            Higher values lead to more sparsity and smoother densities.
            Must be positive.
            
        n_iterations : int, default=100
            Maximum number of proximal Newton iterations.
            
        tol : float, default=1e-6
            Convergence tolerance for parameter changes. Algorithm terminates when
            max|θₖ₊₁ - θₖ| < tol.
            
        n_grid_points : int, default=200
            Number of grid points for numerical integration in normalization constraint.
            Used for computing ∫ f(x) dx via Riemann sum. Higher values improve accuracy
            but increase computational cost.
            
        basis_order : int, default=0
            Order of truncated power basis functions:
            - 0: Step functions (HAL-0), φⱼ(x) = I(x ≥ ξⱼ)
            - k≥1: Splines (HAL-k), includes polynomial terms x, x², ..., xᵏ and
              truncated power terms (x - ξⱼ)₊ᵏ
              
        Line Search Parameters:
        ======================
        line_search_c : float, default=1e-4
            Armijo condition parameter (c ∈ (0,1)). Controls the sufficient decrease
            condition: F(θₖ + αdₖ) ≤ F(θₖ) + c·α·∇F(θₖ)ᵀdₖ.
            
        max_line_search_steps : int, default=20
            Maximum number of backtracking line search steps.
            
        line_search_beta : float, default=0.5
            Step size reduction factor for backtracking (β ∈ (0,1)).
            At each backtracking step: α := β·α.
            
        non_desc_clip_alpha : bool, default=True
            Whether to clip step size for non-descent directions.
            
        non_descent_step_size : float, default=0.1
            Step size used when search direction is not a descent direction
            (∇F(θₖ)ᵀdₖ ≥ 0).
            
        Simplified L-BFGS Parameters:
        ============================
        lbfgs_memory : int, default=5
            Number of (sₖ, yₖ) pairs to keep in L-BFGS memory (m).
            Used only for computing the scaling factor γₖ, not for two-loop recursion.
            Typical values: 3-10 (less critical than in full L-BFGS).
            
        lbfgs_initial_gamma : float, default=1.0
            Initial L-BFGS scaling factor (γ₀). Used to initialize the diagonal
            approximation H₀⁻¹ = (1/γ₀)I.
            
        lbfgs_update_tolerance : float, default=1e-12
            Tolerance for L-BFGS updates. Only update γₖ if sₖᵀyₖ > tol
            to ensure numerical stability.
            
        lbfgs_gamma_clip_range : Tuple[float, float], default=(1e-6, 1e3)
            Clipping range for scaling factor γₖ = yₖᵀyₖ / sₖᵀyₖ.
            Prevents γₖ from becoming too large or too small.
            
        Logging Parameters:
        ==================
        log_dir : str, default="./local/logs/experiment.log"
            Directory for logging output.
            
        log_frequency : int, default=10
            Frequency of iteration logging. Set to -1 to disable logging.
            
        Key Algorithmic Differences from Full L-BFGS:
        =============================================
        1. **Diagonal Approximation**: Uses H⁻¹ ≈ (1/γₖ)I instead of full quasi-Newton matrix
        2. **Vectorized Updates**: Applies proximal operator element-wise instead of to search direction
        3. **Simpler Implementation**: No two-loop recursion, just γₖ updates
        4. **Lower Per-Iteration Cost**: O(K) vs O(mK) operations per iteration
        5. **Same Memory Efficiency**: Still O(mK) storage for (s,y) pairs
        
        This makes the algorithm more accessible and easier to understand while maintaining
        good practical performance for most HAL density estimation problems.
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
        
        # LBFGS-specific parameters
        self.n_grid_points = n_grid_points
        self.line_search_c = line_search_c
        self.max_line_search_steps = max_line_search_steps
        self.lbfgs_memory = lbfgs_memory
        self.non_desc_clip_alpha = non_desc_clip_alpha
        self.lbfgs_initial_gamma = lbfgs_initial_gamma
        self.lbfgs_update_tolerance = lbfgs_update_tolerance
        self.lbfgs_gamma_clip_range = lbfgs_gamma_clip_range
        self.line_search_beta = line_search_beta
        self.non_descent_step_size = non_descent_step_size
    
    def _soft_threshold_vectorized(self, z: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Vectorized soft thresholding operator.
        
        Args:
            z: Input array
            tau: Threshold array (same shape as z)
            
        Returns:
            Soft-thresholded array
        """
        return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)
    
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
            validation_frequency: int = -1) -> 'ProximalNewtonLBFGSEstimator':
        """
        Fit the Proximal Newton L-BFGS density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            warm_start_coefficients: Optional initial coefficients for warm starting
            validation_data: Optional validation data for tracking performance
            validation_frequency: Frequency for validation logging (default -1 means no validation)
            
        Returns:
            Self for method chaining
        """

        n_samples = len(data)
        
        # 1) HAL basis on observed W1 (same as first-order methods)
        grid_points_hal = np.unique(data['W1'].dropna())
        b_ik, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)  # (n, K−1)
        self.basis_names = basis_names


        # 2) midpoint grid for the log‐normaliser
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        delta_j = grid_eval[1:] - grid_eval[:-1]  # numpy array
        df_mid = pd.DataFrame({'W1': midpoints})
        b_jk, _ = create_basis_functions(df_mid, grid_points_hal, order=self.basis_order)  # (m, K−1)
        
        # Convert to numpy arrays - these already include intercept in the new basis
        phi_data_full = b_ik  # (N, K)
        phi_grid_full = b_jk  # (m, K)
        
        # 3) Initialize θ = [θ₀, θ₁, ..., θₖ₋₁]
        # K is determined by the basis function output dimensions
        K = phi_data_full.shape[1]
        
        if warm_start_coefficients is None:
            theta = np.zeros(K)
        elif len(warm_start_coefficients) == K:
            theta = warm_start_coefficients.copy()
        else:
            if self.do_log:
                self.logger.info(f"Warm start failed: expected {K} coefficients, got {len(warm_start_coefficients)}")
            theta = np.zeros(K)

        
        # Initialize L-BFGS storage
        s_list: list[np.ndarray] = []  # stores s_k = θ_{k+1} - θ_k
        y_list: list[np.ndarray] = []  # stores y_k = g_{k+1} - g_k  
        gamma_k = self.lbfgs_initial_gamma  # initial Hessian scale
        
        if self.do_log:
            self.logger.info(f"Starting Proximal Newton + L-BFGS with K={K} parameters, memory={self.lbfgs_memory}")
        
        # Simple initialization log
        if self.do_log:
            self.logger.info(f"ProximalNewtonLBFGS: lam={self.lam}, memory={self.lbfgs_memory}, n_grid={self.n_grid_points}, basis_order={self.basis_order}, n_samples={n_samples}, K={K}")
        
        # Main proximal Newton loop
        for iter_k in range(self.n_iterations):
            # 1. Compute gradient
            gradient = self._compute_gradient(theta, phi_data_full, phi_grid_full, 
                                            delta_j, n_samples)
            
            # 2. Use L-BFGS approximation for Hessian diagonal
            # Instead of computing full Hessian, use scalar gamma_k for all diagonal entries
            h_diag = np.full(K, max(gamma_k, 1e-8))
            
            # 3. Solve Newton subproblem via vectorized proximal update
            # Since h_diag is a scaled identity matrix, we can solve the proximal subproblem directly
            
            # For the intercept (unpenalized): standard Newton step
            d_newton = np.zeros(K)
            d_newton[0] = -gradient[0] / h_diag[0]
            
            # For penalized coordinates: vectorized proximal operator
            if K > 1:
                # Compute the proximal step for all penalized coordinates
                raw = theta[1:] - gradient[1:] / h_diag[1:]  # θ - g/h (Newton direction)
                thresh = self.lam / h_diag[1:]  # Threshold array
                z = self._soft_threshold_vectorized(raw, thresh)
                d_newton[1:] = z - theta[1:]
            
            # 4. Line search with Armijo condition
            alpha = 1.0
            obj_current = self._objective_function(theta, phi_data_full, phi_grid_full, delta_j)
            
            # Early stopping if objective function explodes
            if self._check_objective_explosion(obj_current, iter_k):
                break
            
            # Directional derivative for line search
            directional_deriv = np.dot(gradient, d_newton)
            
            if directional_deriv >= -1e-12:  # Not a descent direction
                if self.do_log and iter_k % self.log_frequency == 0:
                    self.logger.warning(f"Non-descent direction at iteration {iter_k}, directional_deriv={directional_deriv:.2e}")
                if self.non_desc_clip_alpha:
                    alpha = self.non_descent_step_size
            else:
                # Backtracking line search with Armijo condition
                for ls_step in range(self.max_line_search_steps):
                    theta_trial = theta + alpha * d_newton
                    obj_trial = self._objective_function(theta_trial, phi_data_full, phi_grid_full, delta_j)
                    
                    # Armijo condition
                    if obj_trial <= obj_current + self.line_search_c * alpha * directional_deriv:
                        break
                    alpha *= self.line_search_beta
                else:
                    if self.do_log and iter_k % self.log_frequency == 0:
                        self.logger.warning(f"Line search failed at iteration {iter_k}, using small step")
                    if self.non_desc_clip_alpha:
                        alpha = self.non_descent_step_size
            
            # 5. Update
            theta_new = theta + alpha * d_newton
            
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
            
            # 6. Compute new gradient for L-BFGS update
            gradient_new = self._compute_gradient(theta_new, phi_data_full, phi_grid_full, 
                                                delta_j, n_samples)
            
            # 7. Update L-BFGS memory
            s_k = theta_new - theta
            y_k = gradient_new - gradient
            
            # Store (s,y) pair in FIFO buffer
            if len(s_list) == self.lbfgs_memory:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s_k.copy())
            y_list.append(y_k.copy())
            
            # Update gamma_k for next iteration using most recent (s,y) pair
            if np.dot(s_k, y_k) > self.lbfgs_update_tolerance:
                gamma_k = np.dot(y_k, y_k) / np.dot(y_k, s_k)
            # Clip gamma_k to prevent it from becoming too large or too small
            gamma_k = np.clip(gamma_k, self.lbfgs_gamma_clip_range[0], self.lbfgs_gamma_clip_range[1])

            
            # 8. Check convergence
            change = np.max(np.abs(theta_new - theta))
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and iter_k % validation_frequency == 0:
                # Temporarily update parameters for validation
                old_fitted = self.is_fitted
                self.theta_hat = theta_new
                self.grid_midpoints = midpoints
                self.delta_j = delta_j
                self._grid_points_hal = grid_points_hal
                self.is_fitted = True  # Temporarily set to fitted for validation
                
                validation_pts = validation_data['W1'].values
                validation_sum_log_likelihood = self.get_sum_log_likelihood_for_points(validation_pts)
                if self.do_log:
                    self.logger.info(f"Validation at iter {iter_k}: sum_log_likelihood={validation_sum_log_likelihood:.6f}")
                
                # Restore previous fitted state
                self.is_fitted = old_fitted
            
            if self.do_log and iter_k % self.log_frequency == 0:
                l1_norm = np.sum(np.abs(theta_new[1:]))  # Only penalized coefficients
                num_knots = np.sum(np.abs(theta_new[1:]) > self.tol)
                
                self.logger.info(f"Iter {iter_k:3d}: obj={obj_current:.6f}, change={change:.2e}, α={alpha:.3f}, ‖θ[1:]‖₁={l1_norm:.3f}, γ={gamma_k:.3f}, num_knots={num_knots}")
            
            if change < self.tol:
                if self.do_log:
                    self.logger.info(f"Converged at iteration {iter_k}")
                break
                
            theta = theta_new
        
        # Store results
        self.theta_hat = theta
        self.grid_midpoints = midpoints
        self.delta_j = delta_j
        self._grid_points_hal = grid_points_hal
        
        # Select non-zero knots (only for truncated power terms, not polynomial terms)
        if self.basis_order == 0:
            # For order 0: theta = [intercept, step_functions...]
            truncated_power_coeffs = self.theta_hat[1:]
        else:
            # For order k≥1: theta = [intercept, x, x^2, ..., x^k, (x-ξ₁)₊^k, ...]
            truncated_power_coeffs = self.theta_hat[1 + self.basis_order:]
        
        mask = np.abs(truncated_power_coeffs) > self.tol
        self.grid_points_hal_selected = grid_points_hal[mask]
        
        # Create evaluation grid for density
        self.grid_points = np.linspace(0, 1, 200)
        
        # Final log
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={iter_k}")
        
        self.is_fitted = True

        assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
        self.fitted_theta_dict = {name: value for name, value in zip(self.basis_names, self.theta_hat.tolist())}

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