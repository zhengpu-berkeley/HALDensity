import numpy as np
import pandas as pd
from typing import Optional, Tuple
from src.haldensity.utils.basis import create_basis_functions
from scipy.special import logsumexp
from src.haldensity.estimation.base_estimator import BaseEstimator


class ProximalNewtonLBFGSFullEstimator(BaseEstimator):
    """
    Density estimator using Proximal Newton method with full L-BFGS Hessian approximation.
    
    This class implements the proximal Newton algorithm with L-BFGS Hessian approximation for solving 
    the regularized density estimation problem in the context of Highly Adaptive Lasso (HAL) density estimation:
    
    min_θ  F(θ) = −ℓ(θ) + λ‖θ[1:]‖₁
    
    where:
    - ℓ(θ) is the HAL-basis log-likelihood: ℓ(θ) = ∑ᵢ log f(xᵢ) - N log(∫ f(x) dx)
    - f(x) = exp(φ(x)ᵀθ) with φ(x) being HAL basis functions
    - θ = [θ₀, θ₁, ..., θₖ] with θ₀ being the intercept (unpenalized)
    - λ > 0 is the L1 regularization parameter
    - The HAL basis functions are cumulative indicators I(x ≥ grid_point) for order 0,
      or spline basis functions for higher orders
    
    Mathematical Framework:
    The algorithm solves the non-smooth optimization problem by combining:
    1. L-BFGS approximation of the smooth part (log-likelihood Hessian)
    2. Proximal operator for handling the non-smooth L1 penalty
    3. Line search with Armijo condition for step size control
    
    Algorithm Overview:
    1. Compute gradient ∇F(θₖ) = ∇ℓ(θₖ) + λ∂‖θₖ[1:]‖₁
    2. Use L-BFGS two-loop recursion to compute search direction dₖ ≈ -H̃ₖ⁻¹∇F(θₖ)
    3. Apply proximal operator to handle L1 regularization: prox_λ/γ(θₖ + dₖ)
    4. Perform backtracking line search with Armijo condition
    5. Update L-BFGS memory with (sₖ, yₖ) pairs where sₖ = θₖ₊₁ - θₖ, yₖ = ∇F(θₖ₊₁) - ∇F(θₖ)
    6. Update scaling factor γₖ₊₁ = yₖᵀyₖ / sₖᵀyₖ for next iteration
    
    Key Features:
    - Uses full L-BFGS two-loop recursion (not diagonal approximation)
    - Maintains limited memory of m (s,y) pairs instead of full K×K Hessian matrix
    - Applies proximal operator to the L-BFGS direction for proper L1 regularization
    - Memory efficient O(mK) storage for large K (number of basis functions)
    - Numerically stable with conservative step size control and parameter clipping
    - Exact normalization constraint enforcement through intercept adjustment
    - Early stopping mechanisms to prevent numerical explosion
    
    Comparison with Other Methods:
    - vs. ProximalNewtonLBFGS: Uses full L-BFGS vs. diagonal Hessian approximation
    - vs. ProximalNewtonLBFGSCD: Uses full L-BFGS vs. coordinate descent on L-BFGS direction
    - vs. First-order methods: Second-order convergence properties with quasi-Newton approximation
    
    Convergence Properties:
    - Superlinear convergence rate for strongly convex problems
    - Global convergence guarantees under Armijo line search
    - Sparsity-inducing L1 regularization with exact proximal operator
    
    References:
    - Nocedal, J. & Wright, S. J. (2006). Numerical Optimization, 2nd ed. Springer.
    - Benkeser, D. et al. (2016). The highly adaptive lasso estimator. ICML.
    - Liu, D. C. & Nocedal, J. (1989). On the limited memory BFGS method. Mathematical Programming.
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
        # L-BFGS hyperparameters
        lbfgs_initial_gamma: float = 1.0,  # Initial L-BFGS scaling factor
        line_search_beta: float = 0.5,  # Step size reduction factor for line search
        failed_line_search_step_size: float = 0.01,  # Step size when line search fails
        non_descent_step_size: float = 0.1,  # Fallback step size for non-descent directions
        lbfgs_update_tolerance: float = 1e-12,  # Tolerance for L-BFGS updates
        lbfgs_gamma_clip_range: Tuple[float, float] = (1e-6, 1e6),  # Clipping range for gamma_k
    ):
        """
        Initialize Proximal Newton L-BFGS Full estimator.
        
        This implementation uses the full L-BFGS two-loop recursion algorithm
        combined with proximal operators for L1 regularization in HAL density estimation.
        
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
            Smaller values require more decrease but may cause convergence issues.
            
        max_line_search_steps : int, default=20
            Maximum number of backtracking line search steps.
            
        line_search_beta : float, default=0.5
            Step size reduction factor for backtracking (β ∈ (0,1)).
            At each backtracking step: α := β·α.
            
        failed_line_search_step_size : float, default=0.01
            Fallback step size when line search fails to find acceptable step.
            
        non_descent_step_size : float, default=0.1
            Step size used when search direction is not a descent direction
            (∇F(θₖ)ᵀdₖ ≥ 0).
            
        non_desc_clip_alpha : bool, default=True
            Whether to clip step size for non-descent directions.
            
        L-BFGS Parameters:
        =================
        lbfgs_memory : int, default=5
            Number of (sₖ, yₖ) pairs to keep in L-BFGS memory (m).
            Higher values provide better Hessian approximation but increase
            memory usage. Typical values: 3-20.
            
        lbfgs_initial_gamma : float, default=1.0
            Initial L-BFGS scaling factor (γ₀). Used to initialize the Hessian
            approximation H₀ = γ₀·I.
            
        lbfgs_update_tolerance : float, default=1e-12
            Tolerance for L-BFGS updates. Only update memory if sₖᵀyₖ > tol
            and yₖᵀyₖ > tol to ensure numerical stability.
            
        lbfgs_gamma_clip_range : Tuple[float, float], default=(1e-6, 1e3)
            Clipping range for scaling factor γₖ = yₖᵀyₖ / sₖᵀyₖ.
            Prevents γₖ from becoming too large or too small.
            
        Logging Parameters:
        ==================
        log_dir : str, default="./local/logs/experiment.log"
            Directory for logging output.
            
        log_frequency : int, default=10
            Frequency of iteration logging. Set to -1 to disable logging.
            
        Notes:
        ======
        - This implementation replaces the coordinate descent approach with
          full L-BFGS two-loop recursion for better convergence properties.
        - The proximal operator is applied to the L-BFGS direction to handle
          L1 regularization exactly.
        - Conservative parameter clipping and gamma updates ensure numerical stability.
        - The algorithm automatically enforces the normalization constraint
          ∫ f(x) dx = 1 through intercept adjustment.
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
        self.line_search_beta = line_search_beta
        self.failed_line_search_step_size = failed_line_search_step_size
        self.non_descent_step_size = non_descent_step_size
        self.lbfgs_update_tolerance = lbfgs_update_tolerance
        self.lbfgs_gamma_clip_range = lbfgs_gamma_clip_range
        
    def _soft_threshold(self, z: float, tau: float) -> float:
        """
        Soft thresholding operator (proximal operator for L1 norm).
        
        Computes the proximal operator of the L1 norm:
        prox_τ(z) = sign(z) * max(|z| - τ, 0)
        
        This is the solution to: min_x (1/2)(x - z)² + τ|x|
        
        Args:
            z: Input value
            tau: Threshold parameter (must be non-negative)
            
        Returns:
            Soft-thresholded value
            
        Note:
            This is a key component of the proximal operator for L1 regularization.
            Applied element-wise to handle the non-smooth L1 penalty term.
        """
        return np.sign(z) * max(abs(z) - tau, 0.0)
    
    def _objective_function(self, theta: np.ndarray, phi_data_full: np.ndarray, 
                           phi_grid_full: np.ndarray, delta_j: np.ndarray) -> float:
        """
        Compute the penalized objective function F(θ) for HAL density estimation.
        
        The objective function is:
        F(θ) = -ℓ(θ) + λ‖θ[1:]‖₁
        
        where the negative log-likelihood is:
        -ℓ(θ) = -∑ᵢ₌₁ⁿ log f(xᵢ) + N log(∫ f(x) dx)
               = -∑ᵢ₌₁ⁿ φ(xᵢ)ᵀθ + N log(∫ exp(φ(x)ᵀθ) dx)
        
        The integral is approximated using Riemann sum over the grid:
        ∫ exp(φ(x)ᵀθ) dx ≈ ∑ⱼ exp(φ(xⱼ)ᵀθ) * Δⱼ
        
        Args:
            theta: Parameter vector [θ₀, θ₁, ..., θₖ] where θ₀ is the intercept
            phi_data_full: Basis matrix at data points (N, K) including intercept column
            phi_grid_full: Basis matrix at grid midpoints (m, K) including intercept column
            delta_j: Grid widths for Riemann sum integration (m,)
            
        Returns:
            Objective function value F(θ)
            
        Notes:
            - Uses logsumexp for numerical stability in computing log integral
            - Only θ[1:] are penalized (intercept θ₀ is unpenalized)
            - Lower values indicate better fit
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
        Compute gradient of the smooth part of the objective function.
        
        The gradient of the negative log-likelihood is:
        ∇(-ℓ(θ)) = -∑ᵢ₌₁ⁿ φ(xᵢ) + N * E_f[φ(X)]
        
        where E_f[φ(X)] is the expectation under the current density estimate:
        E_f[φ(X)] = ∫ φ(x) f(x) dx / ∫ f(x) dx
                  ≈ ∑ⱼ φ(xⱼ) exp(φ(xⱼ)ᵀθ) Δⱼ / ∑ⱼ exp(φ(xⱼ)ᵀθ) Δⱼ
        
        Note: This computes the gradient of only the smooth part (log-likelihood).
        The L1 penalty gradient is handled separately in the proximal operator.
        
        Args:
            theta: Current parameter vector [θ₀, θ₁, ..., θₖ]
            phi_data_full: Basis matrix at data points (N, K) including intercept
            phi_grid_full: Basis matrix at grid points (m, K) including intercept
            delta_j: Grid widths for numerical integration (m,)
            n_samples: Number of data samples (N)
            
        Returns:
            Gradient vector ∇(-ℓ(θ)) of shape (K,)
            
        Implementation Notes:
            - Uses max subtraction for numerical stability in exp calculations
            - Computes weighted expectation under current density estimate
            - Does not include L1 penalty gradient (handled by proximal operator)
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
            validation_frequency: int = -1) -> 'ProximalNewtonLBFGSFullEstimator':
        """
        Fit the Proximal Newton L-BFGS density estimator to data.
        
        This method implements the full proximal Newton algorithm with L-BFGS Hessian
        approximation to solve the regularized HAL density estimation problem:
        
        min_θ  F(θ) = -ℓ(θ) + λ‖θ[1:]‖₁
        
        where ℓ(θ) is the HAL log-likelihood and θ[0] is the unpenalized intercept.
        
        Algorithm Outline:
        ================
        1. **Setup**: Create HAL basis functions φ(x) at observed data points and
           evaluation grid for numerical integration
        2. **Initialization**: Initialize θ (with warm start if provided) and L-BFGS memory
        3. **Main Loop**: For each iteration k = 0, 1, ..., max_iterations:
           a. Compute gradient ∇F(θₖ) of the smooth part
           b. Compute L-BFGS search direction using two-loop recursion
           c. Solve proximal subproblem with L1 regularization
           d. Perform backtracking line search with Armijo condition
           e. Update parameters θₖ₊₁ and clip to prevent explosion
           f. Enforce normalization constraint by adjusting intercept
           g. Update L-BFGS memory with new (s,y) pair
           h. Check convergence based on parameter change
        4. **Post-processing**: Select non-zero knots and prepare final results
        
        Args:
            data : pd.DataFrame
                Input data with column 'W1' containing the observations.
                Should contain values in [0, 1] for density estimation.
                
            warm_start_coefficients : np.ndarray, optional
                Initial parameter values for warm starting the optimization.
                If provided, must have length K matching the number of basis functions.
                If None or incompatible, initializes with zeros.
                
            validation_data : pd.DataFrame, optional
                Optional validation dataset with same structure as training data.
                Used for tracking validation performance during training.
                
            validation_frequency : int, default=-1
                Frequency for computing validation metrics.
                If > 0, computes validation log-likelihood every validation_frequency iterations.
                If -1, no validation is performed.
                
        Returns:
            self : ProximalNewtonLBFGSFullEstimator
                Returns the fitted estimator for method chaining.
                
        Attributes Set After Fitting:
            theta_hat : np.ndarray
                Fitted parameters [θ₀, θ₁, ..., θₖ]
            grid_points_hal_selected : np.ndarray
                Selected knot points where |θⱼ| > tolerance
            basis_names : list[str]
                Names of basis functions corresponding to parameters
            fitted_theta_dict : dict[str, float]
                dictionary mapping basis names to fitted coefficients
            is_fitted : bool
                Set to True after successful fitting
                
        Raises:
            ValueError: If data format is incorrect or contains invalid values
            
        Implementation Details:
        ======================
        - **Basis Construction**: Uses HAL basis functions I(x ≥ ξⱼ) for order 0,
          or spline basis for higher orders
        - **Numerical Integration**: Approximates ∫ f(x) dx using Riemann sum
          over uniform grid with n_grid_points
        - **Normalization**: Enforces ∫ f(x) dx = 1 by adjusting intercept after each step
        - **Line Search**: Backtracking with Armijo condition for step size control
        - **Memory Management**: FIFO buffer for L-BFGS (s,y) pairs
        - **Numerical Stability**: Parameter clipping, conservative gamma updates,
          early stopping for explosion detection
        - **Convergence**: Based on max absolute change in parameters
        
        Performance Characteristics:
        ===========================
        - **Time Complexity**: O(K² + mK) per iteration where K is basis size, m is L-BFGS memory
        - **Space Complexity**: O(K² + mK) for basis matrices and L-BFGS storage
        - **Convergence Rate**: Superlinear for strongly convex problems
        - **Sparsity**: L1 regularization induces sparse solutions
        
        Notes:
        ======
        - The algorithm automatically handles the normalization constraint
        - L-BFGS memory is updated using FIFO buffer to maintain constant memory usage
        - Early stopping prevents numerical explosion in pathological cases
        - Validation can be used for hyperparameter tuning or early stopping
        
        Example:
        ========
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({'W1': np.random.uniform(0, 1, 1000)})
        >>> estimator = ProximalNewtonLBFGSFullEstimator(lam=3.0, n_iterations=50)
        >>> estimator.fit(data)
        >>> grid, density = estimator.get_density()
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
        
        # Basis matrices already include intercept in the new basis
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
            
            # 2. Compute L-BFGS search direction using two-loop recursion
            lbfgs_direction = self._compute_lbfgs_direction(gradient, s_list, y_list, gamma_k)
            
            # 3. Solve proximal subproblem with L-BFGS direction
            d_newton = self._solve_proximal_subproblem(theta, lbfgs_direction, gamma_k, iteration=iter_k)
            
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
                    self.logger.warning(f"Non-descent direction at iteration {iter_k}, directional_deriv={directional_deriv:.2e}, setting alpha from {alpha} to 0.1")
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
                    alpha = self.failed_line_search_step_size
            
            # 5. Update
            theta_new = theta + alpha * d_newton
            
            # Clip parameters to prevent explosion
            theta_new = np.clip(theta_new, -50.0, 50.0)

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
            sy_dot = np.dot(s_k, y_k)
            yy_dot = np.dot(y_k, y_k)
            
            if sy_dot > self.lbfgs_update_tolerance and yy_dot > self.lbfgs_update_tolerance:
                gamma_k_new = yy_dot / sy_dot
                # Use a more conservative update for gamma_k
                gamma_k = 0.9 * gamma_k + 0.1 * gamma_k_new
            
            # Clip gamma_k to prevent it from becoming too large or too small
            gamma_k = np.clip(gamma_k, self.lbfgs_gamma_clip_range[0], self.lbfgs_gamma_clip_range[1])

            
            # 8. Check convergence
            change = np.max(np.abs(theta_new - theta))
            
            # Validation and logging
            if validation_data is not None and validation_frequency > 0 and iter_k % validation_frequency == 0:
                # Update parameters for validation
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
            
            if self.do_log and iter_k % self.log_frequency == 0:  # Report every log interval
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
        self.grid_points = np.linspace(0, 1, self.n_grid_points)
        
        # Final log
        final_selected_knots = np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        if self.do_log:
            self.logger.info(f"Final: selected_knots={final_selected_knots}, iterations={iter_k}")
        
        self.is_fitted = True

        # Store the fitted theta as a dictionary for inspection
        assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
        self.fitted_theta_dict = {name: value for name, value in zip(self.basis_names, self.theta_hat.tolist())}

        return self
    
    def get_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """Use shared, numerically stable density from BaseEstimator."""
        return super().get_density()
    
    def get_results(self) -> dict:
        """
        Get comprehensive results from the fitting process.
        
        Returns a dictionary containing all relevant results from the density estimation,
        including fitted parameters, selected knots, basis information, and metadata.
        
        Returns:
            dict containing the following keys:
            
            **Parameters & Coefficients:**
            - 'theta_hat': Fitted parameter vector [θ₀, θ₁, ..., θₖ]
            - 'fitted_theta_dict': dictionary mapping basis names to coefficients
            - 'intercept': Intercept term θ₀ (unpenalized)
            - 'non_zero_coefficients': Number of non-zero coefficients
            - 'selected_knots': Number of selected knot points
            
            **Basis Information:**
            - 'basis_names': list of basis function names
            - 'basis_order': Order of the basis functions used
            - 'grid_points_hal': HAL basis knot points
            - 'grid_points_hal_selected': Selected knot points (non-zero coefficients)
            
            **Density Estimation:**
            - 'grid_points': Evaluation grid points
            - 'density_values': Density estimates at grid points
            - 'log_likelihood': Log-likelihood at fitted parameters
            
            **Regularization:**
            - 'lambda': L1 regularization parameter used
            - 'l1_penalty': L1 penalty value ||θ[1:]||₁
            - 'objective_value': Final objective function value
            
            **Algorithm Metadata:**
            - 'n_iterations_run': Number of iterations performed
            - 'converged': Whether algorithm converged
            - 'method': Algorithm name and version
            - 'lbfgs_memory': L-BFGS memory size used
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
            
        Example:
        ========
        >>> estimator.fit(data)
        >>> results = estimator.get_results()
        >>> print(f"Selected knots: {results['selected_knots']}")
        >>> print(f"L1 penalty: {results['l1_penalty']:.3f}")
        >>> print(f"Log-likelihood: {results['log_likelihood']:.3f}")
        
        Notes:
        ======
        - This method calls the parent class `_get_common_results()` method
        - Results are computed based on the current fitted state
        - Density values are computed on the evaluation grid
        - All results are returned as native Python types (not PyTorch tensors)
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        
        return self._get_common_results()
    
    def _compute_lbfgs_direction(self, gradient: np.ndarray, s_list: list[np.ndarray], 
                                y_list: list[np.ndarray], gamma_k: float) -> np.ndarray:
        """
        Compute L-BFGS search direction using two-loop recursion algorithm.
        
        This implements the L-BFGS two-loop recursion to compute the search direction:
        dₖ = -H̃ₖ⁻¹ ∇F(θₖ)
        
        where H̃ₖ is the L-BFGS Hessian approximation built from the most recent
        m (sᵢ, yᵢ) pairs stored in memory.
        
        Algorithm (Nocedal & Wright, 2006, Algorithm 7.4):
        1. First loop (backward): q := ∇F(θₖ), then for i = m-1, ..., 0:
           - αᵢ := ρᵢ sᵢᵀ q
           - q := q - αᵢ yᵢ
        2. Scaling: r := H₀⁻¹ q = γₖ q
        3. Second loop (forward): for i = 0, ..., m-1:
           - βᵢ := ρᵢ yᵢᵀ r
           - r := r + (αᵢ - βᵢ) sᵢ
        4. Return: dₖ := -r
        
        Args:
            gradient: Current gradient vector ∇F(θₖ) of shape (K,)
            s_list: list of parameter differences sᵢ = θᵢ₊₁ - θᵢ
            y_list: list of gradient differences yᵢ = ∇F(θᵢ₊₁) - ∇F(θᵢ)
            gamma_k: L-BFGS scaling factor for initial Hessian approximation
            
        Returns:
            L-BFGS search direction dₖ (negative for descent)
            
        Implementation Notes:
            - Uses conservative gamma clipping to prevent numerical instability
            - Includes direction clipping to prevent extreme steps
            - Falls back to scaled steepest descent if no history available
            - Skips updates when sᵢᵀyᵢ is too small (curvature condition)
            
        References:
            Nocedal, J. & Wright, S. J. (2006). Numerical Optimization, 2nd ed.
            Algorithm 7.4: L-BFGS two-loop recursion.
        """
        # Use a more conservative gamma_k to prevent explosion
        effective_gamma = np.clip(gamma_k, self.lbfgs_gamma_clip_range[0], self.lbfgs_gamma_clip_range[1])
        
        if len(s_list) == 0:
            # No history available, use scaled steepest descent
            return -effective_gamma * gradient
            
        q = gradient.copy()
        m = len(s_list)
        alpha = np.zeros(m)
        
        # First loop (backward through history)
        for i in range(m-1, -1, -1):
            sy_dot = np.dot(y_list[i], s_list[i])
            if abs(sy_dot) > self.lbfgs_update_tolerance:
                rho_i = 1.0 / sy_dot
                alpha[i] = rho_i * np.dot(s_list[i], q)
                q -= alpha[i] * y_list[i]
        
        # Scale by initial Hessian approximation
        r = effective_gamma * q
        
        # Second loop (forward through history)
        for i in range(m):
            sy_dot = np.dot(y_list[i], s_list[i])
            if abs(sy_dot) > self.lbfgs_update_tolerance:
                rho_i = 1.0 / sy_dot
                beta = rho_i * np.dot(y_list[i], r)
                r += (alpha[i] - beta) * s_list[i]
        
        # Return negative for descent direction with additional clipping
        direction = -r
        
        # Clip the direction to prevent extreme steps
        max_direction_norm = 5.0
        direction_norm = np.linalg.norm(direction)
        if direction_norm > max_direction_norm:
            direction = direction * (max_direction_norm / direction_norm)
        
        return direction
    
    def _solve_proximal_subproblem(self, theta: np.ndarray, lbfgs_direction: np.ndarray, gamma_k: float, iteration: Optional[int] = None) -> np.ndarray:
        """
        Solve the proximal subproblem using the L-BFGS direction.
        
        This method computes the proximal Newton step by combining the L-BFGS direction
        with the proximal operator for L1 regularization.
        
        Theoretical Background:
        The proximal Newton method solves subproblems of the form:
        min_d  (1/2) dᵀ H̃ₖ d + ∇F(θₖ)ᵀ d + λ ‖θₖ + d‖₁
        
        For L1 regularization, this can be approximately solved by:
        1. Computing the Newton direction d̃ₖ = -H̃ₖ⁻¹ ∇F(θₖ) (L-BFGS direction)
        2. Applying the proximal operator element-wise:
           For i = 0: dᵢ = d̃ᵢ (intercept, unpenalized)
           For i ≥ 1: dᵢ = prox_{λ/γₖ}(θᵢ + d̃ᵢ) - θᵢ
        
        where prox_{τ}(z) = sign(z) * max(|z| - τ, 0) is the soft thresholding operator.
        
        Args:
            theta: Current parameter vector [θ₀, θ₁, ..., θₖ]
            lbfgs_direction: L-BFGS search direction d̃ₖ
            gamma_k: L-BFGS scaling factor (used for proximal step size)
            
        Returns:
            Proximal Newton step direction dₖ
            
        Implementation Details:
            - Clips the L-BFGS direction to prevent extreme steps
            - Uses conservative gamma_k for proximal operator step size
            - Applies soft thresholding only to regularized parameters (θ[1:])
            - Intercept θ₀ is not regularized and uses pure Newton direction
            
        Notes:
            This approximation works well when the L-BFGS direction provides
            a good second-order approximation to the smooth part of the objective.
        """
        K = len(theta)
        # On the first iteration, skip proximal shrinkage to avoid zero update
        if iteration is not None and iteration == 0:
            return lbfgs_direction.copy()
        d = lbfgs_direction.copy()
        
        # Clip the L-BFGS direction to prevent extreme steps
        max_step = 1.0
        d_norm = np.linalg.norm(d)
        if d_norm > max_step:
            d = d * (max_step / d_norm)
        
        # Use a more conservative step size for proximal operator
        # Ensure gamma_k doesn't become too small
        effective_gamma = max(gamma_k, self.lbfgs_gamma_clip_range[0])
        
        # For regularized coordinates (all except intercept), apply proximal operator
        for r in range(1, K):  # Skip intercept (r=0)
            # Compute the proposed new value
            theta_new_r = theta[r] + d[r]
            
            # Apply soft thresholding with controlled step size
            # Use effective gamma to determine the proximal step size
            thresh = self.lam / effective_gamma
            
            # Soft threshold the proposed new value
            theta_prox_r = self._soft_threshold(theta_new_r, thresh)
            
            # Update the direction to reach the proximal point
            d[r] = theta_prox_r - theta[r]
        
        return d