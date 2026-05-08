import numpy as np
import pandas as pd
from typing import Optional
import logging
import os
from scipy.interpolate import interp1d
from pydantic import BaseModel
from haldensity.utils.basis import create_basis_functions

EXPLOSION_THRESHOLD = 1e5  # Threshold for detecting numerical explosion in optimization
LOGGER = logging.getLogger(__name__)

class CommonEstimatorResults(BaseModel):
    """Standardized results across estimators (Pydantic v2 model)."""
    fitted_theta_dict: Optional[dict[str, float]] = None
    theta_hat: list[float]
    data_points: Optional[list[float]] = None
    grid_points_hal_selected: Optional[list[float]] = None
    n_selected_knots: int
    estimated_density: list[float]
    grid_points: list[float]
    intercept: float
    hal_coeffs: list[float]


class BaseEstimator:
    """Lightweight base providing logging, density utilities, and common result helpers."""
    
    def __init__(
        self,
        lam: float = 3.0,
        n_iterations: int = 100,
        tol: float = 1e-18,
        basis_order: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        **kwargs  # Allow subclasses to pass additional parameters
    ):
        """
        Initialize base estimator.
        
        Args:
            lam: L1 regularization parameter
            n_iterations: Maximum number of iterations
            tol: Convergence tolerance
            basis_order: Order of the truncated power basis
            log_dir: Directory for logging
            log_frequency: Frequency of logging (-1 means no logging)
            **kwargs: Additional parameters for subclasses
        """
        self.lam = lam
        self.n_iterations = n_iterations
        self.tol = tol
        self.basis_order = basis_order
        self.is_fitted = False

        self.explosion_threshold = EXPLOSION_THRESHOLD

        self.basis_names: Optional[list] = None
        self.fitted_theta_dict: Optional[dict[str, float]] = None
        
        # Will be set during fitting
        self.theta_hat: Optional[np.ndarray] = None
        self.grid_points: Optional[np.ndarray] = None
        self._grid_points_hal: Optional[np.ndarray] = None
        self.grid_midpoints: Optional[np.ndarray] = None
        self.delta_j: Optional[np.ndarray] = None
        self.grid_points_hal_selected: Optional[np.ndarray] = None
        
        # Setup logging
        self.log_dir = log_dir
        self.do_log = (log_frequency > 0) and (log_dir is not None)
        self.log_frequency = log_frequency
        if self.do_log:
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup a dedicated logger for the estimator instance."""
        # Ensure the log directory exists.
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)

        # Create a unique logger name for this instance to avoid conflicts.
        logger_name = f"{self.__class__.__name__}-{id(self)}"
        self.logger = logging.getLogger(logger_name)
        
        # Prevent messages from propagating to the root logger.
        self.logger.propagate = False
        
        # Set the logging level.
        self.logger.setLevel(logging.INFO)

        # If handlers are already present, clear them to avoid duplicate logs.
        if self.logger.hasHandlers():
            for handler in list(self.logger.handlers):
                try:
                    handler.close()
                finally:
                    self.logger.removeHandler(handler)

        # Create a file handler to write to the specified log file (overwriting previous logs).
        file_handler = logging.FileHandler(self.log_dir, mode='w')
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler.
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger.
        self.logger.addHandler(file_handler)
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseEstimator':
        """Fit the estimator to data (must be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement fit().")
    
    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated density on the evaluation grid.
        
        Returns:
            Tuple of (grid_points, density_values)
        
        Raises:
            ValueError: If the estimator hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")

        if self.grid_points is None:
            raise ValueError("Evaluation grid points (self.grid_points) are not set. Call fit() first.")
        
        density = self.get_density_at_points(self.grid_points)
        
        return self.grid_points, density
    
    def get_results(self) -> dict:
        """Return estimator-specific results (must be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement get_results().")

    def _get_common_results(self) -> dict:
        """Build common result payload shared by all estimators (dict for compatibility)."""
        if not self.is_fitted or self.theta_hat is None:
            raise ValueError("Estimator must be fitted before getting results.")

        grid_points, density = self.get_density()
        
        # Count selected knots
        # When grid_points_hal_selected is available (e.g., from EM M-step with skip_coefficient_pruning=True),
        # use its length as the knot count to reflect the preserved knot structure.
        # Otherwise, fall back to counting non-zero HAL coefficients.
        hal_coeffs = self.theta_hat[self.basis_order + 1:]
        selected_knots = getattr(self, 'grid_points_hal_selected', None)
        if selected_knots is not None and len(selected_knots) > 0:
            selected_knots_count = len(selected_knots)
        else:
            selected_knots_count = np.sum(np.abs(hal_coeffs) > self.tol)
        if self.do_log:
            self.logger.info(f"Number of selected knots: {int(selected_knots_count)}")
        
        return {
            "fitted_theta_dict": self.fitted_theta_dict,
            "theta_hat": self.theta_hat,
            "data_points": getattr(self, '_grid_points_hal', None),
            "grid_points_hal_selected": getattr(self, 'grid_points_hal_selected', None),
            "n_selected_knots": selected_knots_count,
            "estimated_density": density,
            "grid_points": grid_points,
            "intercept": self.theta_hat[0],
            "hal_coeffs": hal_coeffs,
        }

    def get_common_results_model(self) -> CommonEstimatorResults:
        """Return standardized results as `CommonEstimatorResults` (Pydantic)."""
        payload = self._get_common_results()

        def to_list(x: Optional[np.ndarray]) -> Optional[list[float]]:
            if x is None:
                return None
            return x.tolist() if isinstance(x, np.ndarray) else list(x)

        return CommonEstimatorResults(
            fitted_theta_dict=payload.get("fitted_theta_dict"),
            theta_hat=to_list(payload["theta_hat"]),
            data_points=to_list(payload.get("data_points")),
            grid_points_hal_selected=to_list(payload.get("grid_points_hal_selected")),
            n_selected_knots=int(payload["n_selected_knots"]),
            estimated_density=to_list(payload["estimated_density"]),
            grid_points=to_list(payload["grid_points"]),
            intercept=float(payload["intercept"]),
            hal_coeffs=to_list(payload["hal_coeffs"]),
        )
    
    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Get the estimated density at specific points using the fitted model.
        
        Args:
            points: Array of points where to evaluate the density
            
        Returns:
            Array of density values at the given points
            
        Raises:
            ValueError: If the estimator hasn't been fitted yet
        """
        if not self.is_fitted or self.theta_hat is None or self._grid_points_hal is None or self.grid_midpoints is None or self.delta_j is None:
            raise ValueError("Estimator must be fitted before getting density. Call fit() first.")
        
        return BaseEstimator.calculate_density_at_points(
            points=points,
            theta_hat=self.theta_hat,
            basis_grid_points=self._grid_points_hal,
            basis_order=self.basis_order,
            norm_grid_midpoints=self.grid_midpoints,
            norm_delta_j=self.delta_j
        )
    
    @staticmethod
    def calculate_density_at_points(
        points: np.ndarray,
        theta_hat: np.ndarray,
        basis_grid_points: np.ndarray,
        basis_order: int,
        norm_grid_midpoints: Optional[np.ndarray] = None,
        norm_delta_j: Optional[np.ndarray] = None,
        n_norm_grid_points: int = 1000
    ) -> np.ndarray:
        """Compute density values at `points` for a HAL log-density φ(x)^T θ, normalized by ∫ exp(φ^Tθ)."""
        pts = np.asarray(points, dtype=float).ravel()

        if norm_grid_midpoints is None:
            norm_grid = np.linspace(0, 1 + 1 / n_norm_grid_points, n_norm_grid_points)
            norm_delta = np.diff(norm_grid)
            norm_grid_midpoints = norm_grid[:-1] + norm_delta / 2
            norm_delta_j = norm_delta

        # Pass the caller's delta to ensure consistent normalization
        density_mid, delta_used, max_log_f, norm_const = BaseEstimator.normalized_hal_density(
            grid=norm_grid_midpoints,
            theta_hat=theta_hat,
            basis_grid_points=basis_grid_points,
            basis_order=basis_order,
            delta=norm_delta_j,
        )

        if not np.isclose(np.sum(density_mid * delta_used), 1.0):
            LOGGER.warning("Density normalization failed to sum to 1 on the provided grid.")

        df_eval = pd.DataFrame({'W1': pts})
        b_eval, _ = create_basis_functions(df_eval, basis_grid_points, order=basis_order)
        log_f_eval = b_eval @ theta_hat
        shifted = np.clip(log_f_eval - max_log_f, -700, 700)
        density_eval = np.exp(shifted) / norm_const
        return density_eval

    @staticmethod
    def _integration_widths(grid: np.ndarray) -> np.ndarray:
        grid = np.asarray(grid, dtype=float).reshape(-1)
        if grid.size <= 1:
            return np.ones_like(grid, dtype=float)
        delta = np.empty_like(grid, dtype=float)
        diffs = np.diff(grid)
        delta[0] = diffs[0]
        delta[1:] = diffs
        return delta

    @staticmethod
    def normalized_hal_density(
        grid: np.ndarray,
        theta_hat: np.ndarray,
        basis_grid_points: np.ndarray,
        basis_order: int,
        include_intercept: bool = True,
        delta: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Evaluate exp(phi(x)^T theta) on `grid`, normalize it to integrate to 1, and
        return (density, delta, max_log, normalizer).

        Args:
            grid: Points at which to evaluate the density.
            theta_hat: Fitted HAL coefficients.
            basis_grid_points: Knot points for the HAL basis.
            basis_order: Order of the truncated power basis.
            include_intercept: Whether to include an intercept in the basis.
            delta: Integration widths for normalization. If None, computed from grid.

        Returns:
            Tuple of (density, delta, max_log, normalizer).
        """
        grid_arr = np.asarray(grid, dtype=float).reshape(-1)
        theta_arr = np.asarray(theta_hat, dtype=float).ravel()
        knots = np.asarray(basis_grid_points, dtype=float).ravel()

        df_grid = pd.DataFrame({'W1': grid_arr})
        basis_matrix, _ = create_basis_functions(
            df_grid,
            knots,
            order=basis_order,
            include_intercept=include_intercept,
        )
        if basis_matrix.shape[1] != theta_arr.size:
            raise ValueError("Theta length does not match basis design width.")

        log_density = basis_matrix @ theta_arr
        max_log = float(np.max(log_density))
        density = np.exp(np.clip(log_density - max_log, -700, 700))

        # Use provided delta or compute from grid
        if delta is not None:
            delta_arr = np.asarray(delta, dtype=float).ravel()
            if delta_arr.size != grid_arr.size:
                raise ValueError(
                    f"Delta length ({delta_arr.size}) must match grid length ({grid_arr.size})."
                )
        else:
            delta_arr = BaseEstimator._integration_widths(grid_arr)

        normalizer = float(np.sum(density * delta_arr))
        if normalizer <= 0:
            raise ValueError("Density normalization constant must be positive.")
        density /= normalizer
        return density, delta_arr, max_log, normalizer

    def _get_interpolated_log_density(self, points: np.ndarray) -> np.ndarray:
        """
        Get interpolated log-density values for specific points.
        """
        grid_points, estimated_density = self.get_density()

        density_interp = interp1d(
            grid_points, 
            estimated_density, 
            kind="linear",
            bounds_error=False, 
            fill_value=(estimated_density[0], estimated_density[-1])
        )

        interpolated_density = density_interp(points)
        return np.log(interpolated_density)
    
    def get_avg_log_likelihood_for_points(self, points: np.ndarray) -> float:
        """
        Get the average log-likelihood for specific points.
        """
        log_density = self._get_interpolated_log_density(points)
        return np.mean(log_density)
    
    def get_sum_log_likelihood_for_points(self, points: np.ndarray) -> float:
        """
        Get the sum of log-likelihood for specific points.
        """
        log_density = self._get_interpolated_log_density(points)
        return float(np.sum(log_density))
    
    def compute_bic(self, data: pd.DataFrame) -> float:
        """Compute BIC = -2 * log_likelihood + k * log(n), where k counts non-zero parameters."""
        if not self.is_fitted or self.theta_hat is None:
            raise ValueError("Estimator must be fitted before computing BIC.")
        
        n = len(data)
        points = np.asarray(data['W1'].values)
        sum_log_likelihood = self.get_sum_log_likelihood_for_points(points)
        k = 1 + np.sum(np.abs(self.theta_hat[1:]) > self.tol)
        
        return float(-2 * sum_log_likelihood + k * np.log(n))
    
    def _check_objective_explosion(self, obj: float, iteration: int) -> bool:
        """Guard against non-finite or exploding objectives during optimization."""
        if not np.isfinite(obj):
            LOGGER.warning(f"Objective became non-finite at iteration {iteration}; stopping optimization")
            return True
        elif obj > self.explosion_threshold or obj < -self.explosion_threshold:
            LOGGER.warning(f"Objective exploded to {obj:.2e} at iteration {iteration}; stopping optimization")
            return True
        return False
