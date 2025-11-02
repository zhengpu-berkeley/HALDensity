import numpy as np
import pandas as pd
from typing import Union, Any, Tuple
from src.haldensity.targeting.base_target_learner import BaseTargetLearner
from src.haldensity.estimation.base_estimator import BaseEstimator
from src.haldensity.utils.density_computations import generic_compute_median_from_density

N_POINTS = 2001
DENSITY_EVAL_GRID = np.linspace(0, 1, N_POINTS)


def _compute_median_from_density(targeted_fit: dict[str, Any]) -> Tuple[float, float]:
    """
    Compute median from density estimate using cumulative integration.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from M-step containing density estimates and grid information.
        
    Returns
    -------
    Tuple[float, float]
        Tuple of (median, density_at_median).
    """
    
    density = targeted_fit['estimated_density']
    grid_points = targeted_fit['grid_midpoints']
    
    if density.ndim != 1 or grid_points.ndim != 1:
        raise ValueError("Both density and grid_points must be 1-dimensional arrays.")
    if density.shape[0] != grid_points.shape[0]:
        raise ValueError("Density and grid_points must have the same length.")
    
    # Compute median from density using utility function
    median_value, density_at_median = generic_compute_median_from_density(grid_points, density)
    
    return median_value, density_at_median


class MedianTargetLearner(BaseTargetLearner):
    """
    Target learner that uses median-based targeting basis functions.
    
    This learner creates targeting basis functions that are indicators for 
    whether data points are less than the estimated median.
    """
    
    def get_b_ik_targeting(self, uncensored_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for uncensored data based on median.
        
        Parameters
        ----------
        uncensored_data : pd.DataFrame
            The uncensored data points.
        **kwargs
            Should contain 'old_theta' and 'grid_points_hal_selected' for median calculation.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_data, 1) with indicators for values < median.
        """
        # Extract required parameters
        old_theta = kwargs['_old_theta']
        grid_points_hal_selected = kwargs['_grid_points_hal_selected']
        
        # Calculate density and median
        density = BaseEstimator.calculate_density_at_points(
            points=DENSITY_EVAL_GRID,
            theta_hat=old_theta,
            basis_grid_points=grid_points_hal_selected,
            basis_order=self.basis_order
        )
        
        # Calculate CDF and find median
        cdf = np.cumsum(density * np.diff(DENSITY_EVAL_GRID, prepend=DENSITY_EVAL_GRID[0])) 
        median_from_data_density = DENSITY_EVAL_GRID[np.searchsorted(cdf / cdf[-1], 0.5)]
        
        # Create indicator for values less than median
        b_ik_targeting = uncensored_data['W1'].values < median_from_data_density
        if b_ik_targeting.ndim == 1:
            b_ik_targeting = b_ik_targeting[:, None]
        
        return b_ik_targeting
    
    def get_b_jk_targeting(self, data_grid: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for integration grid based on median.
        
        Parameters
        ----------
        data_grid : pd.DataFrame
            The integration grid points.
        **kwargs
            Should contain 'old_theta' and 'grid_points_hal_selected' for median calculation.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_grid, 1) with indicators for values < median.
        """
        # Extract required parameters
        old_theta = kwargs['_old_theta']
        grid_points_hal_selected = kwargs['_grid_points_hal_selected']
        
        # Calculate density and median (same as in get_b_ik_targeting)
        density = BaseEstimator.calculate_density_at_points(
            points=DENSITY_EVAL_GRID,
            theta_hat=old_theta,
            basis_grid_points=grid_points_hal_selected,
            basis_order=self.basis_order
        )
        
        # Calculate CDF and find median
        cdf = np.cumsum(density * np.diff(DENSITY_EVAL_GRID, prepend=DENSITY_EVAL_GRID[0])) 
        median_from_data_density = DENSITY_EVAL_GRID[np.searchsorted(cdf / cdf[-1], 0.5)]
        
        # Create indicator for values less than median
        b_jk_targeting = data_grid['W1'].values < median_from_data_density
        if b_jk_targeting.ndim == 1:
            b_jk_targeting = b_jk_targeting[:, None]
        
        return b_jk_targeting
    
    def get_b_jk_targeting_full(self, data_grid_full: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for fine evaluation grid based on median.
        
        Parameters
        ----------
        data_grid_full : pd.DataFrame
            The fine evaluation grid points.
        **kwargs
            Should contain 'old_theta' and 'grid_points_hal_selected' for median calculation.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_grid_full, 1) with indicators for values < median.
        """
        # Extract required parameters
        old_theta = kwargs['_old_theta']
        grid_points_hal_selected = kwargs['_grid_points_hal_selected']
        
        # Calculate density and median (same as in get_b_ik_targeting)
        density = BaseEstimator.calculate_density_at_points(
            points=DENSITY_EVAL_GRID,
            theta_hat=old_theta,
            basis_grid_points=grid_points_hal_selected,
            basis_order=self.basis_order
        )
        
        # Calculate CDF and find median
        cdf = np.cumsum(density * np.diff(DENSITY_EVAL_GRID, prepend=DENSITY_EVAL_GRID[0])) 
        median_from_data_density = DENSITY_EVAL_GRID[np.searchsorted(cdf / cdf[-1], 0.5)]
        
        # Create indicator for values less than median
        b_jk_targeting_full = data_grid_full['W1'].values < median_from_data_density
        if b_jk_targeting_full.ndim == 1:
            b_jk_targeting_full = b_jk_targeting_full[:, None]
        
        return b_jk_targeting_full
    
    def get_estimand_variance(
        self,
        targeted_fit: dict[str, Any],
        uncensored_data: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Compute variance of median estimate using efficient influence curve (EIC).
        
        Based on the methodology where for median m:
        D*(x) = (0.5 - I(x < m)) / f(m)
        
        The variance is estimated as the sample variance of the EIC divided by n.
        This follows the asymptotic theory: Var(√n(Ψ_n - Ψ)) ≈ Var(D*) / n
        
        Parameters
        ----------
        targeted_fit : dict[str, Any]
            Results from the M-step containing density estimates and grid information.
        uncensored_data : pd.DataFrame
            The uncensored data points used to construct the EIC.
        **kwargs
            Additional parameters (not used for median variance estimation).
            
        Returns
        -------
        np.ndarray
            Estimated variance for the median (single element array).
        """
        # Step 1: Compute median and density at median from the targeted density
        median_value, density_at_median = _compute_median_from_density(targeted_fit)
        
        # Step 2: Check for very small density values to avoid numerical issues
        if density_at_median < 1e-8:
            # Use a small positive value to avoid division by zero
            density_at_median = 1e-8
        
        # Step 3: Create indicator for values less than median
        data_values = uncensored_data['W1'].values
        data_values = np.asarray(data_values)  # Ensure numpy array
        indicator_less_than_median = (data_values < median_value).astype(float)
        
        # Step 4: Construct EIC using the formula: D*(x) = (0.5 - I(x < m)) / f(m)
        EIC = (0.5 - indicator_less_than_median) / density_at_median
        
        # Step 5: Compute variance as sample variance of EIC divided by n
        n_data = uncensored_data.shape[0]
        estimand_variance = np.var(EIC, ddof=1) / n_data
        
        # Return as single-element array for consistency
        return np.array([estimand_variance])


# Convenience function for backward compatibility
def median_targeting_M_step(
    uncensored_augmented: pd.DataFrame,
    grid_points_hal_selected: Union[list, np.ndarray],
    old_theta: Union[list, np.ndarray],
    norm_constraint: int = 20,
    basis_order: int = 0
) -> dict:
    """
    Convenience function that wraps the MedianTargetLearner for backward compatibility.
    """
    learner = MedianTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.run_m_step(
        uncensored_augmented=uncensored_augmented,
        grid_points_hal_selected=grid_points_hal_selected,
        old_theta=old_theta,
        _old_theta=old_theta, # serves as kwarg 
        _grid_points_hal_selected=grid_points_hal_selected # serves as kwarg
    )


def median_estimand_variance(
    targeted_fit: dict[str, Any],
    uncensored_data: pd.DataFrame,
    norm_constraint: int = 20,
    basis_order: int = 0
) -> np.ndarray:
    """
    Convenience function that wraps the MedianTargetLearner for estimating variance.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from the M-step containing density estimates and grid information.
    uncensored_data : pd.DataFrame
        The uncensored data points used to construct the EIC.
    norm_constraint : int, optional
        L1-norm constraint (default is 20).
    basis_order : int, optional
        Order of the basis functions (default is 0).
        
    Returns
    -------
    np.ndarray
        Estimated variance for the median (single element array).
    """
    learner = MedianTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.get_estimand_variance(
        targeted_fit=targeted_fit,
        uncensored_data=uncensored_data
    )
