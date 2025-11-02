import numpy as np
import pandas as pd
from typing import Union, Any
from src.haldensity.targeting.base_target_learner import BaseTargetLearner
from src.haldensity.utils.density_computations import generic_compute_moment_from_density


def _compute_moment_from_density(targeted_fit: dict[str, Any], moment_order: int) -> float:
    """
    Compute k-th moment from density estimate using numerical integration.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from M-step containing density estimates and grid information.
    moment_order : int
        The order of the moment to compute (k in μ_k = E[X^k]).
        
    Returns
    -------
    float
        The estimated k-th moment.
    """
    
    density = targeted_fit['estimated_density']
    grid_points = targeted_fit['grid_midpoints']
    
    if density.ndim != 1 or grid_points.ndim != 1:
        raise ValueError("Both density and grid_points must be 1-dimensional arrays.")
    if density.shape[0] != grid_points.shape[0]:
        raise ValueError("Density and grid_points must have the same length.")
    
    # Compute moment from density using utility function
    moment_value = generic_compute_moment_from_density(grid_points, density, moment_order)
    
    return moment_value


class MomentsTargetLearner(BaseTargetLearner):
    """
    Target learner that uses moment-based targeting basis functions.
    
    This learner creates targeting basis functions based on powers of the data points.
    """
    
    def get_b_ik_targeting(self, uncensored_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for uncensored data based on moments.
        
        Parameters
        ----------
        uncensored_data : pd.DataFrame
            The uncensored data points.
        **kwargs
            Should contain 'x_moment' specifying the power/moment to use.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_data, 1) with values raised to x_moment power.
        """
        x_moment = kwargs.get('x_moment', 1)  # Default to first moment if not specified
        
        b_ik_targeting = uncensored_data['W1'].values ** x_moment
        if b_ik_targeting.ndim == 1:
            b_ik_targeting = b_ik_targeting[:, None]
        
        return b_ik_targeting
    
    def get_b_jk_targeting(self, data_grid: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for integration grid based on moments.
        
        Parameters
        ----------
        data_grid : pd.DataFrame
            The integration grid points.
        **kwargs
            Should contain 'x_moment' specifying the power/moment to use.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_grid, 1) with values raised to x_moment power.
        """
        x_moment = kwargs.get('x_moment', 1)  # Default to first moment if not specified
        
        b_jk_targeting = data_grid['W1'].values ** x_moment
        if b_jk_targeting.ndim == 1:
            b_jk_targeting = b_jk_targeting[:, None]
        
        return b_jk_targeting
    
    def get_b_jk_targeting_full(self, data_grid_full: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for fine evaluation grid based on moments.
        
        Parameters
        ----------
        data_grid_full : pd.DataFrame
            The fine evaluation grid points.
        **kwargs
            Should contain 'x_moment' specifying the power/moment to use.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_grid_full, 1) with values raised to x_moment power.
        """
        x_moment = kwargs.get('x_moment', 1)  # Default to first moment if not specified
        
        b_jk_targeting_full = data_grid_full['W1'].values ** x_moment
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
        Compute variance of k-th moment estimates using efficient influence curve (EIC).
        
        Based on the methodology where for k-th moment μ_k:
        D*(x) = x^k - μ_k
        
        The variance is estimated as the sample variance of the EIC divided by n.
        This follows the asymptotic theory: Var(√n(Ψ_n - Ψ)) ≈ Var(D*) / n
        
        Parameters
        ----------
        targeted_fit : dict[str, Any]
            Results from the M-step containing density estimates and grid information.
        uncensored_data : pd.DataFrame
            The uncensored data points used to construct the EIC.
        **kwargs
            Should contain 'x_moment' specifying the order of the moment.
            
        Returns
        -------
        np.ndarray
            Estimated variance for the k-th moment (single element array).
        """
        x_moment = kwargs.get('x_moment', 1)  # Default to first moment if not specified
        
        # Step 1: Compute k-th moment from the targeted density
        moment_value = _compute_moment_from_density(targeted_fit, x_moment)
        
        # Step 2: Create data values raised to the k-th power
        data_values = uncensored_data['W1'].values
        data_values = np.asarray(data_values)  # Ensure numpy array
        data_powers = data_values ** x_moment
        
        # Step 3: Construct EIC using the formula: D*(x) = x^k - μ_k
        EIC = data_powers - moment_value
        
        # Step 4: Compute variance as sample variance of EIC divided by n
        n_data = uncensored_data.shape[0]
        estimand_variance = np.var(EIC, ddof=1) / n_data
        
        # Return as single-element array for consistency
        return np.array([estimand_variance])


# Convenience function for backward compatibility
def moments_targeting_M_step(
    uncensored_augmented: pd.DataFrame,
    x_moment: int,
    grid_points_hal_selected: Union[list, np.ndarray],
    old_theta: Union[list, np.ndarray],
    norm_constraint: int = 20,
    basis_order: int = 0
) -> dict:
    """
    Convenience function that wraps the MomentsTargetLearner for backward compatibility.
    """
    learner = MomentsTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.run_m_step(
        uncensored_augmented=uncensored_augmented,
        grid_points_hal_selected=grid_points_hal_selected,
        old_theta=old_theta,
        x_moment=x_moment
    )


def moments_estimand_variance(
    targeted_fit: dict[str, Any],
    uncensored_data: pd.DataFrame,
    x_moment: int,
    norm_constraint: int = 20,
    basis_order: int = 0
) -> np.ndarray:
    """
    Convenience function that wraps the MomentsTargetLearner for estimating variance.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from the M-step containing density estimates and grid information.
    uncensored_data : pd.DataFrame
        The uncensored data points used to construct the EIC.
    x_moment : int
        The order of the moment to estimate variance for.
    norm_constraint : int, optional
        L1-norm constraint (default is 20).
    basis_order : int, optional
        Order of the basis functions (default is 0).
        
    Returns
    -------
    np.ndarray
        Estimated variance for the k-th moment (single element array).
    """
    learner = MomentsTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.get_estimand_variance(
        targeted_fit=targeted_fit,
        uncensored_data=uncensored_data,
        x_moment=x_moment
    )
