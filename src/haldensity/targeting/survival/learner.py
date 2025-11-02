import numpy as np
import pandas as pd
from typing import Union, Any, Tuple
from haldensity.targeting.base_target_learner import BaseTargetLearner
from haldensity.targeting.survival.targeting_basis import create_targeting_basis_functions
from haldensity.utils.basis import create_basis_functions
from haldensity.utils.density_computations import generic_compute_survival_from_density


def _compute_survival_from_density(targeted_fit: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute survival function from density estimate using cumulative integration.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from M-step containing density estimates and grid information.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (survival_probs, grid_points) where survival_probs are the 
        estimated survival probabilities and grid_points are the corresponding
        evaluation points.
    """
    
    density = targeted_fit['estimated_density']
    grid_points = targeted_fit['grid_midpoints']
    
    if density.ndim != 1 or grid_points.ndim != 1:
        raise ValueError("Both density and grid_points must be 1-dimensional arrays.")
    if density.shape[0] != grid_points.shape[0]:
        raise ValueError("Density and grid_points must have the same length.")
    
    # Compute survival function from density using utility function
    survival, grid_surv = generic_compute_survival_from_density(grid_points, density)
    
    if survival.ndim != 1 or grid_surv.ndim != 1:
        raise ValueError("Computed survival and grid points must be 1-dimensional arrays.") 
    if survival.shape[0] != grid_surv.shape[0]:
        raise ValueError("Computed survival and grid points must have the same length.")
    
    return survival, grid_surv


class SurvivalTargetLearner(BaseTargetLearner):
    """
    Target learner that uses survival-based targeting basis functions.
    
    This learner creates targeting basis functions based on cumulative indicators
    using greater-than comparisons with grid points (survival functions).
    """
    
    def get_b_ik_targeting(self, uncensored_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for uncensored data based on survival indicators.
        
        Parameters
        ----------
        uncensored_data : pd.DataFrame
            The uncensored data points.
        **kwargs
            Should contain 'targeting_points' specifying the grid points for indicators.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_data, n_targeting_points) with survival indicators.
        """
        targeting_points = kwargs['targeting_points']
        
        basis_tensor_targeting = create_targeting_basis_functions(uncensored_data, targeting_points)
        b_ik_targeting = basis_tensor_targeting  # numpy array
        if b_ik_targeting.ndim == 1:
            b_ik_targeting = b_ik_targeting[:, None]
        
        return b_ik_targeting
    
    def get_b_jk_targeting(self, data_grid: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for integration grid based on survival indicators.
        
        Parameters
        ----------
        data_grid : pd.DataFrame
            The integration grid points.
        **kwargs
            Should contain 'targeting_points' specifying the grid points for indicators.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_grid, n_targeting_points) with survival indicators.
        """
        targeting_points = kwargs['targeting_points']
        
        basis_grid_tensor_targeting = create_targeting_basis_functions(data_grid, targeting_points)
        b_jk_targeting = basis_grid_tensor_targeting  # numpy array
        if b_jk_targeting.ndim == 1:
            b_jk_targeting = b_jk_targeting[:, None]
        
        return b_jk_targeting
    
    def get_b_jk_targeting_full(self, data_grid_full: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Create targeting basis design matrix for fine evaluation grid based on survival indicators.
        
        Parameters
        ----------
        data_grid_full : pd.DataFrame
            The fine evaluation grid points.
        **kwargs
            Should contain 'targeting_points' specifying the grid points for indicators.
            
        Returns
        -------
        np.ndarray
            Targeting basis matrix (n_grid_full, n_targeting_points) with survival indicators.
        """
        targeting_points = kwargs['targeting_points']
        
        basis_grid_tensor_targeting_full = create_targeting_basis_functions(data_grid_full, targeting_points)
        b_jk_targeting_full = basis_grid_tensor_targeting_full  # numpy array
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
        Compute variance of survival function estimates using efficient influence curve (EIC).
        
        Based on the methodology where for survival function at point t:
        D*(x) = I(x > t) - S(t)
        
        The variance is estimated as the sample variance of the EIC divided by n.
        This follows the asymptotic theory: Var(√n(Ψ_n - Ψ)) ≈ Var(D*) / n
        
        Note: The research code in estimand_variance_research_code.md uses np.std(EIC, ddof=1)
        which gives the standard error of individual observations. Our implementation 
        computes Var(EIC)/n which is the variance of the estimand. The relationship is:
        research_se = √n * √(our_variance), which matches asymptotic theory.
        
        Parameters
        ----------
        targeted_fit : dict[str, Any]
            Results from the M-step containing density estimates and grid information.
        uncensored_data : pd.DataFrame
            The uncensored data points used to construct the EIC.
        **kwargs
            Should contain 'targeting_points' specifying the grid points for variance estimation.
            
        Returns
        -------
        np.ndarray
            Estimated variances for each targeting point, shape (n_targeting_points,).
        """
        targeting_points = kwargs['targeting_points']
        
        # Step 1: Compute survival function from the targeted density
        survival, grid_surv = _compute_survival_from_density(targeted_fit)
        
        # Step 2: Interpolate survival probabilities at targeting points
        survival_targeting = np.interp(targeting_points, grid_surv, survival, left=1.0, right=0.0)
        
        # Step 3: Build the targeting basis for the data (I(x > t) terms)
        # Note: We use order=0 to get step function indicators as in research code
        basis_tensor, _ = create_basis_functions(uncensored_data, targeting_points, order=0, include_intercept=False)
        b_ik_targeting = basis_tensor  # numpy array
        
        # Ensure proper dimensionality
        if b_ik_targeting.ndim == 1:
            b_ik_targeting = b_ik_targeting[:, None]
        
        # Step 4: Construct EIC matrix
        # EIC = I(x > t) - S(t) for each targeting point t
        EIC_matrix = b_ik_targeting - survival_targeting[None, :]  # Broadcasting: (n_data, n_targeting_points)
        
        # Step 5: Compute variance as sample variance of EIC divided by n
        n_data = uncensored_data.shape[0]
        estimand_variance = np.var(EIC_matrix, axis=0, ddof=1) / n_data
        
        return estimand_variance


# Convenience function for backward compatibility
def survival_targeting_M_step(
    targeting_points: Union[list, np.ndarray],
    uncensored_augmented: pd.DataFrame,
    grid_points_hal_selected: Union[list, np.ndarray],
    old_theta: Union[list, np.ndarray],
    norm_constraint: int = 20,
    basis_order: int = 0
) -> dict:
    """
    Convenience function that wraps the SurvivalTargetLearner for backward compatibility.
    """
    learner = SurvivalTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.run_m_step(
        uncensored_augmented=uncensored_augmented,
        grid_points_hal_selected=grid_points_hal_selected,
        old_theta=old_theta,
        targeting_points=targeting_points
    )

def survival_estimand_variance(
    targeted_fit: dict[str, Any],
    uncensored_data: pd.DataFrame,
    targeting_points: Union[list, np.ndarray],
    norm_constraint: int = 20,
    basis_order: int = 0
) -> np.ndarray:
    """
    Convenience function that wraps the SurvivalTargetLearner for estimating variance.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from the M-step containing density estimates and grid information.
    uncensored_data : pd.DataFrame
        The uncensored data points used to construct the EIC.
    targeting_points : Union[list, np.ndarray]
        The grid points for which to estimate variance.
    norm_constraint : int, optional
        L1-norm constraint (default is 20).
    basis_order : int, optional
        Order of the basis functions (default is 0).
        
    Returns
    -------
    np.ndarray
        Estimated variances for each targeting point.
    """
    learner = SurvivalTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.get_estimand_variance(
        targeted_fit=targeted_fit,
        uncensored_data=uncensored_data,
        targeting_points=targeting_points
    )