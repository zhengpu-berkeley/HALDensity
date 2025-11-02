import numpy as np
from scipy.stats import truncnorm
from typing import Union, Optional
from .truncated_normal import TruncatedNormal


class TruncatedGMM:
    """
    A class for working with truncated Gaussian Mixture Models (GMM).
    
    This class provides methods to generate samples and compute densities
    from a mixture of truncated normal distributions with specified parameters.
    """
    
    def __init__(
        self, 
        components: list[dict[str, float]], 
        weights: Optional[list[float]] = None
    ):
        """
        Initialize a truncated Gaussian Mixture Model.
        
        Args:
            components: list of dictionaries, each containing parameters for a 
                       truncated normal component. Each dict should have keys:
                       'mean', 'std', 'lower', 'upper'
            weights: list of mixture weights (must sum to 1). If None, equal weights
                    are used for all components.
        
        Example:
            components = [
                {'mean': 0.35, 'std': 0.1, 'lower': 0.0, 'upper': 1.0},
                {'mean': 0.75, 'std': 0.05, 'lower': 0.0, 'upper': 1.0}
            ]
            gmm = TruncatedGMM(components, weights=[0.5, 0.5])
        """
        if not components:
            raise ValueError("At least one component must be provided")
        
        self.n_components = len(components)
        self.components = components
        
        # Validate and set weights
        if weights is None:
            self.weights = np.array([1.0 / self.n_components] * self.n_components)
        else:
            weights = np.asarray(weights, dtype=float)
            if len(weights) != self.n_components:
                raise ValueError("Number of weights must match number of components")
            weights = np.array(weights)
            if not np.isclose(weights.sum(), 1.0):
                weights /= weights.sum()  # Normalize weights
                print("Weights normalized to sum to 1")
            if np.any(weights < 0):
                raise ValueError("All weights must be non-negative")
            self.weights = weights
        
        # Create TruncatedNormal objects for each component
        self._tn_components = []
        for comp in components:
            # Validate component parameters
            required_keys = {'mean', 'std', 'lower', 'upper'}
            if not required_keys.issubset(comp.keys()):
                raise ValueError(f"Each component must have keys: {required_keys}")
            
            tn = TruncatedNormal(
                mean=comp['mean'],
                std=comp['std'],
                lower=comp['lower'],
                upper=comp['upper']
            )
            self._tn_components.append(tn)
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from the truncated GMM.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of samples from the truncated GMM
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Determine how many samples to draw from each component
        component_counts = np.random.multinomial(n_samples, self.weights)
        
        # Generate samples from each component
        samples = []
        for i, count in enumerate(component_counts):
            if count > 0:
                component_samples = self._tn_components[i].generate_samples(count)
                samples.append(component_samples)
        
        # Combine and shuffle all samples
        if samples:
            all_samples = np.concatenate(samples)
            np.random.shuffle(all_samples)
            return all_samples
        else:
            return np.array([])
    
    def compute_density(self, grid_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the density of the truncated GMM.
        
        Args:
            grid_points: Points at which to evaluate the density
            
        Returns:
            Array of density values at grid_points
        """
        grid_points = np.asarray(grid_points)
        
        # Compute weighted sum of component densities
        total_density = np.zeros_like(grid_points, dtype=float)
        for i, tn_comp in enumerate(self._tn_components):
            tn_comp: TruncatedNormal = tn_comp  # Type hint for clarity 
            component_density = tn_comp.compute_density(grid_points)
            total_density += self.weights[i] * component_density
        
        return total_density
    
    def compute_cdf(self, grid_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the cumulative distribution function of the truncated GMM.
        
        Args:
            grid_points: Points at which to evaluate the CDF
            
        Returns:
            Array of CDF values at grid_points
        """
        grid_points = np.asarray(grid_points)
        
        # Compute weighted sum of component CDFs
        total_cdf = np.zeros_like(grid_points, dtype=float)
        for i, tn_comp in enumerate(self._tn_components):
            component_cdf = tn_comp.compute_cdf(grid_points)
            total_cdf += self.weights[i] * component_cdf
        
        return total_cdf
    
    def get_stats(self) -> dict:
        """
        Get statistical properties of the truncated GMM.
        
        Returns:
            dictionary containing mean, variance, and component information
        """
        # Compute mixture mean and variance
        component_stats = [comp.get_stats() for comp in self._tn_components]
        
        # Mixture mean: E[X] = Σ w_i * E[X_i]
        mixture_mean = sum(self.weights[i] * stats['mean'] 
                          for i, stats in enumerate(component_stats))
        
        # Mixture variance: Var[X] = Σ w_i * (Var[X_i] + (E[X_i] - E[X])^2)
        mixture_variance = sum(
            self.weights[i] * (stats['variance'] + (stats['mean'] - mixture_mean)**2)
            for i, stats in enumerate(component_stats)
        )
        
        return {
            'mixture_mean': float(mixture_mean),
            'mixture_variance': float(mixture_variance),
            'mixture_std': float(np.sqrt(mixture_variance)),
            'n_components': self.n_components,
            'weights': self.weights.tolist(),
            'components': self.components,
            'component_stats': component_stats
        }
    
    def get_component_weights(self) -> np.ndarray:
        """
        Get the mixture weights.
        
        Returns:
            Array of mixture weights
        """
        return self.weights.copy()
    
    def get_component_parameters(self) -> list[dict[str, float]]:
        """
        Get the parameters for all components.
        
        Returns:
            list of component parameter dictionaries
        """
        return [comp.copy() for comp in self.components]
    
    def __repr__(self) -> str:
        comp_str = ", ".join([
            f"N(μ={comp['mean']}, σ={comp['std']}, [{comp['lower']}, {comp['upper']}])"
            for comp in self.components
        ])
        weight_str = ", ".join([f"{w:.3f}" for w in self.weights])
        return f"TruncatedGMM(components=[{comp_str}], weights=[{weight_str}])"
