import numpy as np
from scipy.stats import truncnorm
from typing import Union


class TruncatedNormal:
    """
    A class for working with truncated normal distributions.
    
    This class provides methods to generate samples and compute densities
    from a truncated normal distribution with specified parameters.
    """
    
    def __init__(
        self, 
        mean: float = 0.5, 
        std: float = 0.1, 
        lower: float = 0, 
        upper: float = 1
    ):
        """
        Initialize a truncated normal distribution.
        
        Args:
            mean: Mean of the underlying normal distribution
            std: Standard deviation of the underlying normal distribution
            lower: Lower bound for truncation
            upper: Upper bound for truncation
        """
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")
        
        self.mean = mean
        self.std = std
        self.lower = lower
        self.upper = upper
        
        # Pre-compute the standardized bounds for efficiency
        self._a = (lower - mean) / std
        self._b = (upper - mean) / std
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from the truncated normal distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of samples from truncated normal distribution
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        return truncnorm.rvs(
            self._a, self._b, 
            loc=self.mean, 
            scale=self.std, 
            size=n_samples
        )
    
    def compute_density(self, grid_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the true density of the truncated normal distribution.
        
        Args:
            grid_points: Points at which to evaluate the density
            
        Returns:
            Array of density values at grid_points
        """
        grid_points = np.asarray(grid_points)
        return truncnorm.pdf(
            grid_points, 
            self._a, self._b, 
            loc=self.mean, 
            scale=self.std
        )
    
    def compute_cdf(self, grid_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the cumulative distribution function.
        
        Args:
            grid_points: Points at which to evaluate the CDF
            
        Returns:
            Array of CDF values at grid_points
        """
        grid_points = np.asarray(grid_points)
        return truncnorm.cdf(
            grid_points, 
            self._a, self._b, 
            loc=self.mean, 
            scale=self.std
        )
    
    def get_stats(self) -> dict:
        """
        Get statistical properties of the truncated normal distribution.
        
        Returns:
            dictionary containing mean, variance, and other statistics
        """
        stats = truncnorm.stats(
            self._a, self._b, 
            loc=self.mean, 
            scale=self.std, 
            moments='mvsk'
        )
        
        return {
            'mean': float(stats[0]),
            'variance': float(stats[1]),
            'skewness': float(stats[2]),
            'kurtosis': float(stats[3]),
            'std': float(np.sqrt(stats[1])),
            'original_mean': self.mean,
            'original_std': self.std,
            'lower_bound': self.lower,
            'upper_bound': self.upper
        }
    
    def __repr__(self) -> str:
        return (f"TruncatedNormal(mean={self.mean}, std={self.std}, "
                f"lower={self.lower}, upper={self.upper})")

