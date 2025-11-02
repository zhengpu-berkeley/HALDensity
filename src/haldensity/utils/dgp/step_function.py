import numpy as np
from typing import Union


class StepFunction:
    """
    A class for working with step function distributions on [0,1].
    
    This class provides methods to generate samples and compute densities
    from a step function distribution with two constant levels.
    """
    
    def __init__(self, level1: float = 1.0, level2: float = 0.5, breakpoint: float = 0.5):
        """
        Initialize a step function distribution.
        
        Args:
            level1: PDF value for x < breakpoint
            level2: PDF value for x >= breakpoint  
            breakpoint: Point where the function changes value
            
        The PDF is:
        f(x) = level1 if x < breakpoint, level2 if x >= breakpoint
        """
        if level1 <= 0 or level2 <= 0:
            raise ValueError("Both levels must be positive")
        if breakpoint <= 0 or breakpoint >= 1:
            raise ValueError("Breakpoint must be in (0, 1)")
        
        self.level1 = level1
        self.level2 = level2
        self.breakpoint = breakpoint
        
        # Pre-compute normalization constant
        # Area = level1 * breakpoint + level2 * (1 - breakpoint)
        self._norm_const = level1 * breakpoint + level2 * (1 - breakpoint)
        
        # Normalized levels
        self._norm_level1 = level1 / self._norm_const
        self._norm_level2 = level2 / self._norm_const
        
        # CDF value at breakpoint
        self._cdf_at_breakpoint = (level1 * breakpoint) / self._norm_const
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from the step function distribution using inverse transform sampling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of samples from the step function distribution
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate uniform random variables
        u = np.random.uniform(0, 1, size=n_samples)
        
        # Apply inverse CDF transformation
        # If u < cdf_at_breakpoint: x = u / norm_level1
        # If u >= cdf_at_breakpoint: x = breakpoint + (u - cdf_at_breakpoint) / norm_level2
        samples = np.where(
            u < self._cdf_at_breakpoint,
            u / self._norm_level1,                                              # First piece
            self.breakpoint + (u - self._cdf_at_breakpoint) / self._norm_level2  # Second piece
        )
        
        return samples
    
    def compute_density(self, grid_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the density of the step function distribution.
        
        Args:
            grid_points: Points at which to evaluate the density
            
        Returns:
            Array of density values at grid_points
        """
        grid_points = np.asarray(grid_points)
        
        # Initialize density array
        density = np.zeros_like(grid_points, dtype=float)
        
        # Set density values based on position relative to breakpoint
        mask_in_support = (grid_points >= 0) & (grid_points <= 1)
        mask_first_level = mask_in_support & (grid_points < self.breakpoint)
        mask_second_level = mask_in_support & (grid_points >= self.breakpoint)
        
        density[mask_first_level] = self._norm_level1
        density[mask_second_level] = self._norm_level2
        
        return density
    
    def compute_cdf(self, grid_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the cumulative distribution function.
        
        Args:
            grid_points: Points at which to evaluate the CDF
            
        Returns:
            Array of CDF values at grid_points
        """
        grid_points = np.asarray(grid_points)
        cdf_values = np.zeros_like(grid_points, dtype=float)
        
        # CDF computation based on piecewise definition
        mask_negative = grid_points < 0
        mask_first_piece = (grid_points >= 0) & (grid_points < self.breakpoint)
        mask_second_piece = (grid_points >= self.breakpoint) & (grid_points <= 1)
        mask_beyond_one = grid_points > 1
        
        # CDF = 0 for x < 0
        cdf_values[mask_negative] = 0.0
        
        # CDF = (level1 * x) / norm_const for 0 <= x < breakpoint
        cdf_values[mask_first_piece] = (self.level1 * grid_points[mask_first_piece]) / self._norm_const
        
        # CDF = (level1 * breakpoint + level2 * (x - breakpoint)) / norm_const for breakpoint <= x <= 1
        cdf_values[mask_second_piece] = (
            self.level1 * self.breakpoint + 
            self.level2 * (grid_points[mask_second_piece] - self.breakpoint)
        ) / self._norm_const
        
        # CDF = 1 for x > 1
        cdf_values[mask_beyond_one] = 1.0
        
        return cdf_values
    
    def get_stats(self) -> dict:
        """
        Get statistical properties of the step function distribution.
        
        Returns:
            dictionary containing mean, variance, and other statistics
        """
        # Mean: E[X] = ∫ x * f(x) dx
        # = (level1/norm_const) * ∫₀^breakpoint x dx + (level2/norm_const) * ∫_breakpoint^1 x dx
        # = (level1/norm_const) * breakpoint²/2 + (level2/norm_const) * (1 - breakpoint²)/2
        mean = (
            self._norm_level1 * (self.breakpoint**2) / 2 + 
            self._norm_level2 * (1 - self.breakpoint**2) / 2
        )
        
        # Second moment: E[X²] = ∫ x² * f(x) dx
        # = (level1/norm_const) * ∫₀^breakpoint x² dx + (level2/norm_const) * ∫_breakpoint^1 x² dx
        # = (level1/norm_const) * breakpoint³/3 + (level2/norm_const) * (1 - breakpoint³)/3
        second_moment = (
            self._norm_level1 * (self.breakpoint**3) / 3 + 
            self._norm_level2 * (1 - self.breakpoint**3) / 3
        )
        
        # Variance = E[X²] - (E[X])²
        variance = second_moment - mean**2
        
        return {
            'mean': float(mean),
            'variance': float(variance),
            'std': float(np.sqrt(variance)),
            'second_moment': float(second_moment),
            'level1': self.level1,
            'level2': self.level2,
            'normalized_level1': self._norm_level1,
            'normalized_level2': self._norm_level2,
            'breakpoint': self.breakpoint,
            'norm_constant': self._norm_const,
            'support': [0, 1]
        }
    
    def __repr__(self) -> str:
        return (f"StepFunction(level1={self.level1}, level2={self.level2}, "
                f"breakpoint={self.breakpoint})")