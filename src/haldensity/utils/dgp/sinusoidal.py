import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from typing import Union


class Sinusoidal:
    """
    A class for working with sinusoidal-based distributions on [0,1].
    
    This class provides methods to generate samples and compute densities
    from a sinusoidal distribution with PDF proportional to sin(π*x) + 1.1.
    """
    
    def __init__(self):
        """
        Initialize a sinusoidal distribution.
        
        The PDF is defined as f(x) ∝ sin(π*x) + 1.1 on [0,1].
        The constant 1.1 ensures the PDF is strictly positive.
        """
        # Pre-compute the normalization constant for efficiency
        self._norm_const, _ = quad(lambda t: np.sin(np.pi * t) + 1.1, 0, 1)
        
        # Pre-compute inverse CDF via interpolation for efficient sampling
        self._setup_inverse_cdf()
    
    def _setup_inverse_cdf(self):
        """Setup the inverse CDF function using interpolation."""
        grid = np.linspace(0, 1, 2000)
        cdf_vals = np.array([self._compute_single_cdf(xi) for xi in grid])
        self._inv_cdf = interp1d(
            cdf_vals, grid, 
            kind='linear', 
            fill_value=(0, 1), 
            bounds_error=False
        )
    
    def _compute_single_cdf(self, x: float) -> float:
        """Compute CDF at a single point by integrating the PDF."""
        if x <= 0:
            return 0.0
        elif x >= 1:
            return 1.0
        else:
            integral, _ = quad(lambda t: self._unnormalized_pdf(t), 0, x)
            return integral / self._norm_const
    
    def _unnormalized_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the unnormalized PDF: sin(π*x) + 1.1"""
        return np.sin(np.pi * x) + 1.1
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from the sinusoidal distribution using inverse transform sampling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of samples from the sinusoidal distribution
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate uniform random variables
        u = np.random.uniform(0, 1, size=n_samples)
        
        # Apply inverse CDF transformation
        return self._inv_cdf(u)
    
    def compute_density(self, grid_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the density of the sinusoidal distribution.
        
        Args:
            grid_points: Points at which to evaluate the density
            
        Returns:
            Array of density values at grid_points
        """
        grid_points = np.asarray(grid_points)
        
        # Return 0 for points outside [0,1]
        density = np.zeros_like(grid_points, dtype=float)
        mask = (grid_points >= 0) & (grid_points <= 1)
        
        # Compute normalized PDF for points in [0,1]
        density[mask] = self._unnormalized_pdf(grid_points[mask]) / self._norm_const
        
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
        
        for i, x in enumerate(grid_points):
            cdf_values[i] = self._compute_single_cdf(x)
        
        return cdf_values
    
    def get_stats(self) -> dict:
        """
        Get statistical properties of the sinusoidal distribution.
        
        Returns:
            dictionary containing mean, variance, and other statistics
        """
        # Compute mean: E[X] = ∫ x * f(x) dx
        mean, _ = quad(lambda x: x * self._unnormalized_pdf(x), 0, 1)
        mean /= self._norm_const
        
        # Compute second moment: E[X²] = ∫ x² * f(x) dx
        second_moment, _ = quad(lambda x: x**2 * self._unnormalized_pdf(x), 0, 1)
        second_moment /= self._norm_const
        
        # Variance = E[X²] - (E[X])²
        variance = second_moment - mean**2
        
        return {
            'mean': float(mean),
            'variance': float(variance),
            'std': float(np.sqrt(variance)),
            'second_moment': float(second_moment),
            'norm_constant': float(self._norm_const),
            'support': [0, 1]
        }
    
    def __repr__(self) -> str:
        return "Sinusoidal(PDF ∝ sin(π*x) + 1.1 on [0,1])"