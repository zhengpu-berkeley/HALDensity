"""
Utility functions for computing statistical functions from density estimates.

This module provides generic functions for computing survival functions, CDFs, 
and other statistics from density estimates using numerical integration.
"""

import numpy as np
from typing import Tuple


def generic_compute_survival_from_density(
    grid_points: np.ndarray,
    density: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute survival function from density estimate using cumulative integration.
    
    Parameters
    ----------
    grid_points : np.ndarray
        The grid points at which the density is evaluated.
    density : np.ndarray
        The estimated density values corresponding to the grid points.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (survival_probs, grid_points) where survival_probs are the 
        estimated survival probabilities and grid_points are the corresponding
        evaluation points.
    """
    # Ensure grid_points and density are numpy arrays
    grid_points = np.asarray(grid_points)
    density = np.asarray(density)
    
    if grid_points.ndim != 1 or density.ndim != 1:
        raise ValueError("grid_points and density must be 1-dimensional arrays.")
    if grid_points.shape[0] != density.shape[0]:
        raise ValueError("grid_points and density must have the same length.")
    
    # Compute delta_j as the difference between consecutive grid points
    delta_j = np.diff(grid_points, prepend=grid_points[0])
    if np.any(delta_j < 0):
        raise ValueError("grid_points must be strictly increasing.")
    
    # Compute CDF using cumulative sum: F(x) = ∫₀ˣ f(t)dt
    # Using trapezoidal rule approximation with midpoint grid
    cdf = np.cumsum(density * delta_j)
    
    # Ensure CDF is properly normalized (should sum to 1)
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    
    # Survival function: S(x) = 1 - F(x)
    survival = 1.0 - cdf
    
    return survival, grid_points


def generic_compute_cdf_from_density(
    grid_points: np.ndarray,
    density: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative distribution function from density estimate using cumulative integration.
    
    Parameters
    ----------
    grid_points : np.ndarray
        The grid points at which the density is evaluated.
    density : np.ndarray
        The estimated density values corresponding to the grid points.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (cdf_probs, grid_points) where cdf_probs are the 
        estimated CDF probabilities and grid_points are the corresponding
        evaluation points.
    """
    # Ensure grid_points and density are numpy arrays
    grid_points = np.asarray(grid_points)
    density = np.asarray(density)
    
    if grid_points.ndim != 1 or density.ndim != 1:
        raise ValueError("grid_points and density must be 1-dimensional arrays.")
    if grid_points.shape[0] != density.shape[0]:
        raise ValueError("grid_points and density must have the same length.")
    
    # Compute delta_j as the difference between consecutive grid points
    delta_j = np.diff(grid_points, prepend=grid_points[0])
    if np.any(delta_j < 0):
        raise ValueError("grid_points must be strictly increasing.")
    
    # Compute CDF using cumulative sum: F(x) = ∫₀ˣ f(t)dt
    # Using trapezoidal rule approximation with midpoint grid
    cdf = np.cumsum(density * delta_j)
    
    # Ensure CDF is properly normalized (should sum to 1)
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    
    return cdf, grid_points


def generic_compute_median_from_density(
    grid_points: np.ndarray,
    density: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute median and density at median from density estimate.
    
    Parameters
    ----------
    grid_points : np.ndarray
        The grid points at which the density is evaluated.
    density : np.ndarray
        The estimated density values corresponding to the grid points.
        
    Returns
    -------
    Tuple[float, float]
        Tuple of (median, density_at_median) where median is the estimated
        median value and density_at_median is the density evaluated at the median.
    """
    # Compute CDF first
    cdf, _ = generic_compute_cdf_from_density(grid_points, density)
    
    # Find median as the point where CDF = 0.5
    median_idx = np.argmin(np.abs(cdf - 0.5))
    median = grid_points[median_idx]
    density_at_median = density[median_idx]
    
    return median, density_at_median


def generic_compute_moment_from_density(
    grid_points: np.ndarray,
    density: np.ndarray,
    moment_order: int,
) -> float:
    """
    Compute k-th moment from density estimate using numerical integration.
    
    Parameters
    ----------
    grid_points : np.ndarray
        The grid points at which the density is evaluated.
    density : np.ndarray
        The estimated density values corresponding to the grid points.
    moment_order : int
        The order of the moment to compute (k in μ_k = E[X^k]).
        
    Returns
    -------
    float
        The estimated k-th moment.
    """
    # Ensure grid_points and density are numpy arrays
    grid_points = np.asarray(grid_points)
    density = np.asarray(density)
    
    if grid_points.ndim != 1 or density.ndim != 1:
        raise ValueError("grid_points and density must be 1-dimensional arrays.")
    if grid_points.shape[0] != density.shape[0]:
        raise ValueError("grid_points and density must have the same length.")
    
    # Compute delta_j as the difference between consecutive grid points
    delta_j = np.diff(grid_points, prepend=grid_points[0])
    if np.any(delta_j < 0):
        raise ValueError("grid_points must be strictly increasing.")
    
    # Compute moment: μ_k = ∫ x^k f(x) dx
    moment = np.sum((grid_points ** moment_order) * density * delta_j)
    
    return moment
