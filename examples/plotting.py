import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

def plot_density(
    grid_points: np.ndarray,
    estimated_density: Optional[np.ndarray] = None,
    true_density: Optional[np.ndarray] = None,
    title: str = "Density Estimation",
    method_label: str = "Estimated Density",
    figsize: Tuple[int, int] = (8, 5),
    show: bool = True,
    # Optional histogram overlay of sample data
    hist_data: Optional[np.ndarray] = None,
    hist_bins: int = 100,
    hist_alpha: float = 0.5,
    hist_label: str = "Sampled Data",
    hist_color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Plot estimated density and optionally compare with true density.
    
    Args:
        grid_points: x-axis points for plotting
        estimated_density: Estimated density values
        true_density: True density values (optional)
        title: Plot title
        method_label: Label for the estimated density
        figsize: Figure size Tuple
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Fix: Check that at least one density is provided
    if estimated_density is None and true_density is None:
        raise ValueError("At least one of estimated_density or true_density must be provided.")
    
    # Add input validation for array shapes
    if estimated_density is not None and len(estimated_density) != len(grid_points):
        raise ValueError("estimated_density must have the same length as grid_points")
    if true_density is not None and len(true_density) != len(grid_points):
        raise ValueError("true_density must have the same length as grid_points")
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if estimated_density is not None:
        ax.plot(grid_points, estimated_density, label=method_label, linewidth=2)
    if true_density is not None:
        ax.plot(grid_points, true_density, label="True Density", linewidth=2, linestyle='--')

    # Optional histogram overlay
    if hist_data is not None:
        ax.hist(
            hist_data,
            bins=hist_bins,
            density=True,
            alpha=hist_alpha,
            label=hist_label,
            color=hist_color,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        # In notebooks, prefer plt.show() (not fig.show()) to avoid backend warnings.
        # Avoid returning the figure to prevent duplicate rendering by the display hook.
        plt.show()
        if ax is None:
            plt.close(fig)
        return None

    return fig


def compare_densities(
    grid_points: np.ndarray,
    methods_estimated_densities: dict[str, np.ndarray],
    true_density: Optional[np.ndarray] = None,
    title: str = "Density Estimation",
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True
) -> plt.Figure:
    """
    Compare multiple estimated density with true density.
    Args:
        grid_points: x-axis points for plotting
        methods_estimated_densities: dictionary of method names and their estimated densities
        true_density: True density values (optional)
        title: Plot title
        figsize: Figure size Tuple
        show: Whether to display the plot
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)

    if not methods_estimated_densities and true_density is None:
        raise ValueError("At least one of methods_estimated_densities or true_density must be provided.")

    if true_density is not None and len(true_density) != len(grid_points):
        raise ValueError("true_density must have the same length as grid_points")

    for method, density in methods_estimated_densities.items():
        if len(density) != len(grid_points):
            raise ValueError(f"Density for method '{method}' must have the same length as grid_points")
        plt.plot(grid_points, density, label=method, linewidth=2)

    if true_density is not None:
        plt.plot(grid_points, true_density, label="True Density", linewidth=2, linestyle='--')

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if show:
        plt.show()
        plt.close(fig)
    else:
        return fig
