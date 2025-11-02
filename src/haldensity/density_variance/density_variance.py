import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings

from src.haldensity.utils.basis import create_basis_functions


def estimate_covariance_beta(
    data: pd.DataFrame,
    estimation_results: dict,
    basis_order: int,
    ridge_param: float = 1e-6,
) -> np.ndarray:
    """Estimate Cov(β̂) via observed Fisher information with ridge regularization."""
    theta_hat_full = np.array(estimation_results['theta_hat'])
    original_data_points = np.array(estimation_results['data_points'])
    grid_points_hal_original = np.unique(original_data_points)

    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        theta_hat_full[basis_order + 1:].nonzero()[0] + basis_order + 1,
    ])
    theta_hat = theta_hat_full[theta_selected_ind]

    n_samples = len(data)

    basis_matrix, _ = create_basis_functions(
        data,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True,
    )
    basis_matrix_selected = basis_matrix[:, theta_selected_ind]

    n_grid_points = len(estimation_results.get('grid_points', np.linspace(0, 1, 200)))
    integration_grid = np.linspace(0, 1, n_grid_points)
    integration_df = pd.DataFrame({'W1': integration_grid})

    phi_integration, _ = create_basis_functions(
        integration_df,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True,
    )
    phi_integration_selected = phi_integration[:, theta_selected_ind]

    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)

    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = np.trapz(density_unnorm, dx=dx)
    density_integration = density_unnorm / normalizing_constant

    expected_phi = np.trapz(
        phi_integration_selected * density_integration[:, np.newaxis],
        dx=dx,
        axis=0,
    )

    scores = basis_matrix_selected - expected_phi[np.newaxis, :]
    observed_information = (scores.T @ scores) / n_samples

    ridge_matrix = ridge_param * np.eye(observed_information.shape[0])
    regularized_information = observed_information + ridge_matrix

    try:
        covariance_matrix = np.linalg.inv(regularized_information) / n_samples
    except np.linalg.LinAlgError:
        warnings.warn("Information matrix is singular. Using pseudo-inverse.")
        covariance_matrix = np.linalg.pinv(regularized_information) / n_samples

    return covariance_matrix


def density_confidence_interval(
    x_values: np.ndarray,
    estimation_results: dict,
    cov_beta: np.ndarray,
    basis_order: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute delta-method CIs for density at x_values."""
    theta_hat_full = np.array(estimation_results['theta_hat'])
    original_data_points = np.array(estimation_results['data_points'])
    grid_points_hal_original = np.unique(original_data_points)

    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        theta_hat_full[basis_order + 1:].nonzero()[0] + basis_order + 1,
    ])
    theta_hat = theta_hat_full[theta_selected_ind]

    x_values = np.clip(x_values, 0, 1)
    eval_df = pd.DataFrame({'W1': x_values})

    phi_eval, _ = create_basis_functions(
        eval_df,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True,
    )
    phi_eval_selected = phi_eval[:, theta_selected_ind]

    n_grid_points = len(estimation_results.get('grid_points', np.linspace(0, 1, 200)))
    integration_grid = np.linspace(0, 1, n_grid_points)
    integration_df = pd.DataFrame({'W1': integration_grid})

    phi_integration, _ = create_basis_functions(
        integration_df,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True,
    )
    phi_integration_selected = phi_integration[:, theta_selected_ind]

    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)

    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = np.trapz(density_unnorm, dx=dx)
    density_integration = density_unnorm / normalizing_constant

    log_density_eval = phi_eval_selected @ theta_hat
    log_Z = np.log(normalizing_constant) + max_log_density
    density_eval = np.exp(log_density_eval - log_Z)

    expected_phi = np.trapz(
        phi_integration_selected * density_integration[:, np.newaxis],
        dx=dx,
        axis=0,
    )

    n_eval = len(x_values)
    standard_errors = np.zeros(n_eval)
    for i in range(n_eval):
        grad_density = density_eval[i] * (phi_eval_selected[i] - expected_phi)
        variance_f = grad_density.T @ cov_beta @ grad_density
        if variance_f < 0:
            warnings.warn(f"Negative variance detected at x={x_values[i]:.3f}, setting to 0")
            variance_f = 0
        standard_errors[i] = np.sqrt(variance_f)

    z_critical = norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * standard_errors
    lower_bounds = np.maximum(density_eval - margin_of_error, 0)
    upper_bounds = density_eval + margin_of_error

    result_df = pd.DataFrame({
        'x': x_values,
        'density': density_eval,
        'se': standard_errors,
        'lower': lower_bounds,
        'upper': upper_bounds,
    })

    return result_df
