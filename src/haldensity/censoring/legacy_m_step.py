import numpy as np
import cvxpy as cp
from typing import Optional, Sequence
from haldensity.estimation.base_estimator import BaseEstimator


def _design_matrix(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    if knots.size == 0:
        return np.zeros((values.size, 0))
    return np.maximum(values[:, None] - knots[None, :], 0.0)


class LegacyMStepResult(BaseEstimator):
    def __init__(
        self,
        theta_hat: np.ndarray,
        knots: np.ndarray,
        eval_grid: np.ndarray,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        tol: float = 1e-4,
        density_values: Optional[np.ndarray] = None,
        grid_midpoints: Optional[np.ndarray] = None,
        delta_j: Optional[np.ndarray] = None,
    ):
        super().__init__(tol=tol, basis_order=0, log_dir=log_dir, log_frequency=log_frequency)
        self.theta_hat = theta_hat
        self._grid_points_hal = knots.copy()
        if density_values is None or grid_midpoints is None or delta_j is None:
            grid_midpoints = (eval_grid[:-1] + eval_grid[1:]) / 2
            delta_j = eval_grid[1:] - eval_grid[:-1]
            basis_mid = _design_matrix(grid_midpoints, knots)
            log_density = theta_hat[0] + basis_mid @ theta_hat[1:]
            density_values = np.exp(log_density)
            density_values /= np.sum(density_values * delta_j)
        self.grid_midpoints = grid_midpoints
        self.delta_j = delta_j
        self.grid_points = eval_grid
        self._precomputed_density = density_values
        self.grid_points_hal_selected = knots[np.abs(theta_hat[1:]) > tol]
        self.is_fitted = True
        self.basis_names = ["Intercept"] + [f"(x - {knot:.6f})_+" for knot in knots]
        self.fitted_theta_dict = {name: float(val) for name, val in zip(self.basis_names, theta_hat)}

    def get_density(self):
        return self.grid_midpoints, self._precomputed_density.copy()


def solve_legacy_m_step(
    pooled_df,
    knots: np.ndarray,
    norm_constraint: float,
    warm_start_theta: Optional[np.ndarray],
    tol: float,
    solver_sequence: Sequence[str],
    n_eval_grid: int = 200,
) -> LegacyMStepResult:
    x = pooled_df["W1"].values.astype(float)
    weights = pooled_df["weight"].values.astype(float) if "weight" in pooled_df.columns else np.ones_like(x)
    b_ik = _design_matrix(x, knots)
    theta = cp.Variable(knots.size + 1)
    log_f_i = theta[0] + b_ik @ theta[1:]
    first_term = -cp.sum(cp.multiply(weights, log_f_i))
    base_grid = np.linspace(0.0, 1.0, n_eval_grid)
    grid_eval = np.sort(np.unique(np.concatenate((base_grid, knots))))
    if grid_eval.size < 2:
        grid_eval = np.array([0.0, 1.0])
    grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
    b_jk = _design_matrix(grid_midpoints, knots)
    delta_j = grid_eval[1:] - grid_eval[:-1]
    log_terms = np.log(delta_j) + theta[0] + b_jk @ theta[1:]
    log_Z = cp.log_sum_exp(log_terms)
    second_term = np.sum(weights) * log_Z
    objective = cp.Minimize(first_term + second_term)
    constraints = [cp.norm1(theta) <= norm_constraint]
    problem = cp.Problem(objective, constraints)

    if warm_start_theta is not None and len(warm_start_theta) == knots.size + 1:
        theta.value = warm_start_theta
        warm_kwargs = {"warm_start": True}
    else:
        warm_kwargs = {}

    last_error = None
    for solver in solver_sequence:
        try:
            problem.solve(solver=solver, **warm_kwargs)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Legacy M-step failed for all solvers. Last error: {last_error}")

    theta_val = np.asarray(theta.value, dtype=float).copy()
    return LegacyMStepResult(theta_val, knots, grid_eval, tol=tol)

