from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Tuple
from haldensity.estimation.base_estimator import BaseEstimator


def _precompute_sampling_components(
    theta_hat: np.ndarray,
    basis_grid_points: np.ndarray,
    basis_order: int,
    S_c_predict: Callable[[np.ndarray], np.ndarray],
    n_grid: int,
    use_sc_adjustment: bool,
    sc_clip: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    grid = np.linspace(0.0, 1.0, n_grid)
    density, delta, _, _ = BaseEstimator.normalized_hal_density(
        grid=grid,
        theta_hat=theta_hat,
        basis_grid_points=basis_grid_points,
        basis_order=basis_order,
    )
    if use_sc_adjustment:
        sc_vals = np.maximum(S_c_predict(grid), sc_clip)
        density = density / sc_vals
        density = density / np.sum(density * delta)
    weights = np.maximum(density * delta, 1e-32)
    cum_weights = np.cumsum(weights)
    total_mass = cum_weights[-1]
    lower_mass = np.concatenate(([0.0], cum_weights[:-1]))
    return grid, cum_weights, lower_mass, total_mass


def _sample_tail(
    y_vals: np.ndarray,
    grid: np.ndarray,
    cum_weights: np.ndarray,
    lower_mass: np.ndarray,
    total_mass: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if y_vals.size == 0:
        return np.empty(0)
    idx = np.searchsorted(grid, y_vals, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    lower = lower_mass[idx]
    tail_mass = np.maximum(total_mass - lower, 1e-16)
    u = rng.random(size=y_vals.size)
    target = lower + u * tail_mass
    near_zero = tail_mass <= 1e-12
    target[near_zero] = total_mass
    samples_idx = np.searchsorted(cum_weights, target, side="left")
    samples_idx = np.clip(samples_idx, 0, len(grid) - 1)
    samples = grid[samples_idx]
    samples[near_zero] = 1.0
    return samples


def e_step_multiple_imputation(
    data: pd.DataFrame,
    theta_hat: np.ndarray,
    basis_grid_points: np.ndarray,
    basis_order: int,
    S_c_predict: Callable[[np.ndarray], np.ndarray],
    m_imputations: int = 20,
    n_grid: int = 1000,
    use_sc_adjustment: bool = True,
    rng: np.random.Generator = np.random.default_rng(0),
) -> pd.DataFrame:
    """Vectorized multiple imputation for censored observations."""
    y = np.asarray(data["T"].values, dtype=float)
    d = np.asarray(data["Delta"].values, dtype=int)
    uncensored = pd.DataFrame({"W1": y[d == 1], "weight": np.ones(np.sum(d == 1), dtype=float)})
    censored_times = y[d == 0]
    if censored_times.size == 0 or m_imputations <= 0:
        return uncensored.reset_index(drop=True)

    grid, cum_weights, lower_mass, total_mass = _precompute_sampling_components(
        theta_hat=theta_hat,
        basis_grid_points=basis_grid_points,
        basis_order=basis_order,
        S_c_predict=S_c_predict,
        n_grid=n_grid,
        use_sc_adjustment=use_sc_adjustment,
    )

    rows = []
    impute_weight = 1.0 / m_imputations
    for _ in range(m_imputations):
        draws = _sample_tail(
            y_vals=censored_times,
            grid=grid,
            cum_weights=cum_weights,
            lower_mass=lower_mass,
            total_mass=total_mass,
            rng=rng,
        )
        rows.append(pd.DataFrame({"W1": draws, "weight": np.full(draws.shape, impute_weight, dtype=float)}))
    censored_imputed = pd.concat(rows, axis=0, ignore_index=True)
    pooled = pd.concat([uncensored, censored_imputed], axis=0, ignore_index=True)
    return pooled
