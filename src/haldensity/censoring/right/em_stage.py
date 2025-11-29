"""EM stage for right-censored density estimation.

Provides RightCensoredEMStage class for EM refinement with multiple imputation,
and sampling functions for the E-step.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd

from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.censoring.core.models import RightCensoredEMStageResult, EM_DEFAULTS
from .ipcw_estimator import RightCensoredIPCWEstimator


logger = logging.getLogger(__name__)


# =============================================================================
# E-step Sampling Functions
# =============================================================================


def _precompute_sampling_components(
    theta_hat: np.ndarray,
    basis_grid_points: np.ndarray,
    basis_order: int,
    S_c_predict: Callable[[np.ndarray], np.ndarray],
    n_grid: int,
    use_sc_adjustment: bool,
    sc_clip: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Precompute components for efficient sampling in E-step.

    Parameters
    ----------
    theta_hat : np.ndarray
        Current theta estimate.
    basis_grid_points : np.ndarray
        Grid points for basis functions.
    basis_order : int
        Order of the basis.
    S_c_predict : Callable
        Function to predict censoring survival S_C(t).
    n_grid : int
        Number of grid points for sampling.
    use_sc_adjustment : bool
        Whether to adjust density by censoring survival.
    sc_clip : float
        Minimum value for S_C to avoid division by zero.

    Returns
    -------
    tuple
        (grid, cum_weights, lower_mass, total_mass)
    """
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
    """Sample from the tail distribution f(T* | T* > C).

    For right-censored observations, we sample T* from the conditional
    distribution given T* > C (the observed censoring time).

    Parameters
    ----------
    y_vals : np.ndarray
        Observed censoring times C for censored observations.
    grid : np.ndarray
        Discretization grid for sampling.
    cum_weights : np.ndarray
        Cumulative weights (CDF approximation).
    lower_mass : np.ndarray
        Mass below each grid point.
    total_mass : float
        Total mass (should be ~1).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Sampled values T* > C for each censored observation.
    """
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
    """Perform E-step via multiple imputation for censored observations.

    For each censored observation, sample m_imputations values from
    f(T* | T* > C, theta) and pool them with uncensored observations.

    Parameters
    ----------
    data : pd.DataFrame
        Data with columns 'T' and 'Delta'.
    theta_hat : np.ndarray
        Current theta estimate.
    basis_grid_points : np.ndarray
        Grid points for basis functions.
    basis_order : int
        Order of the basis.
    S_c_predict : Callable
        Function to predict censoring survival.
    m_imputations : int
        Number of imputations per censored observation.
    n_grid : int
        Grid resolution for sampling.
    use_sc_adjustment : bool
        Whether to adjust for censoring in sampling.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Pooled data with columns 'W1' and 'weight'.
        Uncensored observations have weight 1.
        Imputed observations have weight 1/m_imputations.
    """
    y = np.asarray(data["T"].values, dtype=float)
    d = np.asarray(data["Delta"].values, dtype=int)

    # Uncensored observations (weight = 1)
    uncensored = pd.DataFrame({
        "W1": y[d == 1],
        "weight": np.ones(np.sum(d == 1), dtype=float),
    })

    censored_times = y[d == 0]
    if censored_times.size == 0 or m_imputations <= 0:
        return uncensored.reset_index(drop=True)

    # Precompute sampling components
    grid, cum_weights, lower_mass, total_mass = _precompute_sampling_components(
        theta_hat=theta_hat,
        basis_grid_points=basis_grid_points,
        basis_order=basis_order,
        S_c_predict=S_c_predict,
        n_grid=n_grid,
        use_sc_adjustment=use_sc_adjustment,
    )

    # Sample m imputations for each censored observation
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
        rows.append(pd.DataFrame({
            "W1": draws,
            "weight": np.full(draws.shape, impute_weight, dtype=float),
        }))

    censored_imputed = pd.concat(rows, axis=0, ignore_index=True)
    pooled = pd.concat([uncensored, censored_imputed], axis=0, ignore_index=True)

    return pooled


# =============================================================================
# RightCensoredEMStage Class
# =============================================================================


class RightCensoredEMStage:
    """Standalone EM stage for right-censored density estimation.

    Implements the Expectation-Maximization algorithm with multiple
    imputation for refining an initial density estimate. Can work with
    any estimator that has theta_hat, _grid_points_hal, basis_order,
    and get_density_at_points() method.

    Parameters
    ----------
    m_imputations : int
        Number of imputations per censored observation in E-step.
    max_em_iter : int
        Maximum number of EM iterations.
    em_tol : float
        Convergence tolerance for incomplete-data log-likelihood.
    norm_constraint : float
        L1 norm constraint for M-step HAL fitting.
    n_grid_points : int
        Number of grid points for density evaluation.
    use_sc_adjustment : bool
        Whether to adjust for censoring survival in E-step sampling.
    e_step_n_grid : int
        Number of grid points for E-step sampling.
    tol : float
        Tolerance for pruning small coefficients.
    m_step_solver : str
        CVXPY solver for M-step.
    m_step_solver_sequence : list[str] | None
        Fallback solver sequence for M-step.
    include_intercept_in_constraint : bool
        Whether to include intercept in norm constraint.
    verbose : bool
        Whether to print progress information.
    rng_seed : int
        Random seed for reproducibility.
    log_dir : str | None
        Directory for logging.
    log_frequency : int
        Logging frequency.

    Examples
    --------
    >>> # Fit initial IPCW estimator
    >>> initial_est = RightCensoredIPCWEstimator(...).fit(...)
    >>> 
    >>> # Run EM refinement
    >>> em_stage = RightCensoredEMStage(m_imputations=20, max_em_iter=10)
    >>> result = em_stage.run(initial_est, data, km.predict)
    >>> final_density = result.final_estimator.get_density()
    """

    def __init__(
        self,
        m_imputations: int = EM_DEFAULTS.m_imputations,
        max_em_iter: int = EM_DEFAULTS.max_em_iter,
        em_tol: float = EM_DEFAULTS.em_tol,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        use_sc_adjustment: bool = EM_DEFAULTS.use_sc_adjustment,
        e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        tol: float = EM_DEFAULTS.tol,
        m_step_solver: str = EM_DEFAULTS.m_step_solver,
        m_step_solver_sequence: Optional[list[str]] = None,
        include_intercept_in_constraint: bool = True,
        verbose: bool = False,
        rng_seed: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
    ):
        self.m_imputations = m_imputations
        self.max_em_iter = max_em_iter
        self.em_tol = em_tol
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.use_sc_adjustment = use_sc_adjustment
        self.e_step_n_grid = e_step_n_grid
        self.tol = tol
        self.m_step_solver = m_step_solver
        self.include_intercept_in_constraint = include_intercept_in_constraint
        self.verbose = verbose
        self.rng = np.random.default_rng(rng_seed)
        self.log_dir = log_dir
        self.log_frequency = log_frequency

        # Build solver waterfall
        if m_step_solver_sequence is not None:
            self.m_step_solver_sequence = list(m_step_solver_sequence)
        else:
            self.m_step_solver_sequence = []
            for cand in (m_step_solver, "CLARABEL", "ECOS", "SCS"):
                if cand not in self.m_step_solver_sequence:
                    self.m_step_solver_sequence.append(cand)

    def run(
        self,
        initial_estimator: Any,
        data: pd.DataFrame,
        S_c_predict: Callable[[np.ndarray], np.ndarray],
    ) -> RightCensoredEMStageResult:
        """Run EM iterations starting from an initial estimator.

        Parameters
        ----------
        initial_estimator : Any
            A fitted estimator with theta_hat, _grid_points_hal, basis_order,
            and get_density_at_points() method.
        data : pd.DataFrame
            DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).
        S_c_predict : Callable
            Function to predict censoring survival probability S_C(t).

        Returns
        -------
        RightCensoredEMStageResult
            Container with the refined estimator, theta path, and convergence info.
        """
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")

        # Import here to avoid circular imports
        from .metrics import incomplete_loglik

        # Extract required attributes from initial estimator
        theta_full = np.asarray(initial_estimator.theta_hat, dtype=float).copy()
        basis_grid_points = np.asarray(initial_estimator._grid_points_hal, dtype=float).copy()
        basis_order = int(initial_estimator.basis_order)

        theta_path: list[np.ndarray] = [theta_full.copy()]
        current_estimator = initial_estimator
        final_augmented: Optional[pd.DataFrame] = None

        # Compute initial log-likelihood
        prev_ll = incomplete_loglik(current_estimator, data, time_col="T", delta_col="Delta")
        if self.verbose:
            logger.info(f"RightCensoredEMStage: Initial incomplete-data log-likelihood: {prev_ll:.4f}")

        em_converged = False
        em_iterations = 0

        for em_iter in range(self.max_em_iter):
            em_iterations = em_iter + 1

            # E-step: multiple imputation
            t0 = time.time()
            pooled = e_step_multiple_imputation(
                data=data,
                theta_hat=theta_full,
                basis_grid_points=basis_grid_points,
                basis_order=basis_order,
                S_c_predict=S_c_predict,
                m_imputations=self.m_imputations,
                n_grid=self.e_step_n_grid,
                use_sc_adjustment=self.use_sc_adjustment,
                rng=self.rng,
            )
            e_time = time.time() - t0
            final_augmented = pooled

            # M-step: fit weighted HAL on pooled data
            t1 = time.time()
            mstep_est = self._fit_m_step(
                pooled_df=pooled,
                grid_override=basis_grid_points,
                warm_theta=theta_full,
                basis_order=basis_order,
            )
            m_time = time.time() - t1

            # Update state
            current_estimator = mstep_est
            if mstep_est.theta_hat is None or mstep_est._grid_points_hal is None:
                raise RuntimeError("M-step estimator fitting failed - theta or grid points is None")

            theta_full = mstep_est.theta_hat.copy()
            basis_grid_points = mstep_est._grid_points_hal.copy()
            theta_path.append(theta_full.copy())

            # Check convergence
            curr_ll = incomplete_loglik(mstep_est, data, time_col="T", delta_col="Delta")
            ll_diff = np.abs(curr_ll - prev_ll)

            if self.verbose:
                logger.info(
                    f"RightCensoredEMStage Iter {em_iter + 1}: LL={curr_ll:.4f}, Δ={ll_diff:.6f}, "
                    f"E-step={e_time:.3f}s, M-step={m_time:.3f}s"
                )

            if ll_diff < self.em_tol:
                if self.verbose:
                    logger.info(
                        f"RightCensoredEMStage: Converged at iteration {em_iter + 1}: "
                        f"LL diff {ll_diff:.6f} < tol {self.em_tol}"
                    )
                em_converged = True
                break

            prev_ll = curr_ll

        return RightCensoredEMStageResult(
            final_estimator=current_estimator,
            theta_path=theta_path,
            em_iterations=em_iterations,
            em_converged=em_converged,
            final_augmented_data=final_augmented,
        )

    def _fit_m_step(
        self,
        pooled_df: pd.DataFrame,
        grid_override: np.ndarray,
        warm_theta: np.ndarray,
        basis_order: int,
    ) -> RightCensoredIPCWEstimator:
        """Fit weighted HAL estimator on pooled imputed data."""
        weights = pooled_df["weight"].values.astype(float)
        df_values = pd.DataFrame({"W1": pooled_df["W1"].values})

        return RightCensoredIPCWEstimator(
            tol=self.tol,
            norm_constraint=self.norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=basis_order,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            solver=self.m_step_solver,
            use_secondary_solver=True,
            solver_waterfall=self.m_step_solver_sequence,
            include_intercept_in_constraint=self.include_intercept_in_constraint,
        ).fit(
            df_values,
            sample_weights=weights,
            grid_points_override=grid_override,
            warm_start_theta=warm_theta if len(warm_theta) > 0 else None,
        )

