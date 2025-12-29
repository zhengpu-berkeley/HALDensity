"""EM stage for interval-censored density estimation (parametric EM).

We keep the knot structure fixed (selected knots from the initializer) and update
only coefficients via weighted HAL-MLE M-steps.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from haldensity.censoring.core.models import EM_DEFAULTS
from haldensity.censoring.interval.metrics import incomplete_loglik_interval
from haldensity.censoring.right.ipcw_estimator import RightCensoredIPCWEstimator

logger = logging.getLogger(__name__)


def _precompute_global_cdf(
    theta_hat: np.ndarray,
    basis_grid_points: np.ndarray,
    basis_order: int,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a discrete CDF representation on a global grid in [0, 1]."""
    from haldensity.estimation.base_estimator import BaseEstimator

    grid = np.linspace(0.0, 1.0, int(n_grid))
    density, delta, _, _ = BaseEstimator.normalized_hal_density(
        grid=grid, theta_hat=theta_hat, basis_grid_points=basis_grid_points, basis_order=basis_order
    )
    weights = np.maximum(density * delta, 1e-32)
    cum = np.cumsum(weights)
    total = float(cum[-1])
    if total <= 0:
        # Shouldn't happen, but avoid division by 0.
        cdf = np.linspace(0.0, 1.0, cum.size)
        cdf[-1] = 1.0
        return grid, cdf
    cdf = cum / total
    cdf[-1] = 1.0
    return grid, cdf


def _sample_truncated_intervals(
    L: np.ndarray,
    R: np.ndarray,
    grid: np.ndarray,
    cdf: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized inverse-CDF sampling for each interval [L_i, R_i]."""
    L = np.asarray(L, dtype=float).ravel()
    R = np.asarray(R, dtype=float).ravel()
    if L.size == 0:
        return np.empty(0, dtype=float)

    cdf_L = np.interp(L, grid, cdf, left=0.0, right=1.0)
    cdf_R = np.interp(R, grid, cdf, left=0.0, right=1.0)

    width = np.maximum(cdf_R - cdf_L, 0.0)
    u = rng.random(size=L.size)
    target = cdf_L + u * width

    # If interval has ~zero probability, fall back to L (or nearest feasible).
    near_zero = width <= 1e-14
    target[near_zero] = cdf_L[near_zero]

    # Invert with searchsorted on the discretized CDF.
    idx = np.searchsorted(cdf, target, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    samples = grid[idx]
    samples[near_zero] = np.clip(L[near_zero], 0.0, 1.0)
    return samples


def e_step_multiple_imputation_interval(
    data: pd.DataFrame,
    theta_hat: np.ndarray,
    basis_grid_points: np.ndarray,
    basis_order: int,
    m_imputations: int = 100,
    n_grid: int = 200,
    rng: np.random.Generator = np.random.default_rng(0),
    L_col: str = "L",
    R_col: str = "R",
) -> pd.DataFrame:
    """Perform E-step via multiple imputation for interval-censored observations.

    Returns a pooled pseudo-complete dataset with columns:
    - W1: imputed event times
    - weight: replicate weights (1/m_imputations)
    """
    if L_col not in data.columns or R_col not in data.columns:
        raise ValueError(f"data must contain columns {L_col!r} and {R_col!r}")

    if m_imputations <= 0:
        raise ValueError("m_imputations must be positive")

    L = np.asarray(data[L_col].values, dtype=float)
    R = np.asarray(data[R_col].values, dtype=float)

    grid, cdf = _precompute_global_cdf(
        theta_hat=np.asarray(theta_hat, dtype=float),
        basis_grid_points=np.asarray(basis_grid_points, dtype=float),
        basis_order=int(basis_order),
        n_grid=int(n_grid),
    )

    impute_weight = 1.0 / float(m_imputations)
    rows: list[pd.DataFrame] = []
    for _ in range(int(m_imputations)):
        draws = _sample_truncated_intervals(L=L, R=R, grid=grid, cdf=cdf, rng=rng)
        rows.append(
            pd.DataFrame(
                {
                    "W1": draws,
                    "weight": np.full(draws.shape, impute_weight, dtype=float),
                }
            )
        )
    pooled = pd.concat(rows, axis=0, ignore_index=True)
    return pooled


class IntervalCensoredEMStage:
    """Standalone EM stage for interval-censored density estimation (parametric EM)."""

    def __init__(
        self,
        m_imputations: int = EM_DEFAULTS.m_imputations,
        max_em_iter: int = EM_DEFAULTS.max_em_iter,
        em_tol: float = EM_DEFAULTS.em_tol,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        tol: float = EM_DEFAULTS.tol,
        m_step_solver: str = EM_DEFAULTS.m_step_solver,
        m_step_solver_sequence: Optional[list[str]] = None,
        include_intercept_in_constraint: bool = True,
        verbose: bool = False,
        rng_seed: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        L_col: str = "L",
        R_col: str = "R",
    ):
        self.m_imputations = int(m_imputations)
        self.max_em_iter = int(max_em_iter)
        self.em_tol = float(em_tol)
        self.norm_constraint = float(norm_constraint)
        self.n_grid_points = int(n_grid_points)
        self.tol = float(tol)
        self.m_step_solver = str(m_step_solver)
        self.include_intercept_in_constraint = bool(include_intercept_in_constraint)
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(int(rng_seed))
        self.log_dir = log_dir
        self.log_frequency = int(log_frequency)
        self.e_step_n_grid = int(e_step_n_grid)
        self.L_col = str(L_col)
        self.R_col = str(R_col)

        if m_step_solver_sequence is not None:
            self.m_step_solver_sequence = list(m_step_solver_sequence)
        else:
            self.m_step_solver_sequence = []
            for cand in (self.m_step_solver, "CLARABEL", "ECOS", "SCS"):
                if cand not in self.m_step_solver_sequence:
                    self.m_step_solver_sequence.append(cand)

    @staticmethod
    def _extract_theta_for_selected_knots(initial_estimator: Any) -> tuple[np.ndarray, np.ndarray, int]:
        """Build theta vector aligned with selected knots only: [intercept, poly..., knot_coeffs]."""
        basis_grid_points = np.asarray(initial_estimator.grid_points_hal_selected, dtype=float).copy()
        basis_order = int(initial_estimator.basis_order)

        poly_cols = basis_order if basis_order > 0 else 0
        knot_start = 1 + poly_cols

        all_knots = np.asarray(initial_estimator._grid_points_hal, dtype=float)
        selected_indices = []
        for knot in basis_grid_points:
            idx = np.where(np.abs(all_knots - knot) < 1e-10)[0]
            if len(idx) > 0:
                selected_indices.append(int(idx[0]))

        original_theta = np.asarray(initial_estimator.theta_hat, dtype=float)
        theta_full = np.zeros(1 + poly_cols + len(basis_grid_points), dtype=float)
        theta_full[:knot_start] = original_theta[:knot_start]
        for i, orig_idx in enumerate(selected_indices):
            theta_full[knot_start + i] = original_theta[knot_start + orig_idx]

        return theta_full, basis_grid_points, basis_order

    def run(self, initial_estimator: Any, data: pd.DataFrame):
        """Run EM iterations from an initializer with selected knots."""
        if self.L_col not in data.columns or self.R_col not in data.columns:
            raise ValueError(f"data must contain columns {self.L_col!r} and {self.R_col!r}")

        theta_full, basis_grid_points, basis_order = self._extract_theta_for_selected_knots(
            initial_estimator
        )

        theta_path: list[np.ndarray] = [theta_full.copy()]
        current_estimator = initial_estimator
        final_augmented: Optional[pd.DataFrame] = None

        prev_ll = incomplete_loglik_interval(
            current_estimator, data, L_col=self.L_col, R_col=self.R_col
        )
        if self.verbose:
            logger.info(
                f"IntervalCensoredEMStage: Initial interval log-likelihood: {prev_ll:.4f}"
            )

        em_converged = False
        em_iterations = 0

        for em_iter in range(self.max_em_iter):
            em_iterations = em_iter + 1

            # E-step
            t0 = time.time()
            pooled = e_step_multiple_imputation_interval(
                data=data,
                theta_hat=theta_full,
                basis_grid_points=basis_grid_points,
                basis_order=basis_order,
                m_imputations=self.m_imputations,
                n_grid=self.e_step_n_grid,
                rng=self.rng,
                L_col=self.L_col,
                R_col=self.R_col,
            )
            e_time = time.time() - t0
            final_augmented = pooled

            # M-step: weighted HAL on pooled, fixed knots.
            t1 = time.time()
            mstep_est = self._fit_m_step(
                pooled_df=pooled,
                grid_override=basis_grid_points,
                warm_theta=theta_full,
                basis_order=basis_order,
            )
            m_time = time.time() - t1

            current_estimator = mstep_est
            if mstep_est.theta_hat is None or mstep_est._grid_points_hal is None:
                raise RuntimeError("M-step estimator fitting failed - theta or grid points is None")

            theta_full = mstep_est.theta_hat.copy()
            theta_path.append(theta_full.copy())

            curr_ll = incomplete_loglik_interval(
                mstep_est, data, L_col=self.L_col, R_col=self.R_col
            )
            ll_diff = float(np.abs(curr_ll - prev_ll))

            if self.verbose:
                logger.info(
                    f"IntervalCensoredEMStage Iter {em_iter + 1}: LL={curr_ll:.4f}, Δ={ll_diff:.6f}, "
                    f"E-step={e_time:.3f}s, M-step={m_time:.3f}s"
                )

            if ll_diff < self.em_tol:
                em_converged = True
                break
            prev_ll = curr_ll

        # Reuse existing result container (generic enough)
        from haldensity.censoring.core.models import RightCensoredEMStageResult

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
        weights = np.asarray(pooled_df["weight"].values, dtype=float)
        df_values = pd.DataFrame({"W1": pooled_df["W1"].values})

        return RightCensoredIPCWEstimator(
            tol=self.tol,
            norm_constraint=self.norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=basis_order,
            solver=self.m_step_solver,
            use_secondary_solver=True,
            solver_waterfall=self.m_step_solver_sequence,
            include_intercept_in_constraint=self.include_intercept_in_constraint,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
        ).fit(
            df_values,
            sample_weights=weights,
            grid_points_override=np.asarray(grid_override, dtype=float),
            warm_start_theta=np.asarray(warm_theta, dtype=float) if warm_theta is not None else None,
            skip_coefficient_pruning=True,
        )


