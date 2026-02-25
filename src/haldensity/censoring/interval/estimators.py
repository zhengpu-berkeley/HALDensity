"""Interval-censored density estimators.

Provides:
- IntervalCensoredInitEstimator: Midpoint-imputed HAL-MLE for initialization (Stage 1)
- IntervalCensoredEMEstimator: EM-based refinement (Stage 2, monolithic)
- IntervalCensoredEMStage: Standalone EM stage for use with tuners
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
import numpy as np
import pandas as pd
import cvxpy as cp

from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.utils.basis import create_basis_functions
from haldensity.censoring._defaults import EMStageResult, EM_DEFAULTS
from haldensity.censoring._base_mle import WeightedHALMLEEstimator


logger = logging.getLogger(__name__)


# =============================================================================
# IntervalCensoredInitEstimator (formerly IntervalCensoredMidpointEstimator)
# =============================================================================


class IntervalCensoredInitEstimator(BaseEstimator):
    """HAL density estimator using midpoint imputation for interval-censored data.

    This is the Stage 1 initializer for interval-censored data.
    Computes midpoints W1 = (L+R)/2 and fits HAL-MLE.

    Parameters
    ----------
    tol : float
        Tolerance for pruning small theta coefficients.
    norm_constraint : float
        L1 norm constraint.
    n_grid_points : int
        Number of grid points for density evaluation.
    basis_order : int
        Order of the truncated power basis.
    solver : str
        CVXPY solver.
    log_dir : str | None
        Logging directory.
    log_frequency : int
        Logging frequency.
    use_secondary_solver : bool
        Whether to use fallback solvers.
    solver_waterfall : list[str]
        Fallback solver sequence.
    max_threads : int | None
        Maximum threads for MOSEK.
    include_intercept_in_constraint : bool
        Whether to include intercept in norm constraint.
    """

    def __init__(
        self,
        tol: float = 1e-4,
        norm_constraint: float = 3.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        solver: str = "ECOS",
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        use_secondary_solver: bool = False,
        solver_waterfall: list[str] = ["MOSEK", "CLARABEL", "ECOS", "SCS"],
        max_threads: Optional[int] = None,
        include_intercept_in_constraint: bool = False,
    ):
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = float(norm_constraint)
        self.n_grid_points = int(n_grid_points)
        self.solver = str(solver)
        self.use_secondary_solver = bool(use_secondary_solver)
        self.solver_waterfall = list(solver_waterfall)
        self.max_threads = max_threads
        self.include_intercept_in_constraint = bool(include_intercept_in_constraint)

        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
        self._norm_shift: Optional[float] = None
        self._norm_Z: Optional[float] = None
        self._density_midpoints: Optional[np.ndarray] = None

    @staticmethod
    def _midpoint_impute(df: pd.DataFrame, L_col: str = "L", R_col: str = "R") -> np.ndarray:
        if L_col not in df.columns or R_col not in df.columns:
            raise ValueError(f"data must contain columns {L_col!r} and {R_col!r}")
        L = np.asarray(df[L_col].values, dtype=float).ravel()
        R = np.asarray(df[R_col].values, dtype=float).ravel()
        return 0.5 * (L + R)

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        *,
        L_col: str = "L",
        R_col: str = "R",
        grid_points_override: Optional[np.ndarray] = None,
        warm_start_theta: Optional[np.ndarray] = None,
        skip_coefficient_pruning: bool = False,
        **kwargs: Any,
    ) -> "IntervalCensoredInitEstimator":
        """Fit the midpoint-imputed HAL-MLE.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns (L, R).
        L_col : str
            Name of left interval column.
        R_col : str
            Name of right interval column.
        grid_points_override : np.ndarray | None
            Optional fixed knot locations.
        warm_start_theta : np.ndarray | None
            Optional warm start.
        skip_coefficient_pruning : bool
            If True, skip pruning to keep knot structure fixed.

        Returns
        -------
        self
        """
        x = self._midpoint_impute(data, L_col=L_col, R_col=R_col)
        n_samples = int(x.shape[0])
        if n_samples == 0:
            raise ValueError("data must be non-empty")

        # Knots for basis
        if grid_points_override is not None and len(grid_points_override) > 0:
            grid_points_hal = np.sort(np.unique(np.asarray(grid_points_override, dtype=float)))
        else:
            grid_points_hal = np.unique(np.concatenate(([0.0], x.astype(float), [1.0])))
        self._grid_points_hal = grid_points_hal

        # Basis for observed points
        df_x = pd.DataFrame({"W1": x})
        b_ik, basis_names = create_basis_functions(
            df_x, grid_points_hal, order=self.basis_order, include_intercept=True
        )
        self.basis_names = basis_names

        # Basis for evaluation grid midpoints
        grid_eval = np.linspace(0.0, 1.0, self.n_grid_points)
        grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        df_mid = pd.DataFrame({"W1": grid_midpoints})
        b_jk, _ = create_basis_functions(
            df_mid, grid_points_hal, order=self.basis_order, include_intercept=True
        )

        K = int(b_ik.shape[1])
        theta = cp.Variable(K)

        # Unweighted likelihood for midpoint pseudo-observations
        first_term = -cp.sum(b_ik @ theta)

        delta_j = grid_eval[1:] - grid_eval[:-1]
        log_delta_j = np.log(delta_j)
        log_terms = log_delta_j + b_jk @ theta
        log_Z = cp.log_sum_exp(log_terms)
        second_term = float(n_samples) * log_Z

        loss = first_term + second_term

        if self.include_intercept_in_constraint:
            constraints = [cp.norm1(theta) <= self.norm_constraint]
        else:
            start_idx = 1
            constraints = [cp.norm1(theta[start_idx:]) <= self.norm_constraint] if start_idx < K else []

        problem = cp.Problem(cp.Minimize(loss), constraints)

        warm_args = False
        if warm_start_theta is not None and len(warm_start_theta) == K:
            theta.value = np.asarray(warm_start_theta, dtype=float).ravel()
            warm_args = True

        def _solve_with_kwargs(solver_name: str, warm: bool) -> None:
            solve_kwargs: dict[str, object] = {"solver": solver_name}
            if solver_name.upper() == "MOSEK" and self.max_threads is not None:
                solve_kwargs["mosek_params"] = {"MSK_IPAR_NUM_THREADS": int(self.max_threads)}
            if warm:
                solve_kwargs["warm_start"] = True
            problem.solve(**solve_kwargs)

        try:
            _solve_with_kwargs(self.solver, warm_args)
        except Exception as exc:
            if not self.use_secondary_solver:
                raise RuntimeError(f"CVXPY optimization failed: {exc}")
            last_error: Optional[Exception] = None
            success = False
            for solver in self.solver_waterfall:
                try:
                    if warm_start_theta is not None and len(warm_start_theta) == K:
                        theta.value = np.asarray(warm_start_theta, dtype=float).ravel()
                        _solve_with_kwargs(solver, True)
                    else:
                        _solve_with_kwargs(solver, False)
                    success = True
                    break
                except Exception as e2:
                    last_error = e2
            if not success:
                raise RuntimeError(
                    f"CVXPY optimization failed with all solvers in waterfall; last error: {last_error}"
                )

        if constraints:
            self.lambda_val_lag = float(problem.constraints[0].dual_value)
        if theta.value is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")

        self.optimized_theta_raw = np.asarray(theta.value, dtype=float).copy()
        self.theta_hat = self.optimized_theta_raw.copy()

        # Prune to select knots unless parametric M-step.
        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols
        if self.theta_hat.size < knot_start:
            knot_start = self.theta_hat.size

        if skip_coefficient_pruning:
            self.grid_points_hal_selected = grid_points_hal.copy()
        else:
            self.theta_hat[knot_start:] = np.where(
                np.abs(self.theta_hat[knot_start:]) > self.tol,
                self.theta_hat[knot_start:],
                0.0,
            )
            non_zero = np.where(self.theta_hat[knot_start:] != 0)[0]
            self.grid_points_hal_selected = grid_points_hal[non_zero].copy() if non_zero.size > 0 else np.array([])

        # Normalized density on an evaluation grid.
        #
        # For order=0 (step-function HAL), the fitted log-density can have many
        # sharp jumps if knot locations are clustered. A coarse evaluation grid can
        # miss these jumps, producing densities that appear normalized on the
        # coarse grid but integrate to >1 when evaluated on a finer grid.
        #
        # Mitigation: use a sufficiently dense uniform grid for order=0 when
        # constructing the normalization constants used by get_density_at_points().
        n_out = int(max(self.n_grid_points, 2000)) if self.basis_order == 0 else int(self.n_grid_points)
        output_grid = np.linspace(0.0, 1.0, n_out)

        output_mid = (output_grid[:-1] + output_grid[1:]) / 2
        delta_out = output_grid[1:] - output_grid[:-1]
        density_out, _, max_log, norm_const = BaseEstimator.normalized_hal_density(
            output_mid, self.theta_hat, grid_points_hal, self.basis_order, delta=delta_out
        )
        self._norm_shift = max_log
        self._norm_Z = norm_const
        self._density_midpoints = density_out

        self.grid_midpoints = output_mid
        self.delta_j = delta_out
        self.grid_points = output_grid
        self.is_fitted = True

        self.fitted_theta_dict = {
            name: float(value) for name, value in zip(self.basis_names, self.theta_hat)
        }
        return self

    def _normalized_density(self, points: np.ndarray) -> np.ndarray:
        if self._norm_shift is None or self._norm_Z is None:
            raise RuntimeError("Estimator must be fitted before requesting density")
        if self._grid_points_hal is None:
            raise RuntimeError("Estimator must be fitted before requesting density")
        pts = np.asarray(points, dtype=float).ravel()
        df_pts = pd.DataFrame({"W1": pts})
        basis_eval, _ = create_basis_functions(
            df_pts, self._grid_points_hal, order=self.basis_order, include_intercept=True
        )
        log_eval = basis_eval @ self.theta_hat
        shifted = np.clip(log_eval - self._norm_shift, -700, 700)
        return np.exp(shifted) / self._norm_Z

    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted or self._density_midpoints is None:
            raise ValueError("Estimator must be fitted before getting density.")
        if self.grid_midpoints is None:
            raise ValueError("Estimator must be fitted before getting density.")
        return self.grid_midpoints, self._density_midpoints.copy()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting density.")
        return self._normalized_density(points)

    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({"lambda_val_lag": self.lambda_val_lag, "optimized_theta_raw": self.optimized_theta_raw})
        return base


# =============================================================================
# E-step Sampling Functions
# =============================================================================


def _precompute_global_cdf(
    theta_hat: np.ndarray,
    basis_grid_points: np.ndarray,
    basis_order: int,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a discrete CDF representation on a global grid in [0, 1]."""
    grid = np.linspace(0.0, 1.0, int(n_grid))
    density, delta, _, _ = BaseEstimator.normalized_hal_density(
        grid=grid, theta_hat=theta_hat, basis_grid_points=basis_grid_points, basis_order=basis_order
    )
    weights = np.maximum(density * delta, 1e-32)
    cum = np.cumsum(weights)
    total = float(cum[-1])
    if total <= 0:
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

    near_zero = width <= 1e-14
    target[near_zero] = cdf_L[near_zero]

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
    """Perform E-step via multiple imputation for interval-censored observations."""
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
        rows.append(pd.DataFrame({
            "W1": draws,
            "weight": np.full(draws.shape, impute_weight, dtype=float),
        }))
    pooled = pd.concat(rows, axis=0, ignore_index=True)
    return pooled


# =============================================================================
# IntervalCensoredEMStage
# =============================================================================


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
        """Build theta vector aligned with selected knots only."""
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

    def run(self, initial_estimator: Any, data: pd.DataFrame) -> EMStageResult:
        """Run EM iterations from an initializer with selected knots."""
        from .metrics import incomplete_loglik_interval

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
            logger.info(f"IntervalCensoredEMStage: Initial interval log-likelihood: {prev_ll:.4f}")

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

            # M-step
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
                raise RuntimeError("M-step estimator fitting failed")

            theta_full = mstep_est.theta_hat.copy()
            theta_path.append(theta_full.copy())

            curr_ll = incomplete_loglik_interval(
                mstep_est, data, L_col=self.L_col, R_col=self.R_col
            )
            ll_diff = float(np.abs(curr_ll - prev_ll))

            if self.verbose:
                logger.info(
                    f"IntervalCensoredEMStage Iter {em_iter + 1}: LL={curr_ll:.4f}, "
                    f"delta={ll_diff:.6f}, E={e_time:.3f}s, M={m_time:.3f}s"
                )

            if ll_diff < self.em_tol:
                em_converged = True
                break
            prev_ll = curr_ll

        return EMStageResult(
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
    ) -> WeightedHALMLEEstimator:
        """Fit weighted HAL on pooled imputed data, keeping knot structure fixed."""
        weights = np.asarray(pooled_df["weight"].values, dtype=float)
        df_values = pd.DataFrame({"W1": pooled_df["W1"].values})

        return WeightedHALMLEEstimator(
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


# =============================================================================
# IntervalCensoredEMEstimator
# =============================================================================


class IntervalCensoredEMEstimator(BaseEstimator):
    """Midpoint-initialized parametric EM for interval-censored data on [0, 1]."""

    def __init__(
        self,
        tol: float = EM_DEFAULTS.tol,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        m_imputations: int = EM_DEFAULTS.m_imputations,
        max_em_iter: int = EM_DEFAULTS.max_em_iter,
        em_tol: float = EM_DEFAULTS.em_tol,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        verbose: bool = False,
        init_solver: str = EM_DEFAULTS.init_solver,
        m_step_solver: str = EM_DEFAULTS.m_step_solver,
        init_norm_constraint: Optional[float] = None,
        m_step_norm_constraint: Optional[float] = None,
        e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        rng_seed: int = 0,
        L_col: str = "L",
        R_col: str = "R",
    ):
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = float(norm_constraint)
        self.n_grid_points = int(n_grid_points)
        self.m_imputations = int(m_imputations)
        self.max_em_iter = int(max_em_iter)
        self.em_tol = float(em_tol)
        self.verbose = bool(verbose)
        self.init_solver = str(init_solver)
        self.m_step_solver = str(m_step_solver)
        self.init_norm_constraint = (
            float(init_norm_constraint) if init_norm_constraint is not None else self.norm_constraint
        )
        self.m_step_norm_constraint = (
            float(m_step_norm_constraint) if m_step_norm_constraint is not None else self.norm_constraint
        )
        self.e_step_n_grid = int(e_step_n_grid)
        self.rng_seed = int(rng_seed)
        self.L_col = str(L_col)
        self.R_col = str(R_col)

        # Fitted state
        self.theta_path_: list[np.ndarray] = []
        self.em_iterations_: int = 0
        self.em_converged_: bool = False
        self.uncensored_augmented_: Optional[pd.DataFrame] = None
        self._current_estimator: Optional[BaseEstimator] = None
        self._em_stage_result: Optional[EMStageResult] = None

    def _init_midpoint(self, data: pd.DataFrame) -> IntervalCensoredInitEstimator:
        return IntervalCensoredInitEstimator(
            tol=self.tol,
            norm_constraint=self.init_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            solver=self.init_solver,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            include_intercept_in_constraint=False,
            use_secondary_solver=False,
        ).fit(data, L_col=self.L_col, R_col=self.R_col)

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> "IntervalCensoredEMEstimator":  # type: ignore[override]
        if self.L_col not in data.columns or self.R_col not in data.columns:
            raise ValueError(f"data must contain columns {self.L_col!r} and {self.R_col!r}")

        if self.verbose:
            logger.info("Initializing midpoint HAL-MLE...")
        init_est = self._init_midpoint(data)

        em_stage = IntervalCensoredEMStage(
            m_imputations=self.m_imputations,
            max_em_iter=self.max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=self.m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            tol=self.tol,
            m_step_solver=self.m_step_solver,
            include_intercept_in_constraint=True,
            verbose=self.verbose,
            rng_seed=self.rng_seed,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            e_step_n_grid=self.e_step_n_grid,
            L_col=self.L_col,
            R_col=self.R_col,
        )

        em_result = em_stage.run(initial_estimator=init_est, data=data)
        self._em_stage_result = em_result

        self.theta_path_ = em_result.theta_path
        self.em_iterations_ = em_result.em_iterations
        self.em_converged_ = em_result.em_converged
        self.uncensored_augmented_ = em_result.final_augmented_data

        final_est = em_result.final_estimator
        self._current_estimator = final_est

        if final_est.theta_hat is None or final_est._grid_points_hal is None:
            raise RuntimeError("EM stage failed: final estimator missing theta/grid")

        self.theta_hat = final_est.theta_hat.copy()
        self._grid_points_hal = final_est._grid_points_hal.copy()
        self.grid_midpoints = final_est.grid_midpoints.copy() if final_est.grid_midpoints is not None else None
        self.delta_j = final_est.delta_j.copy() if final_est.delta_j is not None else None
        self.grid_points = final_est.grid_points.copy() if final_est.grid_points is not None else None
        self.grid_points_hal_selected = (
            final_est.grid_points_hal_selected.copy()
            if final_est.grid_points_hal_selected is not None
            else None
        )
        self.basis_names = final_est.basis_names
        self.fitted_theta_dict = final_est.fitted_theta_dict
        self.is_fitted = True
        return self

    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({
            "theta_path": [theta.tolist() for theta in self.theta_path_],
            "em_iterations": self.em_iterations_,
            "em_converged": self.em_converged_,
        })
        return base

    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density()
        return super().get_density()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density_at_points(points)
        return super().get_density_at_points(points)
