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
    knot_strategy : str
        Strategy for constructing knot locations (basis grid points) when
        ``grid_points_override`` is not provided. Supported values:
        ``"midpoint"`` (default), ``"uniform_n"``, ``"uniform_sqrt_n"``,
        and ``"turnbull"`` (Turnbull NPMLE mass points via ``lifelines``).
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
        knot_strategy: str = "midpoint",
        turnbull_tol: float = 1e-5,
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
        knot_strategy : str
            Knot construction strategy when ``grid_points_override`` is None.
        turnbull_tol : float
            Convergence tolerance passed to the Turnbull fitter when
            ``knot_strategy="turnbull"``.
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
            strategy = str(knot_strategy).strip().lower()
            if strategy in {"midpoint", "default"}:
                grid_points_hal = np.unique(np.concatenate(([0.0], x.astype(float), [1.0])))
            elif strategy in {"uniform_n", "uniform-n"}:
                n_knots = max(2, int(n_samples))
                grid_points_hal = np.linspace(0.0, 1.0, n_knots, dtype=float)
            elif strategy in {"uniform_sqrt_n", "uniform-sqrt-n", "uniform_sqrt", "uniform-sqrt"}:
                n_knots = max(2, int(np.ceil(np.sqrt(float(n_samples)))))
                grid_points_hal = np.linspace(0.0, 1.0, n_knots, dtype=float)
            elif strategy in {"turnbull", "npmle", "turnbull_npmle", "turnbull-npmle"}:
                try:
                    from lifelines import KaplanMeierFitter  # type: ignore
                except Exception as exc:  # pragma: no cover
                    raise ImportError(
                        "knot_strategy='turnbull' requires the optional dependency 'lifelines'. "
                        "Install it (e.g. add lifelines to your environment) and retry."
                    ) from exc

                if L_col not in data.columns or R_col not in data.columns:
                    raise ValueError(f"data must contain columns {L_col!r} and {R_col!r}")
                L = np.asarray(data[L_col].values, dtype=float).ravel()
                R = np.asarray(data[R_col].values, dtype=float).ravel()
                if not (np.isfinite(L).all() and np.isfinite(R).all()):
                    raise ValueError("Turnbull knot strategy requires finite interval endpoints.")

                kmf = KaplanMeierFitter().fit_interval_censoring(L, R, tol=float(turnbull_tol))
                cdf_df = kmf.cumulative_density_
                if cdf_df is None or cdf_df.shape[1] == 0:
                    raise RuntimeError("Turnbull fit failed to produce cumulative density.")
                timeline = cdf_df.index.to_numpy(dtype=float)
                cdf = cdf_df.iloc[:, 0].to_numpy(dtype=float)
                dcdf = np.diff(cdf, prepend=0.0)
                jump_times = timeline[dcdf > 0]
                grid_points_hal = np.unique(np.concatenate(([0.0], jump_times.astype(float), [1.0])))
            else:
                raise ValueError(
                    f"Unknown knot_strategy {knot_strategy!r}. "
                    "Supported: 'midpoint', 'uniform_n', 'uniform_sqrt_n', 'turnbull'."
                )
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


class IntervalCensoredFISTAEstimator(BaseEstimator):
    """Direct interval-censored HAL-MLE with L1 penalty via FISTA.

    Optimizes the composite objective:
        g(theta) + lam * ||theta[penalized]||_1
    where
        g(theta) = -sum_i log P_theta(L_i < T <= R_i)
    and ``P_theta`` is approximated on a midpoint integration grid in [0, 1].
    """

    def __init__(
        self,
        lam: float = 0.01,
        n_iterations: int = 2000,
        tol: float = 1e-6,
        ll_change_tol: float = 1e-1,
        n_grid_points: int = 400,
        basis_order: int = 0,
        initial_step: float = 1.0,
        backtracking_factor: float = 0.5,
        step_growth: float = 1.05,
        max_step: float = 1.0,
        max_backtracking: int = 30,
        min_interval_mass: float = 1e-12,
        include_intercept_in_penalty: bool = False,
        history_every: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
    ):
        super().__init__(
            lam=lam,
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.n_grid_points = int(n_grid_points)
        self.ll_change_tol = float(ll_change_tol)
        self.initial_step = float(initial_step)
        self.backtracking_factor = float(backtracking_factor)
        self.step_growth = float(step_growth)
        self.max_step = float(max_step)
        self.max_backtracking = int(max_backtracking)
        self.min_interval_mass = float(min_interval_mass)
        self.include_intercept_in_penalty = bool(include_intercept_in_penalty)
        self.history_every = int(history_every)

        self._interval_mask: Optional[np.ndarray] = None
        self._fallback_basis: Optional[np.ndarray] = None
        self._b_grid: Optional[np.ndarray] = None
        self._delta: Optional[np.ndarray] = None
        self._n_samples: int = 0
        self._n_iterations_run: int = 0
        self._converged: bool = False
        self._final_step: float = self.initial_step

        self._norm_shift: Optional[float] = None
        self._norm_Z: Optional[float] = None
        self._density_midpoints: Optional[np.ndarray] = None
        self.optimization_history_: list[dict[str, float | int]] = []

    @staticmethod
    def _build_knot_grid(
        data: pd.DataFrame,
        x_mid: np.ndarray,
        knot_strategy: str,
        turnbull_tol: float,
        grid_points_override: Optional[np.ndarray],
        L_col: str,
        R_col: str,
    ) -> np.ndarray:
        if grid_points_override is not None and len(grid_points_override) > 0:
            return np.sort(np.unique(np.asarray(grid_points_override, dtype=float)))

        strategy = str(knot_strategy).strip().lower()
        n_samples = int(x_mid.shape[0])

        if strategy in {"midpoint", "default"}:
            return np.unique(np.concatenate(([0.0], x_mid.astype(float), [1.0])))
        if strategy in {"uniform_n", "uniform-n"}:
            n_knots = max(2, int(n_samples))
            return np.linspace(0.0, 1.0, n_knots, dtype=float)
        if strategy in {"uniform_sqrt_n", "uniform-sqrt-n", "uniform_sqrt", "uniform-sqrt"}:
            n_knots = max(2, int(np.ceil(np.sqrt(float(n_samples)))))
            return np.linspace(0.0, 1.0, n_knots, dtype=float)
        if strategy in {"turnbull", "npmle", "turnbull_npmle", "turnbull-npmle"}:
            try:
                from lifelines import KaplanMeierFitter  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "knot_strategy='turnbull' requires the optional dependency 'lifelines'."
                ) from exc

            if L_col not in data.columns or R_col not in data.columns:
                raise ValueError(f"data must contain columns {L_col!r} and {R_col!r}")

            L = np.asarray(data[L_col].values, dtype=float).ravel()
            R = np.asarray(data[R_col].values, dtype=float).ravel()
            kmf = KaplanMeierFitter().fit_interval_censoring(L, R, tol=float(turnbull_tol))
            cdf_df = kmf.cumulative_density_
            if cdf_df is None or cdf_df.shape[1] == 0:
                raise RuntimeError("Turnbull fit failed to produce cumulative density.")
            timeline = cdf_df.index.to_numpy(dtype=float)
            cdf = cdf_df.iloc[:, 0].to_numpy(dtype=float)
            dcdf = np.diff(cdf, prepend=0.0)
            jump_times = timeline[dcdf > 0]
            return np.unique(np.concatenate(([0.0], jump_times.astype(float), [1.0])))

        raise ValueError(
            f"Unknown knot_strategy {knot_strategy!r}. "
            "Supported: 'midpoint', 'uniform_n', 'uniform_sqrt_n', 'turnbull'."
        )

    def _soft_threshold(self, z: np.ndarray, thresh: float) -> np.ndarray:
        out = np.asarray(z, dtype=float).copy()
        start_idx = 0 if self.include_intercept_in_penalty else 1
        if start_idx < out.size:
            u = out[start_idx:]
            out[start_idx:] = np.sign(u) * np.maximum(np.abs(u) - float(thresh), 0.0)
        return out

    def _smooth_loss_and_grad(
        self,
        theta: np.ndarray,
        *,
        compute_grad: bool = True,
    ) -> tuple[float, Optional[np.ndarray]]:
        if self._b_grid is None or self._delta is None or self._interval_mask is None:
            raise RuntimeError("Internal FISTA structures are not initialized")

        log_f = self._b_grid @ theta
        max_log_f = float(np.max(log_f))
        weights = np.exp(np.clip(log_f - max_log_f, -700, 700)) * self._delta
        Z = float(np.sum(weights))
        if Z <= 0.0 or not np.isfinite(Z):
            return float("inf"), (np.zeros_like(theta) if compute_grad else None)
        prob = weights / Z

        weighted_intervals = self._interval_mask * prob[None, :]
        masses = np.sum(weighted_intervals, axis=1)
        safe_masses = np.maximum(masses, self.min_interval_mass)
        loss = -float(np.sum(np.log(safe_masses)))

        if not compute_grad:
            return loss, None

        global_mean = prob @ self._b_grid
        cond_means = weighted_intervals @ self._b_grid

        non_tiny = masses > self.min_interval_mass
        if np.any(non_tiny):
            cond_means[non_tiny] /= masses[non_tiny, None]
        if np.any(~non_tiny):
            if self._fallback_basis is None:
                raise RuntimeError("Fallback basis not initialized")
            cond_means[~non_tiny] = self._fallback_basis[~non_tiny]

        grad = self._n_samples * global_mean - np.sum(cond_means, axis=0)
        return loss, np.asarray(grad, dtype=float)

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        *,
        L_col: str = "L",
        R_col: str = "R",
        grid_points_override: Optional[np.ndarray] = None,
        knot_strategy: str = "midpoint",
        turnbull_tol: float = 1e-5,
        warm_start_theta: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "IntervalCensoredFISTAEstimator":
        if L_col not in data.columns or R_col not in data.columns:
            raise ValueError(f"data must contain columns {L_col!r} and {R_col!r}")

        L = np.asarray(data[L_col].values, dtype=float).ravel()
        R = np.asarray(data[R_col].values, dtype=float).ravel()
        if L.size == 0:
            raise ValueError("data must be non-empty")
        if L.shape != R.shape:
            raise ValueError("L and R must have the same shape")

        self._n_samples = int(L.shape[0])
        x_mid = 0.5 * (L + R)

        grid_points_hal = self._build_knot_grid(
            data=data,
            x_mid=x_mid,
            knot_strategy=knot_strategy,
            turnbull_tol=turnbull_tol,
            grid_points_override=grid_points_override,
            L_col=L_col,
            R_col=R_col,
        )
        self._grid_points_hal = grid_points_hal

        grid_eval = np.linspace(0.0, 1.0, int(self.n_grid_points))
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2.0
        delta = grid_eval[1:] - grid_eval[:-1]

        df_mid = pd.DataFrame({"W1": midpoints})
        b_grid, basis_names = create_basis_functions(
            df_mid, grid_points_hal, order=self.basis_order, include_intercept=True
        )
        self.basis_names = basis_names

        interval_mask = ((midpoints[None, :] > L[:, None]) & (midpoints[None, :] <= R[:, None])).astype(float)
        centers = np.clip(0.5 * (L + R), 0.0, 1.0)
        df_centers = pd.DataFrame({"W1": centers})
        fallback_basis, _ = create_basis_functions(
            df_centers, grid_points_hal, order=self.basis_order, include_intercept=True
        )

        self._b_grid = b_grid
        self._delta = delta
        self._interval_mask = interval_mask
        self._fallback_basis = fallback_basis

        K = int(b_grid.shape[1])
        if warm_start_theta is not None and len(warm_start_theta) == K:
            theta = np.asarray(warm_start_theta, dtype=float).ravel().copy()
        else:
            theta = np.zeros(K, dtype=float)

        y = theta.copy()
        tk = 1.0
        step = min(self.initial_step, self.max_step)
        prev_ll = float("inf")
        self.optimization_history_ = []

        converged = False
        n_run = 0

        for it in range(1, int(self.n_iterations) + 1):
            n_run = it
            g_y, grad_y = self._smooth_loss_and_grad(y, compute_grad=True)
            if grad_y is None:
                raise RuntimeError("Gradient computation failed")
            if not np.isfinite(g_y):
                break

            step_k = step
            x_next = theta.copy()
            g_next = float("inf")
            diff = np.zeros_like(theta)

            for _ in range(self.max_backtracking):
                candidate = self._soft_threshold(y - step_k * grad_y, self.lam * step_k)
                g_cand, _ = self._smooth_loss_and_grad(candidate, compute_grad=False)
                if not np.isfinite(g_cand):
                    step_k *= self.backtracking_factor
                    continue
                cand_diff = candidate - y
                quad_bound = g_y + float(np.dot(grad_y, cand_diff)) + 0.5 / step_k * float(
                    np.dot(cand_diff, cand_diff)
                )
                if g_cand <= quad_bound + 1e-10:
                    x_next = candidate
                    g_next = g_cand
                    diff = cand_diff
                    break
                step_k *= self.backtracking_factor

            reg_start = 0 if self.include_intercept_in_penalty else 1
            reg_term = float(np.sum(np.abs(x_next[reg_start:]))) if reg_start < x_next.size else 0.0
            obj_next = g_next + self.lam * reg_term
            if self._check_objective_explosion(float(obj_next), it):
                break

            param_change = float(np.max(np.abs(x_next - theta)))
            ll_next = float(-g_next)
            ll_change = float(abs(ll_next - prev_ll)) if np.isfinite(prev_ll) else float("inf")

            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            y = x_next + ((tk - 1.0) / t_next) * (x_next - theta)
            theta = x_next
            tk = t_next
            step = min(step_k * self.step_growth, self.max_step)
            prev_ll = ll_next

            if self.do_log and self.log_frequency > 0 and it % self.log_frequency == 0:
                n_nonzero = int(np.sum(np.abs(theta[reg_start:]) > self.tol)) if reg_start < theta.size else 0
                self.logger.info(
                    "Iter %d: obj=%.6f, g=%.6f, ll=%.6f, change=%.3e, ll_change=%.3e, step=%.3e, nnz=%d",
                    it,
                    obj_next,
                    g_next,
                    ll_next,
                    param_change,
                    ll_change,
                    step_k,
                    n_nonzero,
                )

            if self.history_every > 0 and it % self.history_every == 0:
                poly_cols = self.basis_order if self.basis_order > 0 else 0
                knot_start = min(theta.size, 1 + poly_cols)
                n_selected = int(np.sum(np.abs(theta[knot_start:]) > self.tol)) if knot_start < theta.size else 0
                self.optimization_history_.append({
                    "iteration": int(it),
                    "log_likelihood": float(-g_next),
                    "l1_norm": float(reg_term),
                    "n_selected_points": int(n_selected),
                })

            if ll_change < self.ll_change_tol:
                converged = True
                break

        if self.history_every > 0 and n_run > 0:
            if len(self.optimization_history_) == 0 or int(self.optimization_history_[-1]["iteration"]) != int(n_run):
                g_final, _ = self._smooth_loss_and_grad(theta, compute_grad=False)
                reg_start = 0 if self.include_intercept_in_penalty else 1
                reg_term = float(np.sum(np.abs(theta[reg_start:]))) if reg_start < theta.size else 0.0
                poly_cols = self.basis_order if self.basis_order > 0 else 0
                knot_start = min(theta.size, 1 + poly_cols)
                n_selected = int(np.sum(np.abs(theta[knot_start:]) > self.tol)) if knot_start < theta.size else 0
                self.optimization_history_.append({
                    "iteration": int(n_run),
                    "log_likelihood": float(-g_final),
                    "l1_norm": float(reg_term),
                    "n_selected_points": int(n_selected),
                })

        self.theta_hat = theta
        self._n_iterations_run = int(n_run)
        self._converged = bool(converged)
        self._final_step = float(step)

        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols
        if self.theta_hat.size < knot_start:
            knot_start = self.theta_hat.size
        if knot_start < self.theta_hat.size:
            non_zero = np.where(np.abs(self.theta_hat[knot_start:]) > self.tol)[0]
            self.grid_points_hal_selected = (
                grid_points_hal[non_zero].copy() if non_zero.size > 0 else np.array([])
            )
        else:
            self.grid_points_hal_selected = np.array([])

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
        base.update({
            "n_iterations_run": self._n_iterations_run,
            "converged": self._converged,
            "final_step": self._final_step,
            "lam": float(self.lam),
            "ll_change_tol": float(self.ll_change_tol),
            "coef_tol": float(self.tol),
            "optimization_history": list(self.optimization_history_),
        })
        return base


class IntervalCensoredProjectedGDEstimator(BaseEstimator):
    """Direct interval-censored HAL-MLE via projected gradient descent.

    Solves:
        min_theta g(theta)  s.t. ||theta[penalized]||_1 <= norm_constraint
    where g(theta) is the interval-censored negative log-likelihood.
    """

    def __init__(
        self,
        norm_constraint: float = 3.0,
        learning_rate: float = 1e-1,
        n_iterations: int = 3000,
        tol: float = 1e-6,
        ll_change_tol: float = 1e-4,
        n_grid_points: int = 400,
        basis_order: int = 0,
        min_interval_mass: float = 1e-12,
        include_intercept_in_constraint: bool = False,
        use_nesterov: bool = False,
        nesterov_restart: bool = True,
        max_backtracking: int = 20,
        backtracking_factor: float = 0.5,
        min_learning_rate: float = 1e-8,
        history_every: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
    ):
        super().__init__(
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = float(norm_constraint)
        self.learning_rate = float(learning_rate)
        self.ll_change_tol = float(ll_change_tol)
        self.n_grid_points = int(n_grid_points)
        self.min_interval_mass = float(min_interval_mass)
        self.include_intercept_in_constraint = bool(include_intercept_in_constraint)
        self.use_nesterov = bool(use_nesterov)
        self.nesterov_restart = bool(nesterov_restart)
        self.max_backtracking = int(max_backtracking)
        self.backtracking_factor = float(backtracking_factor)
        self.min_learning_rate = float(min_learning_rate)
        self.history_every = int(history_every)

        self._interval_mask: Optional[np.ndarray] = None
        self._fallback_basis: Optional[np.ndarray] = None
        self._b_grid: Optional[np.ndarray] = None
        self._delta: Optional[np.ndarray] = None
        self._n_samples: int = 0
        self._n_iterations_run: int = 0
        self._converged: bool = False
        self._density_midpoints: Optional[np.ndarray] = None
        self._norm_shift: Optional[float] = None
        self._norm_Z: Optional[float] = None
        self.optimization_history_: list[dict[str, float | int]] = []
        self._final_learning_rate: float = self.learning_rate

    @staticmethod
    def _project_onto_l1_ball(v: np.ndarray, z: float) -> np.ndarray:
        if z <= 0:
            return np.zeros_like(v)
        if float(np.sum(np.abs(v))) <= z:
            return v
        u = np.sort(np.abs(v))[::-1]
        sv = np.cumsum(u)
        rho = np.where(u > (sv - z) / np.arange(1, len(u) + 1))[0]
        rho_idx = int(rho[-1]) if len(rho) > 0 else 0
        tau = (sv[rho_idx] - z) / float(rho_idx + 1)
        return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

    def _apply_constraint(self, theta: np.ndarray) -> np.ndarray:
        out = np.asarray(theta, dtype=float).copy()
        start_idx = 0 if self.include_intercept_in_constraint else 1
        if start_idx < out.size:
            out[start_idx:] = self._project_onto_l1_ball(out[start_idx:], self.norm_constraint)
        return out

    def _smooth_loss_and_grad(
        self,
        theta: np.ndarray,
        *,
        compute_grad: bool = True,
    ) -> tuple[float, Optional[np.ndarray]]:
        if self._b_grid is None or self._delta is None or self._interval_mask is None:
            raise RuntimeError("Internal ProjectedGD structures are not initialized")

        log_f = self._b_grid @ theta
        max_log_f = float(np.max(log_f))
        weights = np.exp(np.clip(log_f - max_log_f, -700, 700)) * self._delta
        Z = float(np.sum(weights))
        if Z <= 0.0 or not np.isfinite(Z):
            return float("inf"), (np.zeros_like(theta) if compute_grad else None)
        prob = weights / Z

        weighted_intervals = self._interval_mask * prob[None, :]
        masses = np.sum(weighted_intervals, axis=1)
        safe_masses = np.maximum(masses, self.min_interval_mass)
        loss = -float(np.sum(np.log(safe_masses)))

        if not compute_grad:
            return loss, None

        global_mean = prob @ self._b_grid
        cond_means = weighted_intervals @ self._b_grid
        non_tiny = masses > self.min_interval_mass
        if np.any(non_tiny):
            cond_means[non_tiny] /= masses[non_tiny, None]
        if np.any(~non_tiny):
            if self._fallback_basis is None:
                raise RuntimeError("Fallback basis not initialized")
            cond_means[~non_tiny] = self._fallback_basis[~non_tiny]

        grad = self._n_samples * global_mean - np.sum(cond_means, axis=0)
        return loss, np.asarray(grad, dtype=float)

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        *,
        L_col: str = "L",
        R_col: str = "R",
        grid_points_override: Optional[np.ndarray] = None,
        knot_strategy: str = "midpoint",
        turnbull_tol: float = 1e-5,
        warm_start_theta: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "IntervalCensoredProjectedGDEstimator":
        if L_col not in data.columns or R_col not in data.columns:
            raise ValueError(f"data must contain columns {L_col!r} and {R_col!r}")

        L = np.asarray(data[L_col].values, dtype=float).ravel()
        R = np.asarray(data[R_col].values, dtype=float).ravel()
        if L.size == 0:
            raise ValueError("data must be non-empty")
        if L.shape != R.shape:
            raise ValueError("L and R must have the same shape")

        self._n_samples = int(L.shape[0])
        x_mid = 0.5 * (L + R)

        grid_points_hal = IntervalCensoredFISTAEstimator._build_knot_grid(
            data=data,
            x_mid=x_mid,
            knot_strategy=knot_strategy,
            turnbull_tol=turnbull_tol,
            grid_points_override=grid_points_override,
            L_col=L_col,
            R_col=R_col,
        )
        self._grid_points_hal = grid_points_hal

        grid_eval = np.linspace(0.0, 1.0, int(self.n_grid_points))
        midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2.0
        delta = grid_eval[1:] - grid_eval[:-1]
        df_mid = pd.DataFrame({"W1": midpoints})
        b_grid, basis_names = create_basis_functions(
            df_mid, grid_points_hal, order=self.basis_order, include_intercept=True
        )
        self.basis_names = basis_names

        interval_mask = ((midpoints[None, :] > L[:, None]) & (midpoints[None, :] <= R[:, None])).astype(float)
        centers = np.clip(0.5 * (L + R), 0.0, 1.0)
        df_centers = pd.DataFrame({"W1": centers})
        fallback_basis, _ = create_basis_functions(
            df_centers, grid_points_hal, order=self.basis_order, include_intercept=True
        )

        self._b_grid = b_grid
        self._delta = delta
        self._interval_mask = interval_mask
        self._fallback_basis = fallback_basis

        K = int(b_grid.shape[1])
        if warm_start_theta is not None and len(warm_start_theta) == K:
            theta = np.asarray(warm_start_theta, dtype=float).ravel().copy()
        else:
            theta = np.zeros(K, dtype=float)
        theta = self._apply_constraint(theta)
        self.optimization_history_ = []

        converged = False
        n_run = 0
        prev_ll = float("inf")
        y = theta.copy()
        t_k = 1.0

        for it in range(1, int(self.n_iterations) + 1):
            n_run = it
            base_point = y if self.use_nesterov else theta
            g_base, grad = self._smooth_loss_and_grad(base_point, compute_grad=True)
            if grad is None or not np.isfinite(g_base):
                break

            lr_k = float(self.learning_rate)
            accepted = False
            candidate = theta.copy()
            g_next = float("inf")
            for _ in range(self.max_backtracking):
                proposal = self._apply_constraint(base_point - lr_k * grad)
                g_prop, _ = self._smooth_loss_and_grad(proposal, compute_grad=False)
                if np.isfinite(g_prop):
                    d = proposal - base_point
                    quad_upper = (
                        float(g_base)
                        + float(np.dot(grad, d))
                        + 0.5 / float(lr_k) * float(np.dot(d, d))
                    )
                    if g_prop <= quad_upper + 1e-12:
                        candidate = proposal
                        g_next = float(g_prop)
                        accepted = True
                        break
                lr_k *= self.backtracking_factor
                if lr_k < self.min_learning_rate:
                    break

            if not accepted:
                break

            prev_theta = theta.copy()
            theta = candidate
            if self.use_nesterov:
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
                y_next = theta + ((t_k - 1.0) / t_next) * (theta - prev_theta)
                if self.nesterov_restart:
                    restart_dot = float(np.dot(theta - prev_theta, y_next - theta))
                    if restart_dot > 0.0:
                        t_next = 1.0
                        y_next = theta.copy()
                t_k = t_next
                y = y_next
            else:
                y = theta.copy()
            self._final_learning_rate = lr_k

            obj_next = g_next
            if self._check_objective_explosion(float(obj_next), it):
                break

            param_change = float(np.max(np.abs(theta - prev_theta)))
            ll_next = float(-g_next)
            ll_change = float(abs(ll_next - prev_ll)) if np.isfinite(prev_ll) else float("inf")
            prev_ll = ll_next

            if self.do_log and self.log_frequency > 0 and it % self.log_frequency == 0:
                start_idx = 0 if self.include_intercept_in_constraint else 1
                l1 = float(np.sum(np.abs(theta[start_idx:]))) if start_idx < theta.size else 0.0
                nnz = int(np.sum(np.abs(theta[start_idx:]) > self.tol)) if start_idx < theta.size else 0
                self.logger.info(
                    "Iter %d: obj=%.6f, ll=%.6f, change=%.3e, ll_change=%.3e, l1=%.4f, nnz=%d",
                    it,
                    obj_next,
                    ll_next,
                    param_change,
                    ll_change,
                    l1,
                    nnz,
                )

            if self.history_every > 0 and it % self.history_every == 0:
                start_idx = 0 if self.include_intercept_in_constraint else 1
                l1 = float(np.sum(np.abs(theta[start_idx:]))) if start_idx < theta.size else 0.0
                poly_cols = self.basis_order if self.basis_order > 0 else 0
                knot_start = min(theta.size, 1 + poly_cols)
                n_selected = int(np.sum(np.abs(theta[knot_start:]) > self.tol)) if knot_start < theta.size else 0
                self.optimization_history_.append({
                    "iteration": int(it),
                    "log_likelihood": float(-obj_next),
                    "l1_norm": float(l1),
                    "n_selected_points": int(n_selected),
                })

            if ll_change < self.ll_change_tol:
                converged = True
                break

        if self.history_every > 0 and n_run > 0:
            if len(self.optimization_history_) == 0 or int(self.optimization_history_[-1]["iteration"]) != int(n_run):
                g_final, _ = self._smooth_loss_and_grad(theta, compute_grad=False)
                start_idx = 0 if self.include_intercept_in_constraint else 1
                l1 = float(np.sum(np.abs(theta[start_idx:]))) if start_idx < theta.size else 0.0
                poly_cols = self.basis_order if self.basis_order > 0 else 0
                knot_start = min(theta.size, 1 + poly_cols)
                n_selected = int(np.sum(np.abs(theta[knot_start:]) > self.tol)) if knot_start < theta.size else 0
                self.optimization_history_.append({
                    "iteration": int(n_run),
                    "log_likelihood": float(-g_final),
                    "l1_norm": float(l1),
                    "n_selected_points": int(n_selected),
                })

        self.theta_hat = theta
        self._n_iterations_run = int(n_run)
        self._converged = bool(converged)

        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols
        if self.theta_hat.size < knot_start:
            knot_start = self.theta_hat.size
        if knot_start < self.theta_hat.size:
            non_zero = np.where(np.abs(self.theta_hat[knot_start:]) > self.tol)[0]
            self.grid_points_hal_selected = (
                grid_points_hal[non_zero].copy() if non_zero.size > 0 else np.array([])
            )
        else:
            self.grid_points_hal_selected = np.array([])

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
        base.update({
            "n_iterations_run": self._n_iterations_run,
            "converged": self._converged,
            "norm_constraint": float(self.norm_constraint),
            "learning_rate": float(self.learning_rate),
            "ll_change_tol": float(self.ll_change_tol),
            "coef_tol": float(self.tol),
            "use_nesterov": bool(self.use_nesterov),
            "nesterov_restart": bool(self.nesterov_restart),
            "final_learning_rate": float(self._final_learning_rate),
            "optimization_history": list(self.optimization_history_),
        })
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
