"""Direct observed-data HAL-MLE estimators for right-censored data.

These estimators mirror the current-status / interval-censored direct
optimization pattern, but adapt the smooth observed-data objective to the
right-censored survival likelihood

    sum_i [Delta_i log f(T_i) + (1 - Delta_i) log S(T_i)].

The HAL representation intentionally matches the current right-censored Stage 1
IPCW path: the knot grid defaults to the uncensored observed event times
augmented with the boundary points 0 and 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.utils.basis import create_basis_functions


def build_right_censored_knot_grid(
    data: pd.DataFrame,
    *,
    time_col: str = "T",
    delta_col: str = "Delta",
    grid_points_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the Stage 1-style HAL knot grid for right-censored data."""
    if grid_points_override is not None and len(grid_points_override) > 0:
        return np.sort(np.unique(np.asarray(grid_points_override, dtype=float)))

    if time_col not in data.columns or delta_col not in data.columns:
        raise ValueError(f"data must contain columns {time_col!r} and {delta_col!r}")

    t = np.asarray(data[time_col].values, dtype=float).ravel()
    d = np.asarray(data[delta_col].values, dtype=int).ravel()
    uncensored = t[d == 1]
    return np.unique(np.concatenate(([0.0], uncensored.astype(float), [1.0])))


def _normalize_working_grid_points(working_grid_points: np.ndarray) -> np.ndarray:
    """Return a sorted unique fixed-support grid and validate it is non-empty."""
    working_grid = np.sort(np.unique(np.asarray(working_grid_points, dtype=float).ravel()))
    if working_grid.size == 0:
        raise ValueError("working_grid_points must be non-empty")
    return working_grid


def _expected_num_observed_params(*, basis_order: int, grid_points: np.ndarray) -> int:
    """Return the coefficient dimension induced by a fixed HAL support."""
    poly_cols = int(basis_order) if int(basis_order) > 0 else 0
    return 1 + poly_cols + int(np.asarray(grid_points, dtype=float).size)


@dataclass(frozen=True)
class _RightCensoredObservedProblem:
    """Cached basis evaluations used by direct observed-data optimizers."""

    grid_points_hal: np.ndarray
    basis_names: list[str]
    b_grid: np.ndarray
    delta_grid: np.ndarray
    event_basis: np.ndarray
    tail_mask: np.ndarray
    fallback_basis: np.ndarray
    n_samples: int
    n_events: int
    n_censored: int


def _build_right_censored_observed_problem(
    data: pd.DataFrame,
    *,
    n_grid_points: int,
    basis_order: int,
    time_col: str = "T",
    delta_col: str = "Delta",
    grid_points_override: Optional[np.ndarray] = None,
) -> _RightCensoredObservedProblem:
    """Build the cached basis objects for the observed right-censored loss."""
    if time_col not in data.columns or delta_col not in data.columns:
        raise ValueError(f"data must contain columns {time_col!r} and {delta_col!r}")

    observed_t = np.asarray(data[time_col].values, dtype=float).ravel()
    delta_obs = np.asarray(data[delta_col].values, dtype=int).ravel()
    if observed_t.size == 0:
        raise ValueError("data must be non-empty")
    if observed_t.shape != delta_obs.shape:
        raise ValueError("time and delta columns must have the same shape")

    grid_points_hal = build_right_censored_knot_grid(
        data,
        time_col=time_col,
        delta_col=delta_col,
        grid_points_override=grid_points_override,
    )
    grid_eval = np.linspace(0.0, 1.0, int(n_grid_points))
    midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2.0
    delta_grid = grid_eval[1:] - grid_eval[:-1]

    df_mid = pd.DataFrame({"W1": midpoints})
    b_grid, basis_names = create_basis_functions(
        df_mid, grid_points_hal, order=basis_order, include_intercept=True
    )

    event_t = observed_t[delta_obs == 1]
    if event_t.size > 0:
        df_event = pd.DataFrame({"W1": event_t})
        event_basis, _ = create_basis_functions(
            df_event, grid_points_hal, order=basis_order, include_intercept=True
        )
    else:
        event_basis = np.zeros((0, int(b_grid.shape[1])), dtype=float)

    censored_t = observed_t[delta_obs == 0]
    if censored_t.size > 0:
        tail_mask = (midpoints[None, :] > censored_t[:, None]).astype(float)
        tail_centers = np.clip(0.5 * (censored_t + 1.0), 0.0, 1.0)
        df_tail_centers = pd.DataFrame({"W1": tail_centers})
        fallback_basis, _ = create_basis_functions(
            df_tail_centers, grid_points_hal, order=basis_order, include_intercept=True
        )
    else:
        tail_mask = np.zeros((0, midpoints.size), dtype=float)
        fallback_basis = np.zeros((0, int(b_grid.shape[1])), dtype=float)

    return _RightCensoredObservedProblem(
        grid_points_hal=grid_points_hal,
        basis_names=basis_names,
        b_grid=b_grid,
        delta_grid=delta_grid,
        event_basis=event_basis,
        tail_mask=tail_mask,
        fallback_basis=fallback_basis,
        n_samples=int(observed_t.size),
        n_events=int(np.sum(delta_obs == 1)),
        n_censored=int(np.sum(delta_obs == 0)),
    )


def _observed_loss_and_grad(
    theta: np.ndarray,
    *,
    problem: _RightCensoredObservedProblem,
    min_tail_mass: float,
    compute_grad: bool = True,
) -> tuple[float, Optional[np.ndarray]]:
    """Return observed negative log-likelihood and gradient."""
    theta_vec = np.asarray(theta, dtype=float).ravel()
    expected_dim = int(problem.b_grid.shape[1])
    if theta_vec.size != expected_dim:
        raise ValueError(
            f"theta must have length {expected_dim} for the supplied working model; "
            f"got {theta_vec.size}"
        )

    log_grid = problem.b_grid @ theta_vec
    max_log_grid = float(np.max(log_grid))
    unnormalized = np.exp(np.clip(log_grid - max_log_grid, -700, 700)) * problem.delta_grid
    z_shifted = float(np.sum(unnormalized))
    if z_shifted <= 0.0 or not np.isfinite(z_shifted):
        return float("inf"), (np.zeros_like(theta_vec) if compute_grad else None)

    log_normalizer = max_log_grid + float(np.log(z_shifted))
    prob = unnormalized / z_shifted
    global_mean = prob @ problem.b_grid

    loss = 0.0
    if problem.event_basis.shape[0] > 0:
        event_log_density = problem.event_basis @ theta_vec - log_normalizer
        loss -= float(np.sum(event_log_density))

    censored_cond_means = np.zeros((problem.tail_mask.shape[0], problem.b_grid.shape[1]), dtype=float)
    if problem.tail_mask.shape[0] > 0:
        weighted_tails = problem.tail_mask * prob[None, :]
        tail_masses = np.sum(weighted_tails, axis=1)
        safe_tail_masses = np.maximum(tail_masses, float(min_tail_mass))
        loss -= float(np.sum(np.log(safe_tail_masses)))

        if compute_grad:
            censored_cond_means = weighted_tails @ problem.b_grid
            non_tiny = tail_masses > float(min_tail_mass)
            if np.any(non_tiny):
                censored_cond_means[non_tiny] /= tail_masses[non_tiny, None]
            if np.any(~non_tiny):
                censored_cond_means[~non_tiny] = problem.fallback_basis[~non_tiny]

    if not compute_grad:
        return float(loss), None

    grad = problem.n_samples * global_mean
    if problem.event_basis.shape[0] > 0:
        grad -= np.sum(problem.event_basis, axis=0)
    if censored_cond_means.shape[0] > 0:
        grad -= np.sum(censored_cond_means, axis=0)
    return float(loss), np.asarray(grad, dtype=float)


def right_censored_observed_loglik_and_gradient(
    theta: np.ndarray,
    data: pd.DataFrame,
    *,
    working_grid_points: np.ndarray,
    basis_order: int = 0,
    n_grid_points: int = 400,
    time_col: str = "T",
    delta_col: str = "Delta",
    min_tail_mass: float = 1e-12,
    compute_grad: bool = True,
) -> tuple[float, Optional[np.ndarray]]:
    """Return observed-data log-likelihood and score for a fixed-support model."""
    working_grid = _normalize_working_grid_points(working_grid_points)
    problem = _build_right_censored_observed_problem(
        data,
        n_grid_points=n_grid_points,
        basis_order=basis_order,
        time_col=time_col,
        delta_col=delta_col,
        grid_points_override=working_grid,
    )
    loss, grad = _observed_loss_and_grad(
        theta,
        problem=problem,
        min_tail_mass=min_tail_mass,
        compute_grad=compute_grad,
    )
    return -float(loss), (None if grad is None else -np.asarray(grad, dtype=float))


class _RightCensoredObservedBase(BaseEstimator):
    """Shared problem construction and density utilities for direct RC fits."""

    def __init__(
        self,
        *,
        n_grid_points: int,
        basis_order: int,
        min_tail_mass: float,
        history_every: int,
        tol: float,
        log_dir: Optional[str],
        log_frequency: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
            **kwargs,
        )
        self.n_grid_points = int(n_grid_points)
        self.min_tail_mass = float(min_tail_mass)
        self.history_every = int(history_every)

        self._event_basis: Optional[np.ndarray] = None
        self._tail_mask: Optional[np.ndarray] = None
        self._fallback_basis: Optional[np.ndarray] = None
        self._b_grid: Optional[np.ndarray] = None
        self._delta: Optional[np.ndarray] = None
        self._observed_problem: Optional[_RightCensoredObservedProblem] = None
        self._n_samples: int = 0
        self._n_events: int = 0
        self._n_censored: int = 0

        self._n_iterations_run: int = 0
        self._converged: bool = False
        self._norm_shift: Optional[float] = None
        self._norm_Z: Optional[float] = None
        self._density_midpoints: Optional[np.ndarray] = None
        self._recovery_count: int = 0
        self.optimization_history_: list[dict[str, float | int]] = []

    def _objective_is_unstable(self, obj: float) -> bool:
        """Return True when the objective should trigger numerical recovery."""
        return (not np.isfinite(obj)) or (obj > self.explosion_threshold) or (obj < -self.explosion_threshold)

    @staticmethod
    def _shrink_scale(scale: float, shrink_factor: float, minimum: float) -> float:
        """Shrink a step size while respecting a lower bound."""
        return max(float(scale) * float(shrink_factor), float(minimum))

    def _record_recovery(
        self,
        *,
        optimizer: str,
        iteration: int,
        reason: str,
        old_scale: float,
        new_scale: float,
    ) -> None:
        """Track and optionally log a numerical recovery event."""
        self._recovery_count += 1
        if self.do_log:
            self.logger.warning(
                "%s recovery at iter %d: %s; shrinking step from %.3e to %.3e and restarting momentum",
                optimizer,
                iteration,
                reason,
                old_scale,
                new_scale,
            )

    def _initialize_problem(
        self,
        data: pd.DataFrame,
        *,
        time_col: str,
        delta_col: str,
        grid_points_override: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int]:
        problem = _build_right_censored_observed_problem(
            data,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            time_col=time_col,
            delta_col=delta_col,
            grid_points_override=grid_points_override,
        )
        self._observed_problem = problem
        self._grid_points_hal = problem.grid_points_hal
        self.basis_names = list(problem.basis_names)
        self._b_grid = problem.b_grid
        self._delta = problem.delta_grid
        self._event_basis = problem.event_basis
        self._tail_mask = problem.tail_mask
        self._fallback_basis = problem.fallback_basis
        self._n_samples = int(problem.n_samples)
        self._n_events = int(problem.n_events)
        self._n_censored = int(problem.n_censored)
        return problem.grid_points_hal, int(problem.b_grid.shape[1])

    def _smooth_loss_and_grad(
        self,
        theta: np.ndarray,
        *,
        compute_grad: bool = True,
    ) -> tuple[float, Optional[np.ndarray]]:
        if self._observed_problem is None:
            raise RuntimeError("Internal right-censored structures are not initialized")
        return _observed_loss_and_grad(
            theta,
            problem=self._observed_problem,
            min_tail_mass=self.min_tail_mass,
            compute_grad=compute_grad,
        )

    def observed_loglik_and_gradient(
        self,
        theta: Optional[np.ndarray] = None,
        *,
        compute_grad: bool = True,
    ) -> tuple[float, Optional[np.ndarray]]:
        """Return observed-data log-likelihood and score for this working model."""
        theta_eval = self.theta_hat if theta is None else np.asarray(theta, dtype=float).ravel()
        if theta_eval is None:
            raise RuntimeError("Provide theta or fit the estimator before requesting the observed score.")
        loss, grad = self._smooth_loss_and_grad(theta_eval, compute_grad=compute_grad)
        return -float(loss), (None if grad is None else -np.asarray(grad, dtype=float))

    def _finalize_fit(self, theta: np.ndarray, grid_points_hal: np.ndarray) -> None:
        self.theta_hat = np.asarray(theta, dtype=float).copy()

        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = min(self.theta_hat.size, 1 + poly_cols)
        if knot_start < self.theta_hat.size:
            non_zero = np.where(np.abs(self.theta_hat[knot_start:]) > self.tol)[0]
            self.grid_points_hal_selected = (
                grid_points_hal[non_zero].copy() if non_zero.size > 0 else np.array([])
            )
        else:
            self.grid_points_hal_selected = np.array([])

        n_out = int(max(self.n_grid_points, 2000)) if self.basis_order == 0 else int(self.n_grid_points)
        output_grid = np.linspace(0.0, 1.0, n_out)
        output_mid = (output_grid[:-1] + output_grid[1:]) / 2.0
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


class RightCensoredObservedFISTAEstimator(_RightCensoredObservedBase):
    """Direct right-censored observed-data HAL-MLE via FISTA."""

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
        min_step: float = 1e-8,
        recovery_shrink_factor: float = 0.2,
        max_recovery_attempts: int = 8,
        restart_on_objective_increase: bool = True,
        min_tail_mass: float = 1e-12,
        include_intercept_in_penalty: bool = False,
        history_every: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
    ) -> None:
        super().__init__(
            lam=lam,
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            n_grid_points=n_grid_points,
            min_tail_mass=min_tail_mass,
            history_every=history_every,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.ll_change_tol = float(ll_change_tol)
        self.initial_step = float(initial_step)
        self.backtracking_factor = float(backtracking_factor)
        self.step_growth = float(step_growth)
        self.max_step = float(max_step)
        self.max_backtracking = int(max_backtracking)
        self.min_step = float(min_step)
        self.recovery_shrink_factor = float(recovery_shrink_factor)
        self.max_recovery_attempts = int(max_recovery_attempts)
        self.restart_on_objective_increase = bool(restart_on_objective_increase)
        self.include_intercept_in_penalty = bool(include_intercept_in_penalty)
        self._final_step: float = self.initial_step

    def _soft_threshold(self, z: np.ndarray, thresh: float) -> np.ndarray:
        out = np.asarray(z, dtype=float).copy()
        start_idx = 0 if self.include_intercept_in_penalty else 1
        if start_idx < out.size:
            u = out[start_idx:]
            out[start_idx:] = np.sign(u) * np.maximum(np.abs(u) - float(thresh), 0.0)
        return out

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        *,
        time_col: str = "T",
        delta_col: str = "Delta",
        grid_points_override: Optional[np.ndarray] = None,
        warm_start_theta: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "RightCensoredObservedFISTAEstimator":
        grid_points_hal, n_params = self._initialize_problem(
            data,
            time_col=time_col,
            delta_col=delta_col,
            grid_points_override=grid_points_override,
        )

        if warm_start_theta is not None and len(warm_start_theta) == n_params:
            theta = np.asarray(warm_start_theta, dtype=float).ravel().copy()
        else:
            theta = np.zeros(n_params, dtype=float)

        y = theta.copy()
        tk = 1.0
        step = min(self.initial_step, self.max_step)
        prev_ll = float("inf")
        converged = False
        n_run = 0
        self._recovery_count = 0
        self.optimization_history_ = []

        for it in range(1, int(self.n_iterations) + 1):
            n_run = it
            reg_start = 0 if self.include_intercept_in_penalty else 1
            accepted_iteration = False
            recovered_this_iter = False
            step_k = step
            x_next = theta.copy()
            g_next = float("inf")
            reg_term = 0.0

            for _ in range(self.max_recovery_attempts + 1):
                g_y, grad_y = self._smooth_loss_and_grad(y, compute_grad=True)
                if grad_y is None or not np.isfinite(g_y):
                    new_step = self._shrink_scale(step, self.recovery_shrink_factor, self.min_step)
                    if new_step >= step:
                        break
                    self._record_recovery(
                        optimizer="FISTA",
                        iteration=it,
                        reason="non-finite extrapolated loss",
                        old_scale=step,
                        new_scale=new_step,
                    )
                    step = new_step
                    y = theta.copy()
                    tk = 1.0
                    recovered_this_iter = True
                    continue

                step_k = step
                x_next = theta.copy()
                g_next = float("inf")
                accepted_backtracking = False

                for _ in range(self.max_backtracking):
                    candidate = self._soft_threshold(y - step_k * grad_y, self.lam * step_k)
                    g_cand, _ = self._smooth_loss_and_grad(candidate, compute_grad=False)
                    if not np.isfinite(g_cand):
                        step_k *= self.backtracking_factor
                        if step_k < self.min_step:
                            break
                        continue
                    cand_diff = candidate - y
                    quad_bound = g_y + float(np.dot(grad_y, cand_diff)) + 0.5 / step_k * float(
                        np.dot(cand_diff, cand_diff)
                    )
                    if g_cand <= quad_bound + 1e-10:
                        x_next = candidate
                        g_next = g_cand
                        accepted_backtracking = True
                        break
                    step_k *= self.backtracking_factor
                    if step_k < self.min_step:
                        break

                if not accepted_backtracking or not np.isfinite(g_next):
                    scale_source = max(step_k, self.min_step)
                    new_step = self._shrink_scale(
                        scale_source,
                        self.recovery_shrink_factor,
                        self.min_step,
                    )
                    if new_step >= scale_source:
                        break
                    self._record_recovery(
                        optimizer="FISTA",
                        iteration=it,
                        reason="line search failed",
                        old_scale=scale_source,
                        new_scale=new_step,
                    )
                    step = new_step
                    y = theta.copy()
                    tk = 1.0
                    recovered_this_iter = True
                    continue

                reg_term = float(np.sum(np.abs(x_next[reg_start:]))) if reg_start < x_next.size else 0.0
                obj_next = g_next + self.lam * reg_term
                if self._objective_is_unstable(float(obj_next)):
                    scale_source = max(step_k, self.min_step)
                    new_step = self._shrink_scale(
                        scale_source,
                        self.recovery_shrink_factor,
                        self.min_step,
                    )
                    if new_step >= scale_source:
                        break
                    self._record_recovery(
                        optimizer="FISTA",
                        iteration=it,
                        reason=f"unstable objective {obj_next:.2e}",
                        old_scale=scale_source,
                        new_scale=new_step,
                    )
                    step = new_step
                    y = theta.copy()
                    tk = 1.0
                    recovered_this_iter = True
                    continue

                if self.restart_on_objective_increase:
                    g_theta, _ = self._smooth_loss_and_grad(theta, compute_grad=False)
                    if np.isfinite(g_theta) and g_next > g_theta + 1e-10:
                        scale_source = max(step_k, self.min_step)
                        new_step = self._shrink_scale(
                            scale_source,
                            self.recovery_shrink_factor,
                            self.min_step,
                        )
                        if new_step >= scale_source:
                            break
                        self._record_recovery(
                            optimizer="FISTA",
                            iteration=it,
                            reason="objective increased under momentum",
                            old_scale=scale_source,
                            new_scale=new_step,
                        )
                        step = new_step
                        y = theta.copy()
                        tk = 1.0
                        recovered_this_iter = True
                        continue

                accepted_iteration = True
                break

            if not accepted_iteration:
                if self._objective_is_unstable(float(g_next)):
                    self._check_objective_explosion(float(g_next), it)
                break

            param_change = float(np.max(np.abs(x_next - theta)))
            ll_next = float(-g_next)
            ll_change = float(abs(ll_next - prev_ll)) if np.isfinite(prev_ll) else float("inf")

            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            y = x_next + ((tk - 1.0) / t_next) * (x_next - theta)
            theta = x_next
            tk = t_next
            if recovered_this_iter:
                step = max(self.min_step, min(step_k, self.max_step))
            else:
                step = min(max(step_k, self.min_step) * self.step_growth, self.max_step)
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

        self._n_iterations_run = int(n_run)
        self._converged = bool(converged)
        self._final_step = float(step)
        self._finalize_fit(theta, grid_points_hal)
        return self

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
            "recovery_count": int(self._recovery_count),
            "optimization_history": list(self.optimization_history_),
            "n_events": int(self._n_events),
            "n_censored": int(self._n_censored),
        })
        return base


class RightCensoredObservedProjectedGDEstimator(_RightCensoredObservedBase):
    """Direct right-censored observed-data HAL-MLE via projected gradient descent."""

    def __init__(
        self,
        norm_constraint: float = 3.0,
        learning_rate: float = 1e-1,
        n_iterations: int = 3000,
        tol: float = 1e-6,
        ll_change_tol: float = 1e-4,
        n_grid_points: int = 400,
        basis_order: int = 0,
        min_tail_mass: float = 1e-12,
        include_intercept_in_constraint: bool = False,
        use_nesterov: bool = False,
        nesterov_restart: bool = True,
        max_backtracking: int = 20,
        backtracking_factor: float = 0.5,
        min_learning_rate: float = 1e-8,
        recovery_shrink_factor: float = 0.2,
        max_recovery_attempts: int = 8,
        restart_on_objective_increase: bool = True,
        history_every: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
    ) -> None:
        super().__init__(
            n_iterations=n_iterations,
            tol=tol,
            basis_order=basis_order,
            n_grid_points=n_grid_points,
            min_tail_mass=min_tail_mass,
            history_every=history_every,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = float(norm_constraint)
        self.learning_rate = float(learning_rate)
        self.ll_change_tol = float(ll_change_tol)
        self.include_intercept_in_constraint = bool(include_intercept_in_constraint)
        self.use_nesterov = bool(use_nesterov)
        self.nesterov_restart = bool(nesterov_restart)
        self.max_backtracking = int(max_backtracking)
        self.backtracking_factor = float(backtracking_factor)
        self.min_learning_rate = float(min_learning_rate)
        self.recovery_shrink_factor = float(recovery_shrink_factor)
        self.max_recovery_attempts = int(max_recovery_attempts)
        self.restart_on_objective_increase = bool(restart_on_objective_increase)
        self._final_learning_rate: float = self.learning_rate

    @staticmethod
    def _project_onto_l1_ball(v: np.ndarray, z: float) -> np.ndarray:
        if z <= 0.0:
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

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        *,
        time_col: str = "T",
        delta_col: str = "Delta",
        grid_points_override: Optional[np.ndarray] = None,
        warm_start_theta: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "RightCensoredObservedProjectedGDEstimator":
        grid_points_hal, n_params = self._initialize_problem(
            data,
            time_col=time_col,
            delta_col=delta_col,
            grid_points_override=grid_points_override,
        )

        if warm_start_theta is not None and len(warm_start_theta) == n_params:
            theta = np.asarray(warm_start_theta, dtype=float).ravel().copy()
        else:
            theta = np.zeros(n_params, dtype=float)
        theta = self._apply_constraint(theta)

        converged = False
        n_run = 0
        prev_ll = float("inf")
        y = theta.copy()
        t_k = 1.0
        learning_rate = max(self.min_learning_rate, self.learning_rate)
        self._recovery_count = 0
        self.optimization_history_ = []

        for it in range(1, int(self.n_iterations) + 1):
            n_run = it
            accepted_iteration = False
            lr_k = learning_rate
            candidate = theta.copy()
            g_next = float("inf")

            for _ in range(self.max_recovery_attempts + 1):
                base_point = y if self.use_nesterov else theta
                g_base, grad = self._smooth_loss_and_grad(base_point, compute_grad=True)
                if grad is None or not np.isfinite(g_base):
                    new_lr = self._shrink_scale(
                        learning_rate,
                        self.recovery_shrink_factor,
                        self.min_learning_rate,
                    )
                    if new_lr >= learning_rate:
                        break
                    self._record_recovery(
                        optimizer="FPGD",
                        iteration=it,
                        reason="non-finite extrapolated loss",
                        old_scale=learning_rate,
                        new_scale=new_lr,
                    )
                    learning_rate = new_lr
                    y = theta.copy()
                    t_k = 1.0
                    continue

                lr_k = learning_rate
                accepted = False
                candidate = theta.copy()
                g_next = float("inf")

                for _ in range(self.max_backtracking):
                    proposal = self._apply_constraint(base_point - lr_k * grad)
                    g_prop, _ = self._smooth_loss_and_grad(proposal, compute_grad=False)
                    if np.isfinite(g_prop):
                        diff = proposal - base_point
                        quad_upper = (
                            float(g_base)
                            + float(np.dot(grad, diff))
                            + 0.5 / float(lr_k) * float(np.dot(diff, diff))
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
                    scale_source = max(lr_k, self.min_learning_rate)
                    new_lr = self._shrink_scale(
                        scale_source,
                        self.recovery_shrink_factor,
                        self.min_learning_rate,
                    )
                    if new_lr >= scale_source:
                        break
                    self._record_recovery(
                        optimizer="FPGD",
                        iteration=it,
                        reason="line search failed",
                        old_scale=scale_source,
                        new_scale=new_lr,
                    )
                    learning_rate = new_lr
                    y = theta.copy()
                    t_k = 1.0
                    continue

                if self._objective_is_unstable(float(g_next)):
                    scale_source = max(lr_k, self.min_learning_rate)
                    new_lr = self._shrink_scale(
                        scale_source,
                        self.recovery_shrink_factor,
                        self.min_learning_rate,
                    )
                    if new_lr >= scale_source:
                        break
                    self._record_recovery(
                        optimizer="FPGD",
                        iteration=it,
                        reason=f"unstable objective {g_next:.2e}",
                        old_scale=scale_source,
                        new_scale=new_lr,
                    )
                    learning_rate = new_lr
                    y = theta.copy()
                    t_k = 1.0
                    continue

                if self.use_nesterov and self.restart_on_objective_increase:
                    g_theta, _ = self._smooth_loss_and_grad(theta, compute_grad=False)
                    if np.isfinite(g_theta) and g_next > g_theta + 1e-10:
                        scale_source = max(lr_k, self.min_learning_rate)
                        new_lr = self._shrink_scale(
                            scale_source,
                            self.recovery_shrink_factor,
                            self.min_learning_rate,
                        )
                        if new_lr >= scale_source:
                            break
                        self._record_recovery(
                            optimizer="FPGD",
                            iteration=it,
                            reason="objective increased under momentum",
                            old_scale=scale_source,
                            new_scale=new_lr,
                        )
                        learning_rate = new_lr
                        y = theta.copy()
                        t_k = 1.0
                        continue

                accepted_iteration = True
                break

            if not accepted_iteration:
                if self._objective_is_unstable(float(g_next)):
                    self._check_objective_explosion(float(g_next), it)
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
            learning_rate = max(lr_k, self.min_learning_rate)
            self._final_learning_rate = learning_rate

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
                    g_next,
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
                    "log_likelihood": float(-g_next),
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

        self._n_iterations_run = int(n_run)
        self._converged = bool(converged)
        self._finalize_fit(theta, grid_points_hal)
        return self

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
            "recovery_count": int(self._recovery_count),
            "optimization_history": list(self.optimization_history_),
            "n_events": int(self._n_events),
            "n_censored": int(self._n_censored),
        })
        return base


class RightCensoredObservedFPGDEstimator(RightCensoredObservedProjectedGDEstimator):
    """Fast projected-GD variant matching the current-status optimizer style."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("use_nesterov", True)
        super().__init__(*args, **kwargs)


class RightCensoredObservedL1MLE(RightCensoredObservedProjectedGDEstimator):
    """Fixed-support observed-data HAL-MLE with an L1 norm budget."""

    def __init__(
        self,
        *,
        working_grid_points: np.ndarray,
        norm_constraint: float = 3.0,
        basis_order: int = 0,
        n_grid_points: int = 400,
        learning_rate: float = 1e-1,
        n_iterations: int = 3000,
        ll_change_tol: float = 1e-4,
        include_intercept_in_constraint: bool = False,
        warm_start_theta: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        near_zero_tol: float = 1e-8,
        min_tail_mass: float = 1e-12,
        max_backtracking: int = 20,
        backtracking_factor: float = 0.5,
        min_learning_rate: float = 1e-8,
        recovery_shrink_factor: float = 0.2,
        max_recovery_attempts: int = 8,
        restart_on_objective_increase: bool = True,
        history_every: int = 0,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
    ) -> None:
        self.working_grid_points = _normalize_working_grid_points(working_grid_points)
        self.near_zero_tol = float(near_zero_tol)
        self.warm_start_theta = self._validate_warm_start_theta(
            warm_start_theta,
            basis_order=basis_order,
        )
        self.penalized_l1_norm_: float = 0.0
        self.n_near_zero_coeffs_: int = 0
        self.n_exact_zero_coeffs_: int = 0
        self.constraint_active_: bool = False
        self.n_active_knot_coeffs_: int = 0

        super().__init__(
            norm_constraint=norm_constraint,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            tol=tol,
            ll_change_tol=ll_change_tol,
            n_grid_points=n_grid_points,
            basis_order=basis_order,
            min_tail_mass=min_tail_mass,
            include_intercept_in_constraint=include_intercept_in_constraint,
            use_nesterov=True,
            nesterov_restart=True,
            max_backtracking=max_backtracking,
            backtracking_factor=backtracking_factor,
            min_learning_rate=min_learning_rate,
            recovery_shrink_factor=recovery_shrink_factor,
            max_recovery_attempts=max_recovery_attempts,
            restart_on_objective_increase=restart_on_objective_increase,
            history_every=history_every,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )

    def _expected_n_params(self) -> int:
        return _expected_num_observed_params(
            basis_order=self.basis_order,
            grid_points=self.working_grid_points,
        )

    def _validate_warm_start_theta(
        self,
        warm_start_theta: Optional[np.ndarray],
        *,
        basis_order: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if warm_start_theta is None:
            return None
        warm_start = np.asarray(warm_start_theta, dtype=float).ravel().copy()
        order = self.basis_order if basis_order is None else int(basis_order)
        expected = _expected_num_observed_params(
            basis_order=order,
            grid_points=self.working_grid_points,
        )
        if warm_start.size != expected:
            raise ValueError(
                f"warm_start_theta must have length {expected} for basis_order={order} "
                f"and {self.working_grid_points.size} working support points; got {warm_start.size}"
            )
        return warm_start

    def _collect_fixed_support_diagnostics(self) -> None:
        start_idx = 0 if self.include_intercept_in_constraint else 1
        penalized_block = self.theta_hat[start_idx:] if self.theta_hat is not None else np.array([])
        self.penalized_l1_norm_ = (
            float(np.sum(np.abs(penalized_block))) if penalized_block.size > 0 else 0.0
        )
        self.n_near_zero_coeffs_ = (
            int(np.sum(np.abs(penalized_block) <= self.near_zero_tol))
            if penalized_block.size > 0
            else 0
        )
        self.n_exact_zero_coeffs_ = (
            int(np.sum(penalized_block == 0.0)) if penalized_block.size > 0 else 0
        )
        self.constraint_active_ = bool(
            np.isclose(
                self.penalized_l1_norm_,
                self.norm_constraint,
                atol=max(1e-8, 10.0 * self.tol),
                rtol=0.0,
            )
        )

        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = min(self.theta_hat.size, 1 + poly_cols) if self.theta_hat is not None else 0
        knot_block = self.theta_hat[knot_start:] if self.theta_hat is not None else np.array([])
        self.n_active_knot_coeffs_ = (
            int(np.sum(np.abs(knot_block) > self.tol)) if knot_block.size > 0 else 0
        )

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        *,
        time_col: str = "T",
        delta_col: str = "Delta",
        warm_start_theta: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "RightCensoredObservedL1MLE":
        if "grid_points_override" in kwargs:
            raise TypeError(
                "RightCensoredObservedL1MLE fixes the working model via working_grid_points; "
                "do not pass grid_points_override."
            )
        warm_start = self._validate_warm_start_theta(
            self.warm_start_theta if warm_start_theta is None else warm_start_theta
        )
        return super().fit(
            data,
            time_col=time_col,
            delta_col=delta_col,
            grid_points_override=self.working_grid_points,
            warm_start_theta=warm_start,
        )

    def _finalize_fit(self, theta: np.ndarray, grid_points_hal: np.ndarray) -> None:
        self.theta_hat = np.asarray(theta, dtype=float).copy()
        self._grid_points_hal = self.working_grid_points.copy()
        self.grid_points_hal_selected = self.working_grid_points.copy()

        n_out = int(max(self.n_grid_points, 2000)) if self.basis_order == 0 else int(self.n_grid_points)
        output_grid = np.linspace(0.0, 1.0, n_out)
        output_mid = (output_grid[:-1] + output_grid[1:]) / 2.0
        delta_out = output_grid[1:] - output_grid[:-1]
        density_out, _, max_log, norm_const = BaseEstimator.normalized_hal_density(
            output_mid, self.theta_hat, self._grid_points_hal, self.basis_order, delta=delta_out
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
        self._collect_fixed_support_diagnostics()

    def get_results(self) -> dict:
        base = super().get_results()
        base.update({
            "working_grid_points": self.working_grid_points.copy(),
            "penalized_l1_norm": float(self.penalized_l1_norm_),
            "constraint_active": bool(self.constraint_active_),
            "n_exact_zero_coeffs": int(self.n_exact_zero_coeffs_),
            "n_near_zero_coeffs": int(self.n_near_zero_coeffs_),
            "n_active_knot_coeffs": int(self.n_active_knot_coeffs_),
        })
        return base
