"""Right-censored density estimators.

Provides:
- RightCensoredInitEstimator: IPCW-weighted HAL-MLE for initialization (Stage 1)
- RightCensoredEMEstimator: EM-based refinement (Stage 2, monolithic)
- RightCensoredEMStage: Standalone EM stage for use with tuners
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd

from haldensity.estimation.base_estimator import BaseEstimator
from haldensity.censoring._defaults import EMStageResult, EM_DEFAULTS
from haldensity.censoring._base_mle import WeightedHALMLEEstimator


logger = logging.getLogger(__name__)


# =============================================================================
# RightCensoredInitEstimator (formerly RightCensoredIPCWEstimator)
# =============================================================================


class RightCensoredInitEstimator(WeightedHALMLEEstimator):
    """IPCW-weighted HAL-MLE for right-censored data (Stage 1 initializer).

    This estimator fits a HAL-MLE density on uncensored observations,
    weighted by inverse probability of censoring weights (IPCW).
    Inherits all functionality from WeightedHALMLEEstimator.

    The optimization problem is:
        min_theta  -sum_i w_i * phi(x_i)^T theta + n_eff * log(Z(theta))
        s.t.       ||theta[k:]||_1 <= norm_constraint

    where:
    - w_i are IPCW weights (Delta_i / S_C(T_i))
    - n_eff = sum(w_i) is the effective sample size
    - Z(theta) is the normalizing constant

    Parameters
    ----------
    tol : float
        Tolerance for pruning small theta coefficients.
    norm_constraint : float
        L1 norm constraint for theta coefficients.
    n_grid_points : int
        Number of grid points for density evaluation.
    basis_order : int
        Order of the truncated power basis (0 = step functions).
    solver : str
        CVXPY solver to use.
    log_dir : str | None
        Directory for logging.
    log_frequency : int
        Frequency of logging (-1 = no logging).
    use_secondary_solver : bool
        Whether to try fallback solvers on failure.
    solver_waterfall : list[str]
        Fallback solver sequence.
    max_threads : int | None
        Maximum threads for MOSEK solver.
    include_intercept_in_constraint : bool
        Whether to include intercept in norm constraint.

    Examples
    --------
    >>> from haldensity.censoring.right import KaplanMeier, compute_ipcw_weights
    >>> km = KaplanMeier().fit(data)
    >>> weights = compute_ipcw_weights(data["T"], data["Delta"], km.predict)
    >>> uncensored = data[data["Delta"] == 1]
    >>> est = RightCensoredInitEstimator(norm_constraint=50.0)
    >>> est.fit(pd.DataFrame({"W1": uncensored["T"]}), sample_weights=weights[data["Delta"] == 1])
    """
    pass  # All functionality inherited from WeightedHALMLEEstimator


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
    """Precompute components for efficient sampling in E-step."""
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
    """Sample from the tail distribution f(T* | T* > C)."""
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
    """Perform E-step via multiple imputation for censored observations."""
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
# RightCensoredEMStage
# =============================================================================


class RightCensoredEMStage:
    """Standalone EM stage for right-censored density estimation.

    Implements the Expectation-Maximization algorithm with multiple
    imputation for refining an initial density estimate.

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
    ) -> EMStageResult:
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
        EMStageResult
            Container with the refined estimator, theta path, and convergence info.
        """
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")

        # Import here to avoid circular imports
        from .metrics import incomplete_loglik

        # Extract required attributes from initial estimator
        basis_grid_points = np.asarray(initial_estimator.grid_points_hal_selected, dtype=float).copy()
        basis_order = int(initial_estimator.basis_order)
        
        poly_cols = basis_order if basis_order > 0 else 0
        knot_start = 1 + poly_cols
        
        all_knots = np.asarray(initial_estimator._grid_points_hal, dtype=float)
        selected_indices = []
        for knot in basis_grid_points:
            idx = np.where(np.abs(all_knots - knot) < 1e-10)[0]
            if len(idx) > 0:
                selected_indices.append(idx[0])
        
        original_theta = np.asarray(initial_estimator.theta_hat, dtype=float)
        theta_full = np.zeros(1 + poly_cols + len(basis_grid_points))
        theta_full[:knot_start] = original_theta[:knot_start]
        for i, orig_idx in enumerate(selected_indices):
            theta_full[knot_start + i] = original_theta[knot_start + orig_idx]

        theta_path: list[np.ndarray] = [theta_full.copy()]
        current_estimator = initial_estimator
        final_augmented: Optional[pd.DataFrame] = None

        prev_ll = incomplete_loglik(current_estimator, data, time_col="T", delta_col="Delta")
        if self.verbose:
            logger.info(f"RightCensoredEMStage: Initial incomplete-data log-likelihood: {prev_ll:.4f}")

        em_converged = False
        em_iterations = 0

        for em_iter in range(self.max_em_iter):
            em_iterations = em_iter + 1

            # E-step
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

            curr_ll = incomplete_loglik(mstep_est, data, time_col="T", delta_col="Delta")
            ll_diff = np.abs(curr_ll - prev_ll)

            if self.verbose:
                logger.info(
                    f"RightCensoredEMStage Iter {em_iter + 1}: LL={curr_ll:.4f}, "
                    f"delta={ll_diff:.6f}, E={e_time:.3f}s, M={m_time:.3f}s"
                )

            if ll_diff < self.em_tol:
                if self.verbose:
                    logger.info(f"RightCensoredEMStage: Converged at iteration {em_iter + 1}")
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
    ) -> "RightCensoredInitEstimator":
        """Fit weighted HAL estimator on pooled imputed data."""
        weights = pooled_df["weight"].values.astype(float)
        df_values = pd.DataFrame({"W1": pooled_df["W1"].values})

        est = RightCensoredInitEstimator(
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
        )
        est.fit(
            df_values,
            sample_weights=weights,
            grid_points_override=grid_override,
            warm_start_theta=warm_theta if len(warm_theta) > 0 else None,
            skip_coefficient_pruning=True,
        )
        return est


# =============================================================================
# RightCensoredEMEstimator
# =============================================================================


class RightCensoredEMEstimator(BaseEstimator):
    """EM with multiple imputation for right-censored data on [0, 1].

    This estimator combines:
    1. IPCW-HAL-MLE initialization on uncensored observations
    2. EM refinement via RightCensoredEMStage with multiple imputation

    Parameters
    ----------
    tol : float
        Tolerance for pruning small coefficients.
    norm_constraint : float
        L1 norm constraint for theta.
    n_grid_points : int
        Number of grid points for density evaluation.
    basis_order : int
        Order of the truncated power basis.
    m_imputations : int
        Number of imputations per censored observation.
    max_em_iter : int
        Maximum EM iterations.
    em_tol : float
        Convergence tolerance for log-likelihood.
    use_sc_adjustment : bool
        Whether to adjust for censoring in E-step sampling.
    log_dir : str | None
        Logging directory.
    log_frequency : int
        Logging frequency.
    verbose : bool
        Whether to print progress.
    init_solver : str
        Solver for initial IPCW fit.
    m_step_solver : str
        Solver for M-step fits.
    init_norm_constraint : float | None
        Norm constraint for initialization.
    m_step_norm_constraint : float | None
        Norm constraint for M-step.
    e_step_n_grid : int
        Grid resolution for E-step sampling.
    rng_seed : int
        Random seed.
    """

    def __init__(
        self,
        tol: float = EM_DEFAULTS.tol,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        m_imputations: int = EM_DEFAULTS.m_imputations,
        max_em_iter: int = EM_DEFAULTS.max_em_iter,
        em_tol: float = EM_DEFAULTS.em_tol,
        use_sc_adjustment: bool = EM_DEFAULTS.use_sc_adjustment,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        verbose: bool = False,
        init_solver: str = EM_DEFAULTS.init_solver,
        m_step_solver: str = EM_DEFAULTS.m_step_solver,
        init_norm_constraint: Optional[float] = None,
        m_step_norm_constraint: Optional[float] = None,
        e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        rng_seed: int = 0,
    ):
        super().__init__(
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir,
            log_frequency=log_frequency,
        )
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.m_imputations = m_imputations
        self.max_em_iter = max_em_iter
        self.em_tol = em_tol
        self.use_sc_adjustment = use_sc_adjustment
        self.verbose = verbose
        self.init_solver = init_solver
        self.m_step_solver = m_step_solver

        self.m_step_solver_sequence: list[str] = []
        for cand in (m_step_solver, "CLARABEL", "ECOS", "SCS"):
            if cand not in self.m_step_solver_sequence:
                self.m_step_solver_sequence.append(cand)

        self.init_norm_constraint = (
            init_norm_constraint if init_norm_constraint is not None else norm_constraint
        )
        self.m_step_norm_constraint = (
            m_step_norm_constraint if m_step_norm_constraint is not None else norm_constraint
        )
        self.e_step_n_grid = e_step_n_grid
        self.rng_seed = rng_seed

        # Fitted attributes
        self.km_: Optional[Any] = None
        self.uncensored_augmented_: Optional[pd.DataFrame] = None
        self.theta_path_: list[np.ndarray] = []
        self.em_iterations_: int = 0
        self.em_converged_: bool = False
        self._current_estimator: Optional[BaseEstimator] = None
        self._em_stage_result: Optional[EMStageResult] = None

    def _init_ipcw(self, data: pd.DataFrame) -> RightCensoredInitEstimator:
        """Fit initial IPCW-weighted HAL estimator on uncensored observations."""
        from .km import KaplanMeier
        from .weights import compute_ipcw_weights

        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
        self.km_ = km

        T_vals = np.asarray(data["T"].values, dtype=float)
        Delta_vals = np.asarray(data["Delta"].values, dtype=int)

        weights = compute_ipcw_weights(
            T=T_vals,
            Delta=Delta_vals,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        uncensored_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        w_unc = weights[uncensored_mask]

        est = RightCensoredInitEstimator(
            tol=self.tol,
            norm_constraint=self.init_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            use_secondary_solver=True,
            solver=self.init_solver,
            include_intercept_in_constraint=True,
        )
        est.fit(df_unc, sample_weights=w_unc)
        return est

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> "RightCensoredEMEstimator":  # type: ignore[override]
        """Fit the EM-IPCW-HAL estimator.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns 'T' (observed time) and 'Delta' (event indicator).

        Returns
        -------
        self
        """
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")

        if self.verbose:
            logger.info("Initializing IPCW-HAL-MLE...")

        init_est = self._init_ipcw(data)

        em_stage = RightCensoredEMStage(
            m_imputations=self.m_imputations,
            max_em_iter=self.max_em_iter,
            em_tol=self.em_tol,
            norm_constraint=self.m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            use_sc_adjustment=self.use_sc_adjustment,
            e_step_n_grid=self.e_step_n_grid,
            tol=self.tol,
            m_step_solver=self.m_step_solver,
            m_step_solver_sequence=self.m_step_solver_sequence,
            include_intercept_in_constraint=True,
            verbose=self.verbose,
            rng_seed=self.rng_seed,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
        )

        def _s_c_predict_wrapper(x: np.ndarray) -> np.ndarray:
            if self.km_ is not None:
                return np.atleast_1d(self.km_.predict(x))
            return np.ones_like(x)

        em_result = em_stage.run(
            initial_estimator=init_est,
            data=data,
            S_c_predict=_s_c_predict_wrapper,
        )
        self._em_stage_result = em_result

        if init_est.theta_hat is not None:
            self.theta_path_ = [init_est.theta_hat.copy()] + em_result.theta_path[1:]
        else:
            self.theta_path_ = em_result.theta_path

        self.em_iterations_ = em_result.em_iterations
        self.em_converged_ = em_result.em_converged
        self.uncensored_augmented_ = em_result.final_augmented_data

        final_est = em_result.final_estimator
        self._current_estimator = final_est

        if final_est.theta_hat is None:
            raise RuntimeError("EM stage failed: final estimator has no theta_hat")
        if final_est._grid_points_hal is None:
            raise RuntimeError("EM stage failed: final estimator has no _grid_points_hal")

        self.theta_hat = final_est.theta_hat.copy()
        self._grid_points_hal = final_est._grid_points_hal.copy()
        self.grid_midpoints = final_est.grid_midpoints.copy()
        self.delta_j = final_est.delta_j.copy()
        self.grid_points = final_est.grid_points.copy()
        self.grid_points_hal_selected = final_est.grid_points_hal_selected.copy()
        self.basis_names = final_est.basis_names
        self.fitted_theta_dict = final_est.fitted_theta_dict
        self.is_fitted = True

        self.theta_full_ = self.theta_hat.copy()
        self.basis_full_ = self._grid_points_hal.copy()
        self.theta_subset_, self.active_knots_ = self._prune_theta(
            self.theta_full_,
            self.basis_full_,
        )

        return self

    def get_results(self) -> dict:
        """Return standardized results plus EM-specific fields."""
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")

        base = self._get_common_results()
        base.update({
            "theta_path": [theta.tolist() for theta in self.theta_path_],
            "has_km": self.km_ is not None,
            "em_iterations": self.em_iterations_,
            "em_converged": self.em_converged_,
        })
        return base

    def get_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the estimated density on the evaluation grid."""
        if hasattr(self, "_current_estimator") and isinstance(
            self._current_estimator, BaseEstimator
        ):
            return self._current_estimator.get_density()
        return super().get_density()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate density at specific points."""
        if hasattr(self, "_current_estimator") and isinstance(
            self._current_estimator, BaseEstimator
        ):
            return self._current_estimator.get_density_at_points(points)
        return super().get_density_at_points(points)

    def _prune_theta(
        self,
        theta: np.ndarray,
        knots: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prune theta coefficients below tolerance."""
        poly_cols = self.basis_order if self.basis_order > 0 else 0
        knot_start = 1 + poly_cols

        if knot_start > theta.size:
            knot_start = theta.size

        truncated = theta[knot_start:]
        if truncated.size == 0:
            pruned = theta[:knot_start].copy()
            return pruned, np.array([], dtype=float)

        mask = np.abs(truncated) > self.tol
        if not np.any(mask):
            pruned = np.zeros(knot_start + truncated.size, dtype=float)
            pruned[:knot_start] = theta[:knot_start]
            pruned[knot_start:] = truncated
            return pruned, knots.copy()

        active_knots = knots[mask].copy()
        pruned = np.zeros(knot_start + mask.sum(), dtype=float)
        pruned[:knot_start] = theta[:knot_start]
        pruned[knot_start:] = truncated[mask]

        return pruned, active_knots
