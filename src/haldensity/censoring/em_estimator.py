from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from haldensity.estimation.base_estimator import BaseEstimator
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .sampling import e_step_multiple_imputation
from .metrics import incomplete_loglik


class EMIPCWEstimator(BaseEstimator):
    """EM with multiple imputation for right-censored data on [0, 1]."""

    def __init__(
        self,
        tol: float = 1e-4,
        norm_constraint: float = 20.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        m_imputations: int = 20,
        max_em_iter: int = 50,
        em_tol: float = 1e-3,
        use_sc_adjustment: bool = False,
        log_dir: Optional[str] = None,
        log_frequency: int = -1,
        verbose: bool = False,
        init_solver: str = "SCS",
        m_step_solver: str = "ECOS",
        init_norm_constraint: Optional[float] = None,
        m_step_norm_constraint: Optional[float] = None,
        e_step_n_grid: int = 1000,
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
        self.init_norm_constraint = init_norm_constraint if init_norm_constraint is not None else norm_constraint
        self.m_step_norm_constraint = (
            m_step_norm_constraint if m_step_norm_constraint is not None else norm_constraint
        )
        self.e_step_n_grid = e_step_n_grid
        self.rng = np.random.default_rng(rng_seed)
        self.km_: Optional[KaplanMeier] = None
        self.uncensored_augmented_: Optional[pd.DataFrame] = None
        self.theta_path_: list[np.ndarray] = []
        self.em_iterations_: int = 0
        self.em_converged_: bool = False

    def _init_ipcw(self, data: pd.DataFrame) -> WeightedCVXPYEstimator:
        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
        self.km_ = km
        weights = compute_ipcw_weights(
            T=data["T"].values,
            Delta=data["Delta"].values,
            S_c_predict=km.predict,
        )
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data.loc[uncensored_mask, "T"].values})
        w_unc = weights[uncensored_mask]
        return WeightedCVXPYEstimator(
            tol=self.tol,
            norm_constraint=self.init_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            use_secondary_solver=False,
            solver=self.init_solver,
            include_intercept_in_constraint=True,
        ).fit(df_unc, sample_weights=w_unc)

    def fit(self, data: pd.DataFrame) -> "EMIPCWEstimator":
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")
        if self.verbose:
            print("Initializing IPCW-HAL-MLE...")
        init_est = self._init_ipcw(data)
        self._debug_estimator_state("init-ipcw", init_est)
        self._current_estimator = init_est
        self.theta_full_ = init_est.theta_hat.copy()
        self.basis_full_ = init_est._grid_points_hal.copy()
        self.theta_subset_, self.active_knots_ = self._prune_theta(
            self.theta_full_,
            self.basis_full_,
        )
        theta_basis_grid_points = self.basis_full_.copy()
        self.theta_path_.append(self.theta_full_.copy())

        prev_ll = incomplete_loglik(init_est, data, time_col="T", delta_col="Delta")
        if self.verbose:
            print(f"Initial incomplete-data log-likelihood: {prev_ll:.4f}")

        self.em_converged_ = False
        for em_iter in range(self.max_em_iter):
            self.em_iterations_ = em_iter + 1
            import time as _time

            _t0 = _time.time()
            pooled = e_step_multiple_imputation(
                data=data,
                theta_hat=self.theta_full_,
                basis_grid_points=theta_basis_grid_points,
                basis_order=self.basis_order,
                S_c_predict=self.km_.predict if self.km_ is not None else (lambda x: np.ones_like(x)),
                m_imputations=self.m_imputations,
                n_grid=self.e_step_n_grid,
                use_sc_adjustment=self.use_sc_adjustment,
                rng=self.rng,
            )
            e_time = _time.time() - _t0
            self.uncensored_augmented_ = pooled
            self._debug_pooled_data(em_iter + 1, pooled)

            _t1 = _time.time()
            mstep_est = self._fit_weighted_m_step(pooled)
            m_time = _time.time() - _t1
            self._current_estimator = mstep_est
            self.theta_full_ = mstep_est.theta_hat.copy()
            self.basis_full_ = mstep_est._grid_points_hal.copy()
            self.theta_subset_, self.active_knots_ = self._prune_theta(
                self.theta_full_,
                self.basis_full_,
            )
            theta_basis_grid_points = self.basis_full_.copy()
            self.theta_path_.append(self.theta_full_.copy())
            self._debug_estimator_state(f"m-step-{em_iter + 1}", mstep_est)

            curr_ll = incomplete_loglik(mstep_est, data, time_col="T", delta_col="Delta")
            ll_diff = np.abs(curr_ll - prev_ll)
            if self.verbose:
                print(
                    f"Iter {em_iter + 1}: LL={curr_ll:.4f}, Δ={ll_diff:.6f}, "
                    f"E-step={e_time:.3f}s, M-step={m_time:.3f}s"
                )
            if ll_diff < self.em_tol:
                if self.verbose:
                    print(
                        f"\nConverged at iteration {em_iter + 1}: LL diff {ll_diff:.6f} < tol {self.em_tol}"
                    )
                self.em_converged_ = True
                break
            prev_ll = curr_ll

        final_est = self._current_estimator
        self.theta_hat = final_est.theta_hat.copy()
        self._grid_points_hal = final_est._grid_points_hal.copy()
        self.grid_midpoints = final_est.grid_midpoints.copy()
        self.delta_j = final_est.delta_j.copy()
        self.grid_points = final_est.grid_points.copy()
        self.grid_points_hal_selected = final_est.grid_points_hal_selected.copy()
        self.basis_names = final_est.basis_names
        self.fitted_theta_dict = final_est.fitted_theta_dict
        self.is_fitted = True
        return self

    def _fit_weighted_m_step(self, pooled_df: pd.DataFrame) -> WeightedCVXPYEstimator:
        weights = pooled_df["weight"].values.astype(float)
        df_values = pd.DataFrame({"W1": pooled_df["W1"].values})
        warm_estimator = getattr(self, "_current_estimator", None)
        grid_override = getattr(warm_estimator, "_grid_points_hal", None)
        warm_theta = getattr(warm_estimator, "theta_hat", None)
        return WeightedCVXPYEstimator(
            tol=self.tol,
            norm_constraint=self.m_step_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            solver=self.m_step_solver,
            use_secondary_solver=True,
            solver_waterfall=self.m_step_solver_sequence,
            include_intercept_in_constraint=True,
        ).fit(
            df_values,
            sample_weights=weights,
            grid_points_override=grid_override,
            warm_start_theta=warm_theta,
        )

    def get_results(self) -> dict:
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

    def get_density(self):
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density()
        return super().get_density()

    def get_density_at_points(self, points: np.ndarray) -> np.ndarray:
        if hasattr(self, "_current_estimator") and isinstance(self._current_estimator, BaseEstimator):
            return self._current_estimator.get_density_at_points(points)
        return super().get_density_at_points(points)

    def _prune_theta(
        self,
        theta: np.ndarray,
        knots: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def _debug_estimator_state(self, label: str, estimator: BaseEstimator) -> None:
        if not self.verbose:
            return
        try:
            theta = estimator.theta_hat
            if theta is None:
                print(f"[DEBUG][{label}] theta is None")
                return
            theta = np.asarray(theta, dtype=float)
            non_finite = np.logical_not(np.isfinite(theta)).sum()
            l1 = float(np.sum(np.abs(theta)))
            knots = getattr(estimator, "_grid_points_hal", None)
            n_knots = len(knots) if knots is not None else 0
            sel = getattr(estimator, "grid_points_hal_selected", None)
            n_sel = len(sel) if sel is not None else 0
            print(
                f"[DEBUG][{label}] theta stats: shape={theta.shape}, "
                f"min={theta.min():.4f}, max={theta.max():.4f}, "
                f"mean={theta.mean():.4f}, l1={l1:.4f}, non_finite={non_finite}, "
                f"knots={n_knots}, selected_knots={n_sel}"
            )
            grid, density = estimator.get_density()
            delta = BaseEstimator._integration_widths(grid)
            integral = float(np.sum(density * delta))
            print(
                f"[DEBUG][{label}] density stats: min={density.min():.4e}, "
                f"max={density.max():.4e}, integral={integral:.6f}"
            )
        except Exception as exc:
            print(f"[DEBUG][{label}] unable to compute diagnostics: {exc}")

    def _debug_pooled_data(self, em_iter: int, pooled_df: pd.DataFrame) -> None:
        if not self.verbose or pooled_df is None or pooled_df.empty:
            return
        weights = pooled_df["weight"].values.astype(float)
        values = pooled_df["W1"].values.astype(float)
        unc_mask = np.isclose(weights, 1.0)
        unc_min = float(values[unc_mask].min()) if np.any(unc_mask) else float("nan")
        cen_mask = ~unc_mask
        cen_min = float(values[cen_mask].min()) if np.any(cen_mask) else float("nan")
        print(
            f"[DEBUG][E-step-{em_iter}] pooled samples={len(values)}, "
            f"weight_sum={weights.sum():.4f}, weight_range=({weights.min():.4e}, {weights.max():.4e}), "
            f"value_range=({values.min():.4f}, {values.max():.4f}), "
            f"unc_min={unc_min:.4f}, cens_min={cen_min:.4f}"
        )
