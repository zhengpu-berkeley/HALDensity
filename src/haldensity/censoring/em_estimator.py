import numpy as np
import pandas as pd
from typing import Optional
from haldensity.estimation.base_estimator import BaseEstimator
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .sampling import e_step_multiple_imputation
from .metrics import incomplete_loglik
from .legacy_m_step import solve_legacy_m_step


class EMIPCWEstimator(BaseEstimator):
    """
    EM with multiple imputation for right-censored data on [0,1].
    Initialization: IPCW-HAL-MLE via WeightedCVXPYEstimator on uncensored data with w = Delta / S_c(T).
    E-step: impute censored T on [Y,1] using density from current theta and S_c.
    M-step: HAL-MLE on pooled pseudo-complete data with the SAME norm_constraint.
    """
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
        self.m_step_solver_sequence = []
        for cand in (m_step_solver, "CLARABEL", "ECOS", "SCS"):
            if cand not in self.m_step_solver_sequence:
                self.m_step_solver_sequence.append(cand)
        self.init_norm_constraint = init_norm_constraint if init_norm_constraint is not None else norm_constraint
        self.m_step_norm_constraint = m_step_norm_constraint if m_step_norm_constraint is not None else norm_constraint
        self.e_step_n_grid = e_step_n_grid
        self.rng = np.random.default_rng(rng_seed)
        # Artifacts
        self.km_: Optional[KaplanMeier] = None
        self.uncensored_augmented_: Optional[pd.DataFrame] = None
        self.theta_path_: list[np.ndarray] = []

    def _init_ipcw(self, data: pd.DataFrame) -> WeightedCVXPYEstimator:
        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
        self.km_ = km
        Sc = km.predict
        w = compute_ipcw_weights(
            T=data["T"].values,
            Delta=data["Delta"].values,
            S_c_predict=Sc,
        )
        uncensored_mask = (data["Delta"].values == 1)
        df_unc = pd.DataFrame({"W1": data.loc[uncensored_mask, "T"].values})
        w_unc = w[uncensored_mask]
        init_est = WeightedCVXPYEstimator(
            tol=self.tol,
            norm_constraint=self.init_norm_constraint,
            n_grid_points=self.n_grid_points,
            basis_order=self.basis_order,
            log_dir=self.log_dir,
            log_frequency=self.log_frequency,
            use_secondary_solver=False,  # Disable waterfall - just use specified solver
            solver=self.init_solver,
            legacy_mode=True,
            include_intercept_in_constraint=True,
        ).fit(df_unc, sample_weights=w_unc)
        return init_est

    def fit(self, data: pd.DataFrame) -> "EMIPCWEstimator":
        """
        Args:
            data: DataFrame with columns ['T','Delta'] (times in [0,1]).
        """
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")
        # Init
        if self.verbose:
            print("Initializing IPCW-HAL-MLE...")
        init_est = self._init_ipcw(data)
        self._current_estimator = init_est
        self.theta_subset_, self.active_knots_ = self._prune_theta(
            init_est.theta_hat,
            init_est._grid_points_hal,
        )
        theta_basis_grid_points = self.active_knots_.copy()
        self.theta_path_.append(self.theta_subset_.copy())

        prev_ll = incomplete_loglik(init_est, data, time_col="T", delta_col="Delta")
        if self.verbose:
            print(f"Initial incomplete-data log-likelihood: {prev_ll:.4f}")

        for em_iter in range(self.max_em_iter):
            import time as _time
            _t0 = _time.time()
            pooled = e_step_multiple_imputation(
                data=data,
                theta_hat=self.theta_subset_,
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

            # M-step: HAL-MLE on pooled pseudo-complete data (unweighted)
            _t1 = _time.time()
            mstep_est = solve_legacy_m_step(
                pooled_df=pooled[["W1", "weight"]].copy(),
                knots=theta_basis_grid_points,
                norm_constraint=self.m_step_norm_constraint,
                warm_start_theta=self.theta_subset_,
                tol=self.tol,
                solver_sequence=self.m_step_solver_sequence,
                n_eval_grid=self.n_grid_points,
            )
            m_time = _time.time() - _t1
            self._current_estimator = mstep_est
            self.theta_subset_, self.active_knots_ = self._prune_theta(
                mstep_est.theta_hat,
                mstep_est._grid_points_hal,
            )
            theta_basis_grid_points = self.active_knots_.copy()
            self.theta_path_.append(self.theta_subset_.copy())

            # Check improvement on incomplete-data log-likelihood
            curr_ll = incomplete_loglik(mstep_est, data, time_col="T", delta_col="Delta")
            ll_diff = np.abs(curr_ll - prev_ll)
            if self.verbose:
                print(f"Iter {em_iter+1}: LL={curr_ll:.4f}, Δ={ll_diff:.6f}, E-step={e_time:.3f}s, M-step={m_time:.3f}s")
            if ll_diff < self.em_tol:
                if self.verbose:
                    print(f"\nConverged at iteration {em_iter+1}: LL diff {ll_diff:.6f} < tol {self.em_tol}")
                init_est = mstep_est
                break
            prev_ll = curr_ll
            init_est = mstep_est

        # Copy standardized fields from final estimator
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

    def get_results(self) -> dict:
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results.")
        base = self._get_common_results()
        base.update({
            "theta_path": [theta.tolist() for theta in self.theta_path_],
            "has_km": self.km_ is not None,
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
        coef = theta[1:]
        mask = np.abs(coef) > self.tol
        if not np.any(mask):
            pruned = np.zeros(1 + knots.size)
            pruned[0] = theta[0]
            pruned[1:] = coef
            return pruned, knots.copy()
        active_knots = knots[mask].copy()
        pruned = np.zeros(active_knots.size + 1, dtype=float)
        pruned[0] = theta[0]
        pruned[1:] = coef[mask]
        return pruned, active_knots
