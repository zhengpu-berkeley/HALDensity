"""Stage-2 oversmooth + EM refinement (expects Stage-1 CV IPCW output).

This module provides a **second-stage** workflow that mirrors the "oversmooth IPCW → EM"
portion of `examples/example_oversmooth_ipcw_em.ipynb`, but **does not** run CV for IPCW.

Expected usage
--------------
1) Stage 1 (separate): run `RightCensoredOptunaHyperparameterTuner` for IPCW to obtain
   cross-validated IPCW parameters (ideally conservative-adjusted): `(basis_order*, λ*)`.
2) Stage 2 (this file): given `(basis_order*, λ*)`, fit a grid of over-smoothed IPCW
   initial estimators (factor × λ*), select candidates via the notebook's forward-grid
   heuristic, run a *single* `RightCensoredEMStage` on full data for each candidate, and
   return the best EM estimator.

Notes
-----
- The EM stage is **parametric EM** (knot structure fixed by initial estimator), so the
  only way to reduce model size is to oversmooth at IPCW initialization (smaller λ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from haldensity.censoring.core.models import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.ipcw_estimator import RightCensoredIPCWEstimator
from haldensity.censoring.right.em_stage import RightCensoredEMStage
from haldensity.censoring.right.metrics import incomplete_loglik


@dataclass(frozen=True)
class OverSmoothIPCWRecord:
    factor: float
    norm_constraint: float
    estimator: Any
    n_knots: int
    log_likelihood: float


@dataclass(frozen=True)
class OverSmoothEMRecord:
    factor: float
    ipcw_n_knots: int
    ipcw_ll: float
    em_iterations: int
    em_converged: bool
    em_n_knots: int
    em_ll: float
    ll_gain: float
    em_estimator: Any


class RightCensoredEMStageOverSmoothTuner:
    """Stage-2 oversmooth IPCW initializations + single-run EM refinement.

    This class expects you to provide a Stage-1 IPCW CV solution (e.g. conservative
    `(basis_order*, λ*)`) and then performs:
    - IPCW oversmooth grid fit on full data (uncensored subset with IPCW weights),
    - forward-grid candidate selection (by knot-count reduction),
    - single-run EM refinement on full data for each candidate,
    - selection of best candidate by `selection`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        # Stage-1 IPCW CV output
        ipcw_params: dict[str, Any],
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        # Over-smoothing exploration
        oversmooth_factors: Optional[Iterable[float]] = None,
        # EM refinement configuration (NO CV)
        em_norm_factor: float = 1.0,
        em_m_imputations: int = EM_DEFAULTS.m_imputations,
        em_max_em_iter: int = 20,
        em_tol: float = EM_DEFAULTS.em_tol,
        em_use_sc_adjustment: bool = EM_DEFAULTS.use_sc_adjustment,
        em_e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        em_m_step_solver: str = EM_DEFAULTS.m_step_solver,
        em_verbose: bool = False,
        # Selection
        selection: str = "em_ll",
        silent: bool = True,
    ):
        if "T" not in data.columns or "Delta" not in data.columns:
            raise ValueError("data must contain columns 'T' and 'Delta'")
        if em_norm_factor <= 0:
            raise ValueError("em_norm_factor must be positive")
        if selection not in ("em_ll", "ll_gain"):
            raise ValueError("selection must be one of {'em_ll', 'll_gain'}")
        if not isinstance(ipcw_params, dict):
            raise TypeError("ipcw_params must be a dict with keys {'norm_constraint', 'basis_order'}")
        if "norm_constraint" not in ipcw_params or "basis_order" not in ipcw_params:
            raise ValueError("ipcw_params must contain keys {'norm_constraint', 'basis_order'}")

        self.data = data.reset_index(drop=True)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.ipcw_params = dict(ipcw_params)
        self.lambda_star = float(ipcw_params["norm_constraint"])
        self.basis_order_star = int(ipcw_params["basis_order"])

        if oversmooth_factors is None:
            self.oversmooth_factors = [float(x) for x in np.linspace(0.5, 1.0, 11)]
        else:
            self.oversmooth_factors = [float(x) for x in oversmooth_factors]
        if len(self.oversmooth_factors) == 0:
            raise ValueError("oversmooth_factors must be non-empty")

        self.em_norm_factor = float(em_norm_factor)
        self.em_m_imputations = int(em_m_imputations)
        self.em_max_em_iter = int(em_max_em_iter)
        self.em_tol = float(em_tol)
        self.em_use_sc_adjustment = bool(em_use_sc_adjustment)
        self.em_e_step_n_grid = int(em_e_step_n_grid)
        self.em_m_step_solver = str(em_m_step_solver)
        self.em_verbose = bool(em_verbose)

        self.selection = selection
        self.silent = bool(silent)

        # Fitted state (optional to inspect)
        self.ipcw_records: Optional[list[OverSmoothIPCWRecord]] = None
        self.selected_factors: Optional[list[float]] = None
        self.em_records: Optional[list[OverSmoothEMRecord]] = None
        self.best_em_record: Optional[OverSmoothEMRecord] = None

    @classmethod
    def from_ipcw_tuning_result(
        cls,
        data: pd.DataFrame,
        *,
        ipcw_tuning_result: dict[str, Any],
        **kwargs: Any,
    ) -> "RightCensoredEMStageOverSmoothTuner":
        """Construct Stage-2 tuner from `RightCensoredOptunaHyperparameterTuner.optimize()` output.

        The tuner output may include:
        - `conservative_params` (preferred)
        - `best_params`
        - `optuna_params`
        """
        if not isinstance(ipcw_tuning_result, dict):
            raise TypeError("ipcw_tuning_result must be a dict returned by optimize()")

        ipcw_params = (
            ipcw_tuning_result.get("conservative_params")
            or ipcw_tuning_result.get("best_params")
            or ipcw_tuning_result.get("optuna_params")
            or {}
        )
        return cls(data, ipcw_params=ipcw_params, **kwargs)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit_best_estimator(self) -> Any:
        """Run Stage-2 and return the best EM estimator (final_estimator)."""

        # Fit KM once on full data; used both for IPCW weights and EM E-step adjustment.
        km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
        S_c_predict = lambda t: np.atleast_1d(km.predict(t))

        # Prepare IPCW training data (uncensored only) + weights on full data KM.
        T_vals = np.asarray(self.data["T"].values, dtype=float)
        Delta_vals = np.asarray(self.data["Delta"].values, dtype=int)
        w = compute_ipcw_weights(T_vals, Delta_vals, S_c_predict)
        uncensored_mask = Delta_vals == 1
        ipcw_df = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        ipcw_w = w[uncensored_mask]

        # Fit IPCW models for all factors.
        self.ipcw_records = self._fit_ipcw_oversmooth_grid(
            ipcw_df=ipcw_df,
            ipcw_weights=ipcw_w,
            lambda_star=self.lambda_star,
            basis_order=self.basis_order_star,
        )

        # Select candidates using notebook forward-grid logic.
        self.selected_factors = self._select_em_candidates(self.ipcw_records)

        # Run EM refinement for candidates (NO CV).
        m_step_norm_constraint = self.em_norm_factor * self.lambda_star
        self.em_records = self._run_em_for_candidates(
            ipcw_records=self.ipcw_records,
            selected_factors=self.selected_factors,
            m_step_norm_constraint=m_step_norm_constraint,
            S_c_predict=S_c_predict,
        )

        # Pick best according to selection
        self.best_em_record = self._pick_best_em(self.em_records, selection=self.selection)
        return self.best_em_record.em_estimator

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _fit_ipcw_oversmooth_grid(
        self,
        *,
        ipcw_df: pd.DataFrame,
        ipcw_weights: np.ndarray,
        lambda_star: float,
        basis_order: int,
    ) -> list[OverSmoothIPCWRecord]:
        # Ensure baseline 1.0 is always included for candidate selection.
        factors = sorted(set(self.oversmooth_factors + [1.0]))
        records: list[OverSmoothIPCWRecord] = []

        for factor in factors:
            nc = float(lambda_star) * float(factor)
            est = RightCensoredIPCWEstimator(
                tol=EM_DEFAULTS.tol,
                norm_constraint=nc,
                n_grid_points=self.n_grid_points,
                basis_order=basis_order,
                solver=TUNER_DEFAULTS.solver,
                use_secondary_solver=TUNER_DEFAULTS.use_secondary_solver,
            )
            est.fit(ipcw_df, sample_weights=ipcw_weights)

            res = est.get_results()
            n_knots = len(res.get("grid_points_hal_selected", []))
            ll = float(incomplete_loglik(est, self.data, time_col="T", delta_col="Delta"))

            records.append(
                OverSmoothIPCWRecord(
                    factor=float(factor),
                    norm_constraint=nc,
                    estimator=est,
                    n_knots=int(n_knots),
                    log_likelihood=ll,
                )
            )

        return records

    @staticmethod
    def _select_em_candidates(records: list[OverSmoothIPCWRecord]) -> list[float]:
        """Notebook-style forward grid search selection on knot-count reduction."""
        # Build a helper map factor -> record
        by_factor = {float(r.factor): r for r in records}
        if 1.0 not in by_factor:
            raise RuntimeError("Oversmooth grid must include baseline factor=1.0")

        baseline_knots = by_factor[1.0].n_knots
        strictly_smaller = [r for r in records if r.n_knots < baseline_knots]

        if len(strictly_smaller) > 0:
            # For each unique knot count, pick the one with highest factor.
            unique_knot_counts = sorted({r.n_knots for r in strictly_smaller})
            selected: list[float] = []
            for k in unique_knot_counts:
                candidates = [r for r in strictly_smaller if r.n_knots == k]
                best = max(candidates, key=lambda r: r.factor)  # highest factor
                selected.append(float(best.factor))
            selected.append(1.0)
            return sorted(set(selected))

        # Fallback A: models with equal knots but more regularization (factor < 1.0)
        equal_knot = [r for r in records if r.n_knots == baseline_knots and r.factor < 1.0]
        if len(equal_knot) > 0:
            selected = [float(r.factor) for r in equal_knot] + [1.0]
            return sorted(set(selected))

        # Fallback B: baseline + a few representative factors.
        all_factors = sorted({float(r.factor) for r in records})
        if len(all_factors) > 5:
            # Evenly spaced including endpoints.
            idx = np.linspace(0, len(all_factors) - 1, 5, dtype=int)
            selected = [all_factors[i] for i in idx]
        else:
            selected = list(all_factors)
        if 1.0 not in selected:
            selected.append(1.0)
        return sorted(set(selected))

    def _run_em_for_candidates(
        self,
        *,
        ipcw_records: list[OverSmoothIPCWRecord],
        selected_factors: list[float],
        m_step_norm_constraint: float,
        S_c_predict: Callable[[np.ndarray], np.ndarray],
    ) -> list[OverSmoothEMRecord]:
        by_factor = {float(r.factor): r for r in ipcw_records}
        out: list[OverSmoothEMRecord] = []

        for factor in selected_factors:
            r = by_factor[float(factor)]

            em_stage = RightCensoredEMStage(
                m_imputations=self.em_m_imputations,
                max_em_iter=self.em_max_em_iter,
                em_tol=self.em_tol,
                norm_constraint=float(m_step_norm_constraint),
                n_grid_points=self.n_grid_points,
                use_sc_adjustment=self.em_use_sc_adjustment,
                e_step_n_grid=self.em_e_step_n_grid,
                tol=EM_DEFAULTS.tol,
                m_step_solver=self.em_m_step_solver,
                verbose=self.em_verbose,
                rng_seed=self.random_state,
            )

            em_result = em_stage.run(
                initial_estimator=r.estimator,
                data=self.data,
                S_c_predict=S_c_predict,
            )
            em_est = em_result.final_estimator
            em_ll = float(incomplete_loglik(em_est, self.data, time_col="T", delta_col="Delta"))
            em_knots = len(em_est.get_results().get("grid_points_hal_selected", []))
            ll_gain = em_ll - float(r.log_likelihood)

            out.append(
                OverSmoothEMRecord(
                    factor=float(factor),
                    ipcw_n_knots=int(r.n_knots),
                    ipcw_ll=float(r.log_likelihood),
                    em_iterations=int(em_result.em_iterations),
                    em_converged=bool(em_result.em_converged),
                    em_n_knots=int(em_knots),
                    em_ll=em_ll,
                    ll_gain=float(ll_gain),
                    em_estimator=em_est,
                )
            )

        return out

    @staticmethod
    def _pick_best_em(records: list[OverSmoothEMRecord], *, selection: str) -> OverSmoothEMRecord:
        if len(records) == 0:
            raise RuntimeError("No EM records to select from")
        if selection == "em_ll":
            return max(records, key=lambda r: r.em_ll)
        if selection == "ll_gain":
            return max(records, key=lambda r: r.ll_gain)
        raise ValueError("Unsupported selection metric")

