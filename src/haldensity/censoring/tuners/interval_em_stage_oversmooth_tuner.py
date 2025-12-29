"""Stage-2 oversmooth (midpoint init) + EM refinement for interval-censored data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from haldensity.censoring.core.models import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.censoring.interval.midpoint_estimator import IntervalCensoredMidpointEstimator
from haldensity.censoring.interval.em_stage import IntervalCensoredEMStage
from haldensity.censoring.interval.metrics import incomplete_loglik_interval


@dataclass(frozen=True)
class OverSmoothMidpointRecord:
    factor: float
    norm_constraint: float
    estimator: Any
    n_knots: int
    log_likelihood: float


@dataclass(frozen=True)
class OverSmoothEMRecord:
    factor: float
    init_n_knots: int
    init_ll: float
    em_iterations: int
    em_converged: bool
    em_n_knots: int
    em_ll: float
    ll_gain: float
    em_estimator: Any


class IntervalCensoredEMStageOverSmoothTuner:
    """Stage-2: oversmooth midpoint initializations + single-run EM refinement."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        midpoint_params: dict[str, Any],
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        init_solver: str = TUNER_DEFAULTS.solver,
        init_use_secondary_solver: bool = TUNER_DEFAULTS.use_secondary_solver,
        oversmooth_factors: Optional[Iterable[float]] = None,
        em_norm_factor: float = 1.0,
        em_m_imputations: int = EM_DEFAULTS.m_imputations,
        em_max_em_iter: int = 20,
        em_tol: float = EM_DEFAULTS.em_tol,
        em_e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        em_verbose: bool = False,
        selection: str = "em_ll",
        silent: bool = True,
        L_col: str = "L",
        R_col: str = "R",
    ):
        if L_col not in data.columns or R_col not in data.columns:
            raise ValueError("data must contain columns 'L' and 'R'")
        if em_norm_factor <= 0:
            raise ValueError("em_norm_factor must be positive")
        if selection not in ("em_ll", "ll_gain"):
            raise ValueError("selection must be one of {'em_ll', 'll_gain'}")
        if "norm_constraint" not in midpoint_params or "basis_order" not in midpoint_params:
            raise ValueError("midpoint_params must contain keys {'norm_constraint', 'basis_order'}")

        self.data = data.reset_index(drop=True)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.midpoint_params = dict(midpoint_params)
        self.lambda_star = float(midpoint_params["norm_constraint"])
        self.basis_order_star = int(midpoint_params["basis_order"])
        self.init_solver = str(init_solver)
        self.init_use_secondary_solver = bool(init_use_secondary_solver)
        self.L_col = str(L_col)
        self.R_col = str(R_col)

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
        self.em_e_step_n_grid = int(em_e_step_n_grid)
        self.em_verbose = bool(em_verbose)
        self.selection = str(selection)
        self.silent = bool(silent)

        self.init_records: Optional[list[OverSmoothMidpointRecord]] = None
        self.selected_factors: Optional[list[float]] = None
        self.em_records: Optional[list[OverSmoothEMRecord]] = None
        self.best_em_record: Optional[OverSmoothEMRecord] = None

    @classmethod
    def from_midpoint_tuning_result(
        cls,
        data: pd.DataFrame,
        *,
        midpoint_tuning_result: dict[str, Any],
        **kwargs: Any,
    ) -> "IntervalCensoredEMStageOverSmoothTuner":
        midpoint_params = (
            midpoint_tuning_result.get("conservative_params")
            or midpoint_tuning_result.get("best_params")
            or midpoint_tuning_result.get("optuna_params")
            or {}
        )
        return cls(data, midpoint_params=midpoint_params, **kwargs)

    def fit_best_estimator(self) -> Any:
        self.init_records = self._fit_midpoint_oversmooth_grid(
            lambda_star=self.lambda_star, basis_order=self.basis_order_star
        )
        self.selected_factors = self._select_em_candidates(self.init_records)

        m_step_norm_constraint = self.em_norm_factor * self.lambda_star
        self.em_records = self._run_em_for_candidates(
            init_records=self.init_records,
            selected_factors=self.selected_factors,
            m_step_norm_constraint=m_step_norm_constraint,
        )
        self.best_em_record = self._pick_best_em(self.em_records, selection=self.selection)
        return self.best_em_record.em_estimator

    def _fit_midpoint_oversmooth_grid(self, *, lambda_star: float, basis_order: int) -> list[OverSmoothMidpointRecord]:
        factors = sorted(set(self.oversmooth_factors + [1.0]))
        records: list[OverSmoothMidpointRecord] = []
        for factor in factors:
            nc = float(lambda_star) * float(factor)
            est = IntervalCensoredMidpointEstimator(
                tol=EM_DEFAULTS.tol,
                norm_constraint=nc,
                n_grid_points=self.n_grid_points,
                basis_order=basis_order,
                solver=self.init_solver,
                use_secondary_solver=self.init_use_secondary_solver,
                include_intercept_in_constraint=False,
            ).fit(self.data, L_col=self.L_col, R_col=self.R_col)

            res = est.get_results()
            n_knots = len(res.get("grid_points_hal_selected", []))
            ll = float(incomplete_loglik_interval(est, self.data, L_col=self.L_col, R_col=self.R_col))

            records.append(
                OverSmoothMidpointRecord(
                    factor=float(factor),
                    norm_constraint=nc,
                    estimator=est,
                    n_knots=int(n_knots),
                    log_likelihood=ll,
                )
            )
        return records

    @staticmethod
    def _select_em_candidates(records: list[OverSmoothMidpointRecord]) -> list[float]:
        by_factor = {float(r.factor): r for r in records}
        if 1.0 not in by_factor:
            raise RuntimeError("Oversmooth grid must include baseline factor=1.0")

        baseline_knots = by_factor[1.0].n_knots
        strictly_smaller = [r for r in records if r.n_knots < baseline_knots]

        if len(strictly_smaller) > 0:
            unique_knot_counts = sorted({r.n_knots for r in strictly_smaller})
            selected: list[float] = []
            for k in unique_knot_counts:
                candidates = [r for r in strictly_smaller if r.n_knots == k]
                best = max(candidates, key=lambda r: r.factor)
                selected.append(float(best.factor))
            selected.append(1.0)
            return sorted(set(selected))

        equal_knot = [r for r in records if r.n_knots == baseline_knots and r.factor < 1.0]
        if len(equal_knot) > 0:
            return sorted(set([float(r.factor) for r in equal_knot] + [1.0]))

        all_factors = sorted({float(r.factor) for r in records})
        if len(all_factors) > 5:
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
        init_records: list[OverSmoothMidpointRecord],
        selected_factors: list[float],
        m_step_norm_constraint: float,
    ) -> list[OverSmoothEMRecord]:
        by_factor = {float(r.factor): r for r in init_records}
        out: list[OverSmoothEMRecord] = []

        for factor in selected_factors:
            r = by_factor[float(factor)]

            em_stage = IntervalCensoredEMStage(
                m_imputations=self.em_m_imputations,
                max_em_iter=self.em_max_em_iter,
                em_tol=self.em_tol,
                norm_constraint=float(m_step_norm_constraint),
                n_grid_points=self.n_grid_points,
                tol=EM_DEFAULTS.tol,
                m_step_solver=self.init_solver,
                verbose=self.em_verbose,
                rng_seed=self.random_state,
                e_step_n_grid=self.em_e_step_n_grid,
                L_col=self.L_col,
                R_col=self.R_col,
            )
            em_result = em_stage.run(initial_estimator=r.estimator, data=self.data)
            em_est = em_result.final_estimator

            em_ll = float(incomplete_loglik_interval(em_est, self.data, L_col=self.L_col, R_col=self.R_col))
            em_knots = len(em_est.get_results().get("grid_points_hal_selected", []))
            ll_gain = em_ll - float(r.log_likelihood)

            out.append(
                OverSmoothEMRecord(
                    factor=float(factor),
                    init_n_knots=int(r.n_knots),
                    init_ll=float(r.log_likelihood),
                    em_iterations=int(em_result.em_iterations),
                    em_converged=bool(em_result.em_converged),
                    em_n_knots=int(em_knots),
                    em_ll=float(em_ll),
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


