"""Base tuner classes and result types for censored density estimation.

Provides:
- TuningResult: NamedTuple for standardized tuner output
- BaseCensoredInitTuner: Abstract base for Stage 1 (Init) tuners
- BaseCensoredEMTuner: Abstract base for Stage 2 (EM) tuners
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold

from haldensity.censoring.core.models import EM_DEFAULTS, TUNER_DEFAULTS


class TuningResult(NamedTuple):
    """Result from hyperparameter tuning.
    
    Attributes
    ----------
    estimator : Any
        The fitted estimator instance with best hyperparameters.
    best_params : Dict[str, Any]
        Best hyperparameters found during tuning.
    metadata : Dict[str, Any]
        Additional metadata including CV results, study info, adjustment details.
    """
    estimator: Any
    best_params: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class OverSmoothInitRecord:
    """Record for an oversmooth initialization fit."""
    factor: float
    norm_constraint: float
    estimator: Any
    n_knots: int
    log_likelihood: float


@dataclass(frozen=True)
class OverSmoothEMRecord:
    """Record for an EM refinement run."""
    factor: float
    init_n_knots: int
    init_ll: float
    em_iterations: int
    em_converged: bool
    em_n_knots: int
    em_ll: float
    ll_gain: float
    em_estimator: Any


class BaseCensoredInitTuner(ABC):
    """Abstract base class for Stage 1 (Init) hyperparameter tuners.
    
    Provides shared CV tuning logic with Optuna and conservative adjustment.
    Subclasses implement data-specific methods for estimator creation and scoring.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data.
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.
    n_grid_points : int
        Number of grid points for density evaluation.
    param_overrides : dict | None
        Override default tunable parameter ranges.
    use_conservative_adjustment : bool
        Whether to apply conservative 1-SE adjustment.
    conservative_k_percent : float
        Threshold as percentage of SD for conservative adjustment.
    conservative_max_steps : int
        Maximum steps for conservative search.
    conservative_step_pct : float
        Step size as percentage of best value.
    silent : bool
        Whether to suppress progress output.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[Dict[str, Any]] = None,
        use_conservative_adjustment: bool = TUNER_DEFAULTS.use_conservative_adjustment,
        conservative_k_percent: float = TUNER_DEFAULTS.conservative_k_percent,
        conservative_max_steps: int = TUNER_DEFAULTS.conservative_max_steps,
        conservative_step_pct: float = TUNER_DEFAULTS.conservative_step_pct,
        silent: bool = True,
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.param_overrides = param_overrides or {}
        self.silent = bool(silent)
        
        self.use_conservative_adjustment = bool(use_conservative_adjustment)
        self.conservative_k_percent = float(conservative_k_percent)
        self.conservative_max_steps = int(conservative_max_steps)
        self.conservative_step_pct = float(conservative_step_pct)
        
        self.kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Results storage
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        self.optuna_params: Optional[Dict[str, Any]] = None
        self.conservative_params: Optional[Dict[str, Any]] = None
        self.adjustment_results: Optional[list] = None
        
        # Fixed defaults (non-tunable)
        self._defaults = {
            "tol": EM_DEFAULTS.tol,
            "solver": TUNER_DEFAULTS.solver,
            "use_secondary_solver": TUNER_DEFAULTS.use_secondary_solver,
        }
    
    @abstractmethod
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest tunable parameters for the trial."""
        pass
    
    @abstractmethod
    def _evaluate_fold(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> float:
        """Evaluate parameters on a single CV fold. Returns log-likelihood."""
        pass
    
    @abstractmethod
    def _fit_final_estimator(self, params: Dict[str, Any]) -> Any:
        """Fit final estimator on full data with given parameters."""
        pass
    
    def _evaluate_folds(self, params: Dict[str, Any]) -> list[float]:
        """Evaluate parameters across all CV folds."""
        scores = []
        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)
            try:
                score = self._evaluate_fold(train_df, val_df, params)
                scores.append(score)
            except Exception:
                scores.append(float("-inf"))
        return scores
    
    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters via CV. Returns negated mean for Optuna."""
        scores = self._evaluate_folds(params)
        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")
    
    def _compute_cv_score_with_sd(
        self, 
        norm_constraint: float, 
        basis_order: int
    ) -> Tuple[float, float, list[float]]:
        """Compute CV score and SD for given parameters."""
        params = {"norm_constraint": norm_constraint, "basis_order": basis_order}
        fold_scores = self._evaluate_folds(params)
        valid_scores = [s for s in fold_scores if np.isfinite(s)]
        mean_score = float(np.mean(valid_scores)) if valid_scores else float("-inf")
        sd_score = float(np.std(valid_scores, ddof=1)) if len(valid_scores) >= 2 else 0.0
        return mean_score, sd_score, fold_scores
    
    def _apply_conservative_adjustment(self) -> Dict[str, Any]:
        """Apply conservative 1-SE adjustment to reduce oversmoothing."""
        if self.optuna_params is None:
            raise ValueError("Optuna optimization must run first")
        
        optuna_nc = float(self.optuna_params["norm_constraint"])
        basis_order = int(self.optuna_params["basis_order"])
        
        optuna_cv_ll, optuna_cv_sd, _ = self._compute_cv_score_with_sd(optuna_nc, basis_order)
        threshold = optuna_cv_ll - self.conservative_k_percent * optuna_cv_sd
        
        step_size = self.conservative_step_pct * optuna_nc
        best_conservative_nc = optuna_nc
        
        self.adjustment_results = [{
            "step": 0,
            "norm_constraint": optuna_nc,
            "cv_ll": optuna_cv_ll,
            "cv_sd": optuna_cv_sd,
            "above_threshold": True,
        }]
        
        for step in range(1, self.conservative_max_steps + 1):
            candidate_nc = optuna_nc - step * step_size
            if candidate_nc <= 0:
                break
            cand_cv_ll, cand_cv_sd, _ = self._compute_cv_score_with_sd(candidate_nc, basis_order)
            above_threshold = cand_cv_ll >= threshold
            self.adjustment_results.append({
                "step": step,
                "norm_constraint": candidate_nc,
                "cv_ll": cand_cv_ll,
                "cv_sd": cand_cv_sd,
                "above_threshold": above_threshold,
            })
            if above_threshold:
                best_conservative_nc = candidate_nc
            else:
                break
        
        self.conservative_params = {
            "norm_constraint": float(best_conservative_nc),
            "basis_order": int(basis_order),
        }
        return self.conservative_params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = self._suggest_params(trial)
        return self._evaluate(params)
    
    def optimize(self, n_trials: int = 50) -> TuningResult:
        """Run Optuna optimization with optional conservative adjustment.
        
        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.
            
        Returns
        -------
        TuningResult
            Tuple of (estimator, best_params, metadata).
        """
        from tqdm import tqdm
        
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        
        if not self.silent:
            progress = tqdm(total=n_trials, desc="Init Tuner CV", unit="trial")
            best_metric = {"value": float("-inf")}
            
            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                metric = -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                progress.set_postfix({
                    "LL": f"{metric:.2f}",
                    "best": f"{best_metric['value']:.2f}",
                })
            
            try:
                self.study.optimize(
                    self._objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                    callbacks=[update_progress],
                )
            finally:
                progress.close()
        else:
            self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=False)
        
        self.optuna_params = self.study.best_params.copy()
        self.best_metric_value = -self.study.best_value
        
        if self.use_conservative_adjustment:
            self._apply_conservative_adjustment()
            self.best_params = self.conservative_params
        else:
            self.conservative_params = None
            self.adjustment_results = None
            self.best_params = self.optuna_params
        
        # Fit final estimator
        estimator = self._fit_final_estimator(self.best_params)
        
        metadata = {
            "optuna_params": self.optuna_params,
            "conservative_params": self.conservative_params,
            "best_metric_value": self.best_metric_value,
            "study": self.study,
            "adjustment_results": self.adjustment_results,
        }
        
        return TuningResult(
            estimator=estimator,
            best_params=self.best_params,
            metadata=metadata,
        )


class BaseCensoredEMTuner(ABC):
    """Abstract base class for Stage 2 (EM) hyperparameter tuners.
    
    Supports two modes via `do_over_smooth`:
    - True (default): Grid search over oversmooth factors + EM refinement
    - False: CV-based tuning of m_step_norm_multiplier
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data.
    stage1_estimator : Any
        Fitted Stage 1 estimator to use as initialization.
    random_state : int
        Random seed for reproducibility.
    n_grid_points : int
        Number of grid points for density evaluation.
    do_over_smooth : bool
        If True, use oversmooth grid search. If False, use CV tuning.
    oversmooth_factors : Iterable[float] | None
        Factors for oversmooth grid (used when do_over_smooth=True).
    cv_folds : int
        Number of CV folds (used when do_over_smooth=False).
    em_m_imputations : int
        Number of imputations for EM E-step.
    em_max_em_iter : int
        Maximum EM iterations.
    em_tol : float
        EM convergence tolerance.
    em_norm_factor : float
        Factor to apply to Stage 1 norm_constraint for EM M-step.
    selection : str
        Selection criterion for oversmooth: 'em_ll' or 'll_gain'.
    silent : bool
        Whether to suppress progress output.
    solver : str | None
        Solver to use. If None, inherits from stage1_estimator.
    use_secondary_solver : bool | None
        Whether to use secondary solver. If None, inherits from stage1_estimator.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        stage1_estimator: Any,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        do_over_smooth: bool = True,
        oversmooth_factors: Optional[list[float]] = None,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        em_m_imputations: int = EM_DEFAULTS.m_imputations,
        em_max_em_iter: int = 20,
        em_tol: float = EM_DEFAULTS.em_tol,
        em_norm_factor: float = 1.0,
        em_e_step_n_grid: int = EM_DEFAULTS.e_step_n_grid,
        em_use_sc_adjustment: bool = EM_DEFAULTS.use_sc_adjustment,
        selection: str = "em_ll",
        silent: bool = True,
        solver: Optional[str] = None,
        use_secondary_solver: Optional[bool] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.stage1_estimator = stage1_estimator
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.do_over_smooth = bool(do_over_smooth)
        self.cv_folds = int(cv_folds)
        self.silent = bool(silent)
        
        # Extract parameters from Stage 1 estimator
        self.base_norm_constraint = float(getattr(
            stage1_estimator, "norm_constraint",
            getattr(stage1_estimator, "_norm_constraint", 100.0)
        ))
        self.basis_order = int(stage1_estimator.basis_order)
        
        # Solver inheritance
        if solver is not None:
            self.solver = str(solver)
        else:
            self.solver = str(getattr(stage1_estimator, "solver", TUNER_DEFAULTS.solver))
        
        if use_secondary_solver is not None:
            self.use_secondary_solver = bool(use_secondary_solver)
        else:
            self.use_secondary_solver = bool(getattr(
                stage1_estimator, "use_secondary_solver", 
                TUNER_DEFAULTS.use_secondary_solver
            ))
        
        # Oversmooth grid
        if oversmooth_factors is None:
            self.oversmooth_factors = [float(x) for x in np.linspace(0.5, 1.0, 11)]
        else:
            self.oversmooth_factors = [float(x) for x in oversmooth_factors]
        
        # EM configuration
        self.em_m_imputations = int(em_m_imputations)
        self.em_max_em_iter = int(em_max_em_iter)
        self.em_tol = float(em_tol)
        self.em_norm_factor = float(em_norm_factor)
        self.em_e_step_n_grid = int(em_e_step_n_grid)
        self.em_use_sc_adjustment = bool(em_use_sc_adjustment)
        self.selection = str(selection)
        
        if self.selection not in ("em_ll", "ll_gain"):
            raise ValueError("selection must be one of {'em_ll', 'll_gain'}")
        
        # Results storage
        self.init_records: Optional[list[OverSmoothInitRecord]] = None
        self.selected_factors: Optional[list[float]] = None
        self.em_records: Optional[list[OverSmoothEMRecord]] = None
        self.best_em_record: Optional[OverSmoothEMRecord] = None
        self.best_params: Optional[Dict[str, Any]] = None
        
        # For CV mode
        self.study: Optional[optuna.Study] = None
    
    @abstractmethod
    def _fit_init_estimator(self, norm_constraint: float) -> Any:
        """Fit an initialization estimator with given norm_constraint."""
        pass
    
    @abstractmethod
    def _compute_loglik(self, estimator: Any) -> float:
        """Compute log-likelihood for the given estimator."""
        pass
    
    @abstractmethod
    def _run_em_stage(
        self, 
        initial_estimator: Any, 
        m_step_norm_constraint: float
    ) -> Any:
        """Run EM stage and return the EM result."""
        pass
    
    @abstractmethod
    def _evaluate_cv_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        m_step_norm_multiplier: float,
    ) -> float:
        """Evaluate m_step_norm_multiplier on a CV fold (for do_over_smooth=False)."""
        pass
    
    def _fit_oversmooth_grid(self) -> list[OverSmoothInitRecord]:
        """Fit init estimators for all oversmooth factors."""
        factors = sorted(set(self.oversmooth_factors + [1.0]))
        records: list[OverSmoothInitRecord] = []
        
        for factor in factors:
            nc = self.base_norm_constraint * factor
            est = self._fit_init_estimator(nc)
            res = est.get_results()
            n_knots = len(res.get("grid_points_hal_selected", []))
            ll = self._compute_loglik(est)
            
            records.append(OverSmoothInitRecord(
                factor=float(factor),
                norm_constraint=nc,
                estimator=est,
                n_knots=int(n_knots),
                log_likelihood=float(ll),
            ))
        
        return records
    
    @staticmethod
    def _select_em_candidates(records: list[OverSmoothInitRecord]) -> list[float]:
        """Select candidate factors for EM refinement."""
        by_factor = {float(r.factor): r for r in records}
        if 1.0 not in by_factor:
            raise RuntimeError("Oversmooth grid must include baseline factor=1.0")
        
        baseline_knots = by_factor[1.0].n_knots
        strictly_smaller = [r for r in records if r.n_knots < baseline_knots]
        
        if strictly_smaller:
            unique_knot_counts = sorted({r.n_knots for r in strictly_smaller})
            selected: list[float] = []
            for k in unique_knot_counts:
                candidates = [r for r in strictly_smaller if r.n_knots == k]
                best = max(candidates, key=lambda r: r.factor)
                selected.append(float(best.factor))
            selected.append(1.0)
            return sorted(set(selected))
        
        equal_knot = [r for r in records if r.n_knots == baseline_knots and r.factor < 1.0]
        if equal_knot:
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
        init_records: list[OverSmoothInitRecord],
        selected_factors: list[float],
    ) -> list[OverSmoothEMRecord]:
        """Run EM refinement for selected candidates."""
        by_factor = {float(r.factor): r for r in init_records}
        m_step_norm_constraint = self.em_norm_factor * self.base_norm_constraint
        out: list[OverSmoothEMRecord] = []
        
        for factor in selected_factors:
            r = by_factor[float(factor)]
            em_result = self._run_em_stage(r.estimator, m_step_norm_constraint)
            em_est = em_result.final_estimator
            em_ll = self._compute_loglik(em_est)
            em_knots = len(em_est.get_results().get("grid_points_hal_selected", []))
            ll_gain = em_ll - float(r.log_likelihood)
            
            out.append(OverSmoothEMRecord(
                factor=float(factor),
                init_n_knots=int(r.n_knots),
                init_ll=float(r.log_likelihood),
                em_iterations=int(em_result.em_iterations),
                em_converged=bool(em_result.em_converged),
                em_n_knots=int(em_knots),
                em_ll=float(em_ll),
                ll_gain=float(ll_gain),
                em_estimator=em_est,
            ))
        
        return out
    
    @staticmethod
    def _pick_best_em(records: list[OverSmoothEMRecord], selection: str) -> OverSmoothEMRecord:
        """Pick best EM result based on selection criterion."""
        if not records:
            raise RuntimeError("No EM records to select from")
        if selection == "em_ll":
            return max(records, key=lambda r: r.em_ll)
        if selection == "ll_gain":
            return max(records, key=lambda r: r.ll_gain)
        raise ValueError("Unsupported selection metric")
    
    def _tune_with_oversmooth(self) -> TuningResult:
        """Run oversmooth grid search + EM refinement."""
        self.init_records = self._fit_oversmooth_grid()
        self.selected_factors = self._select_em_candidates(self.init_records)
        self.em_records = self._run_em_for_candidates(self.init_records, self.selected_factors)
        self.best_em_record = self._pick_best_em(self.em_records, self.selection)
        
        self.best_params = {
            "oversmooth_factor": self.best_em_record.factor,
            "em_norm_factor": self.em_norm_factor,
        }
        
        metadata = {
            "init_records": self.init_records,
            "selected_factors": self.selected_factors,
            "em_records": self.em_records,
            "best_em_record": self.best_em_record,
            "mode": "oversmooth",
        }
        
        return TuningResult(
            estimator=self.best_em_record.em_estimator,
            best_params=self.best_params,
            metadata=metadata,
        )
    
    def _tune_with_cv(self) -> TuningResult:
        """Run CV-based tuning of m_step_norm_multiplier."""
        from tqdm import tqdm
        
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        def evaluate(m_step_norm_multiplier: float) -> float:
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.data)):
                train_df = self.data.iloc[train_idx].reset_index(drop=True)
                val_df = self.data.iloc[val_idx].reset_index(drop=True)
                try:
                    score = self._evaluate_cv_fold(train_df, val_df, m_step_norm_multiplier)
                    scores.append(score)
                except Exception:
                    scores.append(float("-inf"))
            mean_score = float(np.mean(scores))
            return -mean_score if np.isfinite(mean_score) else float("inf")
        
        def objective(trial: optuna.Trial) -> float:
            mult = trial.suggest_float("m_step_norm_multiplier", 0.5, 1.0, log=True)
            return evaluate(mult)
        
        self.study = optuna.create_study(direction="minimize")
        
        if not self.silent:
            progress = tqdm(total=20, desc="EM Tuner CV", unit="trial")
            best_metric = {"value": float("-inf")}
            
            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                metric = -float(trial.value) if np.isfinite(trial.value) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                mult = trial.params.get("m_step_norm_multiplier", "?")
                mult_str = f"{mult:.2f}" if isinstance(mult, float) else str(mult)
                progress.set_postfix({"LL": f"{metric:.2f}", "mult": mult_str})
            
            try:
                self.study.optimize(objective, n_trials=20, show_progress_bar=False, callbacks=[update_progress])
            finally:
                progress.close()
        else:
            self.study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        best_mult = self.study.best_params["m_step_norm_multiplier"]
        m_step_norm_constraint = self.base_norm_constraint * best_mult
        
        # Fit final model
        em_result = self._run_em_stage(self.stage1_estimator, m_step_norm_constraint)
        
        self.best_params = {"m_step_norm_multiplier": best_mult}
        
        metadata = {
            "study": self.study,
            "best_metric_value": -self.study.best_value,
            "mode": "cv",
        }
        
        return TuningResult(
            estimator=em_result.final_estimator,
            best_params=self.best_params,
            metadata=metadata,
        )
    
    def optimize(self) -> TuningResult:
        """Run tuning and return best estimator.
        
        Returns
        -------
        TuningResult
            Tuple of (estimator, best_params, metadata).
        """
        if self.do_over_smooth:
            return self._tune_with_oversmooth()
        else:
            return self._tune_with_cv()
