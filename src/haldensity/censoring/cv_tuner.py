import numpy as np
import pandas as pd
from typing import Optional, Any
import optuna
from sklearn.model_selection import KFold
from .em_estimator import EMIPCWEstimator
from .weighted_cvxpy_estimator import WeightedCVXPYEstimator
from .km import KaplanMeier
from .weights import compute_ipcw_weights
from .metrics import incomplete_loglik


ESTIMATORS = {
    "EMIPCWEstimator": EMIPCWEstimator,
    "WeightedCVXPYEstimator": WeightedCVXPYEstimator,
}


class CensoredOptunaHyperparameterTuner:
    """
    Censoring-aware CV tuner supporting:
      - EMIPCWEstimator
      - WeightedCVXPYEstimator (IPCW baseline)
    Metrics:
      - 'incomplete' (sum Δ log f + (1-Δ) log S)
      - 'mi_complete' (placeholder, returns 0 unless user extends)
    """
    def __init__(
        self,
        estimator_name: str,
        data: pd.DataFrame,
        cv_folds: int = 3,
        metric: str = "incomplete",
        random_state: int = 42,
        n_grid_points: int = 200,
        param_overrides: Optional[dict[str, Any]] = None,
        silent: bool = True,
    ):
        if estimator_name not in ESTIMATORS:
            raise ValueError(f"Unsupported estimator '{estimator_name}'. Available: {list(ESTIMATORS.keys())}")
        if metric not in ["incomplete", "mi_complete"]:
            raise ValueError("metric must be 'incomplete' or 'mi_complete'")
        self.estimator_name = estimator_name
        self.estimator_class = ESTIMATORS[estimator_name]
        self.data = data.reset_index(drop=True)
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_state = random_state
        self.n_grid_points = n_grid_points
        self.param_overrides = param_overrides or {}
        self.silent = silent

        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        ovr = self.param_overrides
        # Defaults
        basis_order = ovr.get("basis_order", [0, 1, 2])
        norm_constraint = ovr.get("norm_constraint", {"low": 1e-3, "high": 1e3, "log": True})
        m_imputations = ovr.get("m_imputations", {"low": 5, "high": 50})
        max_em_iter = ovr.get("max_em_iter", {"low": 10, "high": 100})

        def choose_categorical(name, values):
            return trial.suggest_categorical(name, values) if isinstance(values, (list, tuple)) else values

        def choose_float(name, spec):
            if not isinstance(spec, dict):
                return spec
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

        def choose_int(name, spec):
            if not isinstance(spec, dict):
                return spec
            return trial.suggest_int(name, spec["low"], spec["high"])

        params = {
            "basis_order": choose_categorical("basis_order", basis_order),
            "norm_constraint": choose_float("norm_constraint", norm_constraint),
        }
        if self.estimator_name == "EMIPCWEstimator":
            params.update({
                "m_imputations": choose_int("m_imputations", m_imputations),
                "max_em_iter": choose_int("max_em_iter", max_em_iter),
            })
        return params

    def _evaluate(self, params: dict[str, Any]) -> float:
        scores = []
        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)

            if self.estimator_name == "WeightedCVXPYEstimator":
                # IPCW baseline on uncensored only
                km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
                w = compute_ipcw_weights(train_df["T"].values, train_df["Delta"].values, km.predict)
                unc = train_df.loc[train_df["Delta"] == 1, "T"].values
                w_unc = w[train_df["Delta"].values == 1]
                df_unc = pd.DataFrame({"W1": unc})
                est = self.estimator_class(
                    tol=1e-4,
                    norm_constraint=params["norm_constraint"],
                    n_grid_points=self.n_grid_points,
                    basis_order=params["basis_order"],
                ).fit(df_unc, sample_weights=w_unc)
            else:
                est = self.estimator_class(
                    tol=1e-4,
                    norm_constraint=params["norm_constraint"],
                    n_grid_points=self.n_grid_points,
                    basis_order=params["basis_order"],
                    m_imputations=params.get("m_imputations", 20),
                    max_em_iter=params.get("max_em_iter", 50),
                ).fit(train_df)

            if self.metric == "incomplete":
                score = incomplete_loglik(est, val_df, time_col="T", delta_col="Delta")
            else:
                score = 0.0  # placeholder
            scores.append(score)

        # Optuna minimizes; we want to maximize log-likelihood
        return -float(np.mean(scores))

    def optimize(self, n_trials: int = 50) -> dict[str, Any]:
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params = self.study.best_params.copy()
        self.best_metric_value = -self.study.best_value
        if not self.silent:
            print("Best params:", self.best_params)
            print("Best metric:", self.best_metric_value)
        return {"best_params": self.best_params, "best_metric_value": self.best_metric_value, "study": self.study}

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        return self._evaluate(params)

    def fit_best_model(self) -> Any:
        if self.best_params is None:
            raise ValueError("Run optimize() first")
        params = self.best_params
        if self.estimator_name == "WeightedCVXPYEstimator":
            # Fit on full with IPCW weights
            km = KaplanMeier().fit(self.data, time_col="T", delta_col="Delta")
            w = compute_ipcw_weights(self.data["T"].values, self.data["Delta"].values, km.predict)
            unc = self.data.loc[self.data["Delta"] == 1, "T"].values
            w_unc = w[self.data["Delta"].values == 1]
            df_unc = pd.DataFrame({"W1": unc})
            est = self.estimator_class(
                tol=1e-4,
                norm_constraint=params["norm_constraint"],
                n_grid_points=self.n_grid_points,
                basis_order=params["basis_order"],
            ).fit(df_unc, sample_weights=w_unc)
            return est
        else:
            est = self.estimator_class(
                tol=1e-4,
                norm_constraint=params["norm_constraint"],
                n_grid_points=self.n_grid_points,
                basis_order=params["basis_order"],
                m_imputations=params.get("m_imputations", 20),
                max_em_iter=params.get("max_em_iter", 50),
            ).fit(self.data)
            return est


