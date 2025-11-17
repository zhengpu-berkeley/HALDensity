# Censored Data Support (Right-Censoring on [0,1])

This subpackage ports the IPCW/EM notebooks (`ZO_CV_EM_IPCW_HAL_MLE.ipynb`, `1O_CV_EM_IPCW_HAL_MLE.ipynb`) into the modern HALDensity package without the heavyweight PyTorch dependency. Everything runs on NumPy/CVXPY while reusing the same HAL utilities documented in `censored_data_workflow.md` and `HALDENSITY_FIX_HISTORY.md`.

## Components

- `KaplanMeier`: minimalist Kaplan–Meier estimator for the censoring survival \(S_c(t)\).
- `weights.compute_ipcw_weights`: Δ / max(\(S_c(T)\), clip) IPCW weights.
- `WeightedCVXPYEstimator`: IPCW-HAL-MLE initializer with per-sample weights, intercept-constraint option, and solver waterfalls.
- `EMIPCWEstimator`: EM with multiple imputation; stores pooled pseudo-complete data in `uncensored_augmented_` and reuses the weighted estimator for the M-step.
- `sampling`: vectorised E-step matching the truncated-normal imputation logic from the notebooks.
- `design_utils`: shared truncated-basis helpers used by every censored estimator (no dependency on `utils/basis.py`).
- `metrics`: incomplete-data log-likelihood, MI-pooled complete-data log-likelihood proxy, KL divergence, and density sanity checks.
- `pipelines`: `run_ipcw_hal_mle` / `run_em_ipcw_hal_mle` return standardized HAL payloads (and optionally the estimator instances).
- `CensoredOptunaHyperparameterTuner`: Optuna CV tuner that exposes all notebook knobs (`use_sc_adjustment`, solver selections, EM tolerances) and lets you optimize either incomplete-data or MI-complete log-likelihood.

## Quickstart

```python
import pandas as pd
from haldensity.censoring import pipelines, CensoredOptunaHyperparameterTuner

# Right-censored data with columns ['T', 'Delta'] scaled to [0, 1]
data = pd.DataFrame({"T": T_tilde, "Delta": delta})

# Fit EM–IPCW–HAL–MLE and keep the estimator for targeting
results, est = pipelines.run_em_ipcw_hal_mle(
    data=data,
    norm_constraint=350.0,
    m_imputations=50,
    max_em_iter=5,
    return_estimator=True,
)

# Tune hyperparameters on the incomplete-data log-likelihood
tuner = CensoredOptunaHyperparameterTuner(
    "EMIPCWEstimator",
    data=data,
    metric="incomplete",                  # or "mi_complete"
    param_overrides={
        "basis_order": [0],
        "norm_constraint": {"low": 10, "high": 500, "log": True},
        "use_sc_adjustment": {"choices": [False, True]},
        "init_solver": {"choices": ["SCS", "ECOS"]},
    },
)
tuner.optimize(n_trials=25)
best_est = tuner.fit_best_model()
```

Both estimators inherit from `BaseEstimator`, so downstream tooling (`get_results`, targeting learners, variance utilities) works unchanged. When targeting censored fits, pass `est.uncensored_augmented_` (column `W1`) wherever an uncensored DataFrame is expected.

## Testing & Parity

- `gtimeout 60s uv run censor_data_comp_tests.py` is the primary acceptance test. It generates a truncated-normal DGP, runs IPCW → EM, checks that the incomplete-data log-likelihood increases each iteration, and asserts KL divergence < 0.1 for both IPCW and EM fits.
- Additional scripts (`test_censored_complete.py`, `test_em_only.py`, `test_minimal.py`, etc.) mirror the incremental experiments from the notebooks.
- `HALDENSITY_FIX_HISTORY.md` records every fix applied while porting the original workflow. Consult `HALDENSITY_CLEANUP_GUIDE.md` when you are ready to prune temporary scripts/notebooks.

The entire censored-data stack is additive—no changes were made to the uncensored estimators—so you can install HALDensity directly from GitHub and immediately use these APIs inside any scientific workflow.
