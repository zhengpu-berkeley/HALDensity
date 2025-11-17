# Censored Data Support (Right-Censoring on [0,1])

This subpackage adds right-censored density estimation on [0,1] using:
- IPCW–HAL–MLE initialization (weights 1 / S_c(T) for Δ=1),
- EM with Multiple Imputation (reusing the same λ in M-steps),
- Censoring-aware cross-validation with either incomplete-data or MI-pooled complete-data log-likelihood,
- Utilities for Kaplan–Meier estimation of the censoring survival S_c(t), IPCW weights, metrics (log-likelihoods, KL divergence), and thin pipelines that mirror the research notebooks.

APIs are designed to be additive and modular; no edits to existing estimators are required.

Main classes and functions:
- `KaplanMeier`: minimalist KM estimator for the censoring survival S_c(t).
- `compute_ipcw_weights`: Δ / max(S_c(T), clip).
- `WeightedCVXPYEstimator`: CVX-based HAL estimator with sample weights.
- `EMIPCWEstimator`: EM with MI; emits `uncensored_augmented_` for targeting.
- `CensoredOptunaHyperparameterTuner`: censoring-aware hyperparameter tuning.
- `metrics`: incomplete/MI-pooled log-likelihood and KL divergence.
- `pipelines`: wrappers `run_ipcw_hal_mle`, `run_em_ipcw_hal_mle`.

All estimators subclass `BaseEstimator` and return `_get_common_results()`-compatible payloads. Targeting works out-of-the-box using `uncensored_augmented_` (column `W1`).


