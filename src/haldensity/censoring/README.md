# Censored Data Density Estimation Module

This module provides HAL-based density estimation for censored data:

- **Right-censoring**: IPCW-weighted HAL-MLE and EM refinement
- **Interval-censoring**: Midpoint initialization and EM refinement

## Estimators

| Estimator | Description |
|-----------|-------------|
| `RightCensoredInitEstimator` | Stage 1 IPCW-weighted HAL for right-censored data |
| `RightCensoredEMEstimator` | Combined Stage 1 + Stage 2 EM for right-censored data |
| `RightCensoredEMStage` | Standalone EM refinement stage |
| `IntervalCensoredInitEstimator` | Stage 1 midpoint-imputed HAL for interval-censored data |
| `IntervalCensoredEMEstimator` | Combined Stage 1 + Stage 2 EM for interval-censored data |
| `IntervalCensoredEMStage` | Standalone EM refinement stage |

## Tuners

| Tuner | Description |
|-------|-------------|
| `RightCensoredInitTuner` | Stage 1 tuner (Optuna CV) |
| `RightCensoredEMTuner` | Stage 2 tuner (oversmooth or CV mode) |
| `RightCensoredJointTuner` | Convenience wrapper (Stage 1 + Stage 2) |
| `IntervalCensoredInitTuner` | Stage 1 tuner for interval-censored data |
| `IntervalCensoredEMTuner` | Stage 2 tuner for interval-censored data |
| `IntervalCensoredJointTuner` | Convenience wrapper for interval-censored data |

## IPCW-EM-HAL-MLE Workflow for Right-Censored Data

```mermaid
flowchart TD
  A["Start: Univariate right-censored data (T_i, Delta_i)"] --> B["Estimate censoring S_c via Kaplan-Meier"]
  B --> C["Choose lambda-grid"]
  C --> D{"Loop over lambda"}

  D --> E["Initial estimator: IPCW-HAL-MLE with weights 1/S_c for Delta=1"]
  E --> F0["Initialize theta from IPCW estimator"]
  F0 --> G{"Converged?"}
  G -->|No| H1["E-step: Multiple Imputation of T for Delta=0"]
  H1 --> H2["Form m pseudo full-data sets; pool sufficient stats"]
  H2 --> H3["M-step: HAL-MLE on pseudo full-data with SAME lambda"]
  H3 --> H4["k <- k+1"]
  H4 --> G
  G -->|Yes| I["theta_EM at lambda"]

  I --> J["Compute CV risk for lambda"]
  J --> K{"All lambda done?"}
  K -->|No| D
  K -->|Yes| L["Select lambda* = argmin CVRisk"]
  L --> M["Refit EM at lambda* on full data"]
  M --> N["Output: theta_EM, density, survival, hazard"]
  N --> O[End]
```

## Module Structure

```
censoring/
  __init__.py           # Main exports
  _defaults.py          # EMDefaults, TunerDefaults, EMStageResult
  _base_mle.py          # WeightedHALMLEEstimator (shared)
  README.md             # This file
  right/
    __init__.py
    estimators.py       # RC estimators
    km.py               # KaplanMeier
    weights.py          # IPCW weights
    metrics.py          # RC metrics
  interval/
    __init__.py
    estimators.py       # IC estimators
    metrics.py          # IC metrics
  tuners/
    __init__.py
    _base.py            # Base tuner classes
    _utils.py           # Tuner helpers
    right_tuners.py     # RC tuners
    interval_tuners.py  # IC tuners
  utils/
    __init__.py
    common_metrics.py   # kl_divergence
```
