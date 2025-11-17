# HALDensity: Highly Adaptive Lasso Density Estimation (HAL-MLE/TMLE)

HALDensity provides fast, principled 1D density estimation using Highly Adaptive Lasso (HAL) and optional targeting (TMLE) for functionals such as survival, moments, and the median. It includes multiple optimization backends (first-order, second-order, and conic optimization via CVXPY), a unified base estimator, cross-validation with Optuna, and tools for uncertainty quantification.

Features

- HAL basis with polynomial and truncated power terms (order 0 or k ≥ 1)
- Multiple solvers: CVX (conic), Proximal Newton (and LBFGS variants), FISTA, Projected/Proximal GD
- Standardized estimator interface with a lightweight `BaseEstimator`
- Pydantic `CommonEstimatorResults` schema for uniform results
- Cross-validation and hyperparameter tuning via Optuna
- Delta-method style density bands; bootstrap example included
- Targeted learning (TMLE) for survival, mean, and median
- Torch-free right-censored density estimation via IPCW initialization, EM + MI, and censoring-aware Optuna tuning

Install
This repo uses Python 3.11+. With uv:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

Or install directly from GitHub:

```bash
pip install "git+https://github.com/zhengpu-berkeley/HALDensity.git"
```

Quickstart

```python
import numpy as np
import pandas as pd
from haldensity.estimation import CVXPYEstimator

# Simulated data on [0, 1]
rng = np.random.default_rng(0)
data = pd.DataFrame({"W1": np.clip(rng.normal(0.07, 0.02, size=2000), 0, 1)})

# Fit CVX-based estimator
est = CVXPYEstimator(basis_order=0, norm_constraint=20, n_grid_points=200)
est.fit(data)

# Evaluate density on a fine grid
grid = np.linspace(0, 1, 1000)
density = est.get_density_at_points(grid)

# Results (standardized)
results = est.get_results()
# Or a Pydantic model:
model = est.get_common_results_model()
```

Cross-Validation (Optuna)

```python
from haldensity.cross_validation.optuna_hyperparam_selector import OptunaHyperparameterTuner
tuner = OptunaHyperparameterTuner(
    'CVXPYEstimator', data, cv_folds=5, metric='sll', silent=True,
    param_overrides={"basis_order": [0, 2]}
)
res = tuner.optimize(n_trials=50)
best_est = tuner.fit_best_model()
```

Targeted Learning (TMLE)

```python
# Common prerequisites (after fitting an estimator)
import numpy as np
from haldensity.targeting import (
    CDFTargetLearner, SurvivalTargetLearner, MomentsTargetLearner, MedianTargetLearner,
    cdf_estimand_variance, survival_estimand_variance, moments_estimand_variance, median_estimand_variance,
)

results = est.get_results()
old_theta = np.array(results["theta_hat"])                     # intercept + HAL coefs
hal_grid = results["grid_points_hal_selected"]                 # selected HAL grid points
```

# 1) CDF targeting (estimate F(x) at chosen points)

```python
import numpy as np

cdf_points = np.linspace(0.02, 0.98, 10)
learner = CDFTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    targeting_points=cdf_points,   # required
)
# Variance (SE^2) at the same points
var = cdf_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=cdf_points)
se = np.sqrt(var)
```

# 2) Survival targeting (estimate S(x)=1-F(x) at chosen points)

```python
surv_points = np.linspace(0.02, 0.98, 10)
learner = SurvivalTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    targeting_points=surv_points,  # required
)
var = survival_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=surv_points)
se = np.sqrt(var)
```

# 3) Moments targeting (estimate E[X^k], e.g., k=2)

```python
k = 2  # 1 for mean, 2 for second moment, etc.
learner = MomentsTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    x_moment=k,                    # required
)
var = moments_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, x_moment=k)
se = np.sqrt(var)[0]  # single scalar
```

# 4) Median targeting (estimate the median)

```python
learner = MedianTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    # no extra kwargs required
)
var = median_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data)
se = np.sqrt(var)[0]  # single scalar
```

Notes

- The example notebook `examples/example_target_learning.ipynb` demonstrates all four targets. Set `TARGET_TYPE` to one of "CDF", "Survival", "Moments", or "Median" to switch modes.
- For CDF/Survival, provide `targeting_points` where the functional is targeted and where standard errors are reported.
- For Moments, provide `x_moment` (integer power).
- For Median, no additional arguments are needed beyond `old_theta` and the selected HAL grid.

Examples

- `examples/example_density_estimator.ipynb`: quick demo of estimators
- `examples/example_target_learning.ipynb`: targeting examples
- `examples/example_case_study.ipynb`: real-data galaxy velocity case study
- `examples/example_censored_pipeline.py`: minimal right-censored HAL workflow (pipelines + Optuna)

### Censored Data (Right-Censoring on [0,1])

The `haldensity.censoring` subpackage ports the IPCW/EM notebooks to the modern codebase while reusing the same HAL utilities as the uncensored estimators:

- `KaplanMeier` and `compute_ipcw_weights` estimate the censoring survival \(S_c\) and Δ / \(S_c(T)\) weights.
- `WeightedCVXPYEstimator` reproduces the IPCW-HAL-MLE initializer (per-sample weights, intercept constraint option, solver waterfall).
- `EMIPCWEstimator` reuses the initializer, performs multiple-imputation E-steps, and runs a weighted HAL M-step via the shared estimator stack.
- `CensoredOptunaHyperparameterTuner` lets you choose between incomplete-data log-likelihood and MI-pooled complete-data log-likelihood and exposes the knobs used in the notebooks (`use_sc_adjustment`, solver selections, EM tolerances, etc.).
- `haldensity.censoring.pipelines` packages the workflows into `run_ipcw_hal_mle` / `run_em_ipcw_hal_mle`, returning standardized HAL payloads (and optionally the fitted estimator) so notebooks/tests can focus on analysis rather than boilerplate.

Example:

```python
from haldensity.censoring import pipelines, CensoredOptunaHyperparameterTuner

# Fit EM–IPCW–HAL–MLE
results, est = pipelines.run_em_ipcw_hal_mle(
    data=censored_df, norm_constraint=350, m_imputations=50,
    max_em_iter=5, return_estimator=True,
)

# Tune hyperparameters with Optuna on incomplete-data log-likelihood
tuner = CensoredOptunaHyperparameterTuner(
    "EMIPCWEstimator",
    data=censored_df,
    metric="incomplete",       # or "mi_complete"
    param_overrides={"use_sc_adjustment": {"choices": [False, True]}},
)
tuner.optimize(n_trials=25)
best_est = tuner.fit_best_model()
```

`metric="incomplete"` optimizes the Δ log f + (1−Δ) log S criterion used in the notebooks,
while `metric="mi_complete"` switches to the MI-pooled complete-data proxy without changing
any other tuner code.

Validation:

- `gtimeout 60s uv run censor_data_comp_tests.py` exercises the full pipeline on a truncated-normal DGP and checks KL divergence (< 0.1) plus monotone log-likelihood growth.
- Additional smoke tests (`test_censored_complete.py`, `test_em_only.py`, etc.) and the historical notebooks remain for parity; once satisfied, follow `HALDENSITY_CLEANUP_GUIDE.md` if you want to remove temporary assets.
- The regression script accepts CLI overrides (`python censor_data_comp_tests.py --help`) so CI can dial down runtime or stress-test different settings.

See `src/haldensity/censoring/README.md` for module-by-module documentation, references back to `censored_data_workflow.md`, and tips on reproducing the historical experiments without PyTorch.

Repository Structure

```

src/haldensity/
estimation/ # Estimators and base class
cross_validation/ # Optuna tuner and lambda selector
targeting/ # TMLE learners for survival/mean/median
density_variance/ # Variance/band utilities
utils/ # Basis construction + math utilities
examples/ # Notebooks and plotting helpers
local/setups/ # JSON configs for reproducible runs

```

License
MIT
