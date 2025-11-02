HALDensity: Highly Adaptive Lasso Density Estimation (HAL-MLE/TMLE)
===============================================================

HALDensity provides fast, principled 1D density estimation using Highly Adaptive Lasso (HAL) and optional targeting (TMLE) for functionals such as survival, moments, and the median. It includes multiple optimization backends (first-order, second-order, and conic optimization via CVXPY), a unified base estimator, cross-validation with Optuna, and tools for uncertainty quantification.

Features
- HAL basis with polynomial and truncated power terms (order 0 or k ≥ 1)
- Multiple solvers: CVX (conic), Proximal Newton (and LBFGS variants), FISTA, Projected/Proximal GD
- Standardized estimator interface with a lightweight `BaseEstimator`
- Pydantic `CommonEstimatorResults` schema for uniform results
- Cross-validation and hyperparameter tuning via Optuna
- Delta-method style density bands; bootstrap example included
- Targeted learning (TMLE) for survival, mean, and median

Install
This repo uses Python 3.11+. With uv:
```bash
uv sync
```
Or with pip:
```bash
pip install -e .
```

Quickstart
```python
import numpy as np
import pandas as pd
from src.haldensity.estimation import CVXPYEstimator

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
from src.haldensity.cross_validation.optuna_hyperparam_selector import OptunaHyperparameterTuner
tuner = OptunaHyperparameterTuner(
    'CVXPYEstimator', data, cv_folds=5, metric='sll', silent=True,
    param_overrides={"basis_order": [0, 2]}
)
res = tuner.optimize(n_trials=50)
best_est = tuner.fit_best_model()
```

Targeted Learning (TMLE)
```python
from src.haldensity.targeting import SurvivalTargetLearner
learner = SurvivalTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=results['grid_points_hal_selected'],
    old_theta=np.array(results['theta_hat']),
    _old_theta=np.array(results['theta_hat']),
    _grid_points_hal_selected=results['grid_points_hal_selected'],
    targeting_points=np.linspace(0.02, 0.12, 10),
)
```

Examples
- `examples/example_density_estimator.ipynb`: quick demo of estimators
- `examples/example_target_learning.ipynb`: targeting examples
- `examples/example_case_study.ipynb`: real-data galaxy velocity case study

Repository Structure
```
src/haldensity/
  estimation/                 # Estimators and base class
  cross_validation/           # Optuna tuner and lambda selector
  targeting/                  # TMLE learners for survival/mean/median
  density_variance/           # Variance/band utilities
  utils/                      # Basis construction + math utilities
examples/                     # Notebooks and plotting helpers
local/setups/                 # JSON configs for reproducible runs
```

Notes
- Numerical stability: density normalization uses midpoint Riemann sums with log-exp stabilization.
- Estimators standardize outputs via `BaseEstimator._get_common_results()` and `CommonEstimatorResults`.
- All math uses NumPy/SciPy; PyTorch has been removed.

License
MIT




