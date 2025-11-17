# HALDensity Developer Guide

This document describes the internal structure of the `src/haldensity` package, the estimator base class, common result schema, and how to add a new estimator.

Architecture

```
haldensity/
  estimation/
    base_estimator.py          # BaseEstimator and CommonEstimatorResults (Pydantic)
    conic_optimization_method/
      cvxpy/estimator.py       # CVXPYEstimator (inherits BaseEstimator)
    first_order_method/        # FISTA, ProjectedGD, ProximalGD
    second_order_method/       # ProximalNewton variants, ProximalAdaGrad
  cross_validation/            # Optuna tuner, lambda selector
  targeting/                   # TMLE learners (survival, mean, median)
  density_variance/            # Variance and band utilities
  utils/                       # Basis, math helpers, DGPs
  censoring/                   # Right-censored pipelines, IPCW/EM estimators, Optuna tuner
```

BaseEstimator

- Provides logging, density evaluation, log-likelihood helpers, BIC.
- Implements numerically stable density normalization on a midpoint grid.
- Exposes standardized results via:
  - `_get_common_results() -> dict`
  - `get_common_results_model() -> CommonEstimatorResults`
- Subclasses must implement:
  - `fit(self, data: pd.DataFrame, **kwargs) -> BaseEstimator`
  - `get_results(self) -> dict` (can return `_get_common_results()` plus extras)

CommonEstimatorResults (Pydantic)

```python
class CommonEstimatorResults(BaseModel):
    fitted_theta_dict: Optional[dict[str, float]]
    theta_hat: list[float]
    data_points: Optional[list[float]]
    grid_points_hal_selected: Optional[list[float]]
    n_selected_knots: int
    estimated_density: list[float]
    grid_points: list[float]
    intercept: float
    hal_coeffs: list[float]
```

Use `get_common_results_model()` for serialization and consistent downstream consumption.

Adding a New Estimator

1. Create a module under `estimation/<your_method>/estimator.py`.
2. Subclass `BaseEstimator`.
3. In `__init__`, call `super().__init__(basis_order=..., log_dir=..., log_frequency=..., tol=...)` and store method-specific params.
4. Implement `fit`: construct basis on data and normalization grid midpoints; set:
   - `self.theta_hat` (1D numpy array)
   - `self._grid_points_hal` (knot points)
   - `self.grid_midpoints`, `self.delta_j` (for normalization)
   - `self.grid_points` (public evaluation grid)
   - `self.grid_points_hal_selected` (mask on truncated power coeffs)
   - `self.basis_names`, `self.fitted_theta_dict`, `self.is_fitted = True`
5. Implement `get_results` by returning `self._get_common_results()` and optionally merging method-specific extras.
6. Export your estimator from `estimation/__init__.py` and add a JSON config under `local/setups/` if desired.

Basis Functions

- Implemented in `utils/basis.py` as truncated power basis.
- Order 0: `{1, I(x ≥ ξ_j)}`; Order k≥1: `{1, x, …, x^k, (x-ξ_j)_+^k}`.

Numerical Notes

- Density normalization uses midpoint Riemann sums with log-exp stabilization.
- `calculate_density_at_points` accepts external normalization grids for consistency.

Targeting (TMLE)

- Learners in `targeting/` compute updated density estimates targeting functionals.
- They consume `old_theta` and `grid_points_hal_selected` from the estimator results.
