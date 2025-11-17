# Notebook API Fix Summary

## Date: November 17, 2025

This document summarizes the fixes made to `example_censored_data_density.ipynb` to ensure API consistency with the `haldensity.censoring` module.

## Issues Found and Fixed

### 1. **Missing `em_iterations` and `em_converged` in EMIPCWEstimator Results** ✅

**Problem:**
- The notebook expected `em_iterations` and `em_converged` keys in the results dictionary from `EMIPCWEstimator.get_results()`
- These keys were not present in the implementation

**Solution:**
Modified `/src/haldensity/censoring/em_estimator.py`:
- Added instance attributes `self.em_iterations_` and `self.em_converged_` (initialized in `__init__`)
- Updated the `fit()` method to track:
  - `self.em_iterations_` increments each iteration
  - `self.em_converged_` set to `True` when convergence criterion is met
- Updated `get_results()` to include these fields in the returned dictionary

**Files Modified:**
- `/src/haldensity/censoring/em_estimator.py` (lines 66-67, 110-112, 153, 200-201)

---

### 2. **Incorrect `kl_divergence` Function Signature** ✅

**Problem:**
- Notebook used: `kl_divergence(estimated_density=..., true_density=..., grid=...)`
- Actual signature: `kl_divergence(true_pdf_fn, grid, est_density)`
- The function expects a callable that generates the true PDF, not a pre-computed array

**Solution:**
Fixed all calls to `kl_divergence` in the notebook to use:
```python
kl_divergence(
    true_pdf_fn=lambda x: truncnorm.pdf(x, a, b, loc=mean, scale=std),
    grid=eval_grid,
    est_density=density,
)
```

**Cells Modified:**
- Cell 9 (IPCW estimator)
- Cell 12 (EM estimator)
- Cell 16 (Tuned estimator)
- Cell 19 (Basis order comparison loop)

---

### 3. **Incorrect Result Dictionary Key** ✅

**Problem:**
- Notebook used: `ipcw_results['theta_hat_selected']`
- Actual key: `theta_hat`
- The results dictionary from `get_results()` does not have a `theta_hat_selected` key

**Solution:**
- Changed Cell 9 to use `ipcw_results['theta_hat']` instead

**Cells Modified:**
- Cell 9

---

## Verification

Created and ran a comprehensive test script (`test_censored_notebook.py`) that:
1. ✅ Imported all necessary modules successfully
2. ✅ Simulated right-censored data
3. ✅ Fit Kaplan-Meier estimator
4. ✅ Fit IPCW-HAL-MLE (confirmed correct keys in results)
5. ✅ Fit EM-IPCW-HAL-MLE (confirmed `em_iterations` and `em_converged` present)
6. ✅ Ran Optuna hyperparameter tuning
7. ✅ Tested pipeline convenience functions
8. ✅ Accessed augmented complete data
9. ✅ Computed MI-complete log-likelihood

All tests passed successfully with no errors.

---

## API Reference Summary

### Results Dictionary Keys

**From `BaseEstimator._get_common_results()`:**
- `fitted_theta_dict`: Dict mapping basis names to coefficients
- `theta_hat`: Full coefficient vector (numpy array)
- `data_points`: Grid points used for HAL basis
- `grid_points_hal_selected`: Selected knot locations
- `n_selected_knots`: Count of non-zero knot coefficients
- `estimated_density`: Density values on evaluation grid
- `grid_points`: Evaluation grid points
- `intercept`: Intercept coefficient (theta_hat[0])
- `hal_coeffs`: HAL basis coefficients (excluding intercept/polynomials)

**Additional keys from `EMIPCWEstimator.get_results()`:**
- `theta_path`: List of theta vectors from each EM iteration
- `has_km`: Boolean indicating if KM estimator was fitted
- `em_iterations`: Number of EM iterations completed
- `em_converged`: Boolean indicating convergence

### `kl_divergence` Function Signature

```python
def kl_divergence(
    true_pdf_fn: Callable[[np.ndarray], np.ndarray],  # Function that computes true PDF
    grid: np.ndarray,                                  # Grid points for integration
    est_density: np.ndarray,                           # Estimated density values
    eps: float = 1e-12                                 # Numerical stability parameter
) -> float:
    """Compute D_KL(true || est) ≈ Σ p_true(x) log(p_true(x) / p_est(x)) Δx"""
```

---

## Files Modified

1. **Source Code:**
   - `/src/haldensity/censoring/em_estimator.py`

2. **Notebooks:**
   - `/examples/example_censored_data_density.ipynb` (Cells 9, 12, 16, 19)

---

## Testing Recommendation

To verify the notebook works end-to-end, you can run:

```bash
# Quick acceptance test
uv run python censor_data_comp_tests.py

# Extended Optuna test
uv run python censor_optuna_tuner_tests.py
```

Or execute the notebook directly in Jupyter to ensure all cells run without errors.

---

## Notes

- All API inconsistencies between the notebook and the `haldensity.censoring` module have been resolved
- The notebook now correctly uses the current API signatures and result dictionary keys
- No breaking changes were made to the existing API; only documentation and notebook corrections
- The EMIPCWEstimator now properly reports convergence information, which is useful for diagnostics and tuning

