# Univariate Fixed-Sobol Normalizer Demo Notebook

## Summary

- Create one notebook at `examples/example_univariate_sobol_normalizer.ipynb`.
- Keep this phase notebook-only: no `src/` edits, no estimator API changes, no cross-validation work.
- Use the package’s `TruncatedNormal()` generator on `[0,1]`, `basis_order=1`, and `norm_constraint=65`.
- Compare the current package `CVXPYEstimator` against a notebook-local CVXPY fit that differs only in the normalizer approximation.
- Treat this as an estimator-validity experiment, not a speed experiment.

## Core Design

- Fix one training sample and one validation sample for the whole notebook.
- Use `n_train = 500`, `n_valid = 5000`, and a fixed data seed `0`.
- Fit the baseline with `CVXPYEstimator(norm_constraint=65, basis_order=1, n_grid_points=400, solver="ECOS", use_secondary_solver=True)`.
- In the notebook, build a helper `fit_sobol_cvx(train_df, basis_order, norm_constraint, n_bank, scramble_seed)` that reuses the exact same knot set and basis construction as the baseline:
  - `grid_points_hal = np.unique(train_df["W1"].dropna())`
  - `create_basis_functions(..., order=1, include_intercept=True)`
  - the same `L1` constraint on `theta[1:]`
- Use a fixed scrambled Sobol bank on `[0,1]` for the normalizer approximation.
- Use bank sizes `S in {256, 1024, 4096}`.
- Use scramble seeds `{7, 17, 29, 41, 53}` for the robustness block.
- For the main visual comparison, use seed `7` as the representative run for each bank size.
- Define the Sobol normalizer in CVXPY as
  \[
  \log \hat Z*{\text{sobol}}(\theta)=\operatorname{logsumexp}(\Phi*{\text{bank}}\theta)-\log S,
  \]
  which corresponds to the uniform empirical approximation
  \[
  \hat Z*{\text{sobol}}(\theta)=\frac1S\sum*{s=1}^S \exp\{ \phi(z_s)^\top \theta \}.
  \]
- Use the same solver policy as the package baseline: `ECOS` first, with the same fallback sequence if needed.

## Notebook Flow

- **Setup**
  - Import `numpy`, `pandas`, `matplotlib`, `cvxpy`, `scipy.stats.qmc.Sobol`, `CVXPYEstimator`, `TruncatedNormal`, and `create_basis_functions`.
  - Add a short markdown statement that the Sobol path changes only the normalizer approximation; knots, basis, constraint, and solver setup are otherwise matched to the baseline.
- **Data generation**
  - Draw training and validation samples from `TruncatedNormal()`.
  - Build a fine reference grid of size `10000` on `(0.001, 0.999)` for truth plots and deterministic reference integration.
  - Compute the true truncated-normal density on that grid.
- **Baseline fit**
  - Fit the package `CVXPYEstimator`.
  - Save `theta_hat`, `grid_points_hal_selected`, validation log-likelihood, and the native estimated density on the fine grid.
- **Notebook-local Sobol fit**
  - For each representative bank size with seed `7`, solve the notebook-local CVXPY problem.
  - Save `theta_hat`, bank points, native Sobol-normalized density on the fine grid, and validation log-likelihood.
- **Reference normalizer diagnostics**
  - For every fitted `theta`, compute a high-resolution deterministic reference
    \[
    \log Z*{\text{ref}}(\theta)=\log \int_0^1 \exp\{f*\theta(x)\}\,dx
    \]
    using the common fine grid and stable log-sum-exp style integration.
  - For every fitted `theta`, also compute its native normalizer:
    - midpoint-native `logZ` for the baseline using the package estimator’s own normalization grid,
    - bank-native `logZ` for Sobol using the same fixed Sobol bank used in the fit.
  - Report `logZ_native`, `logZ_ref`, and `|logZ_native - logZ_ref|` in a summary table.
- **Density comparisons**
  - For every fitted `theta`, compute two densities on the fine grid:
    - the native estimator density using that method’s own normalizer,
    - a reference-renormalized density using `logZ_ref(theta)` only for diagnostic shape comparison.
  - Label the second curve explicitly as `reference-renormalized diagnostic`, not as the estimator output.
- **Robustness block**
  - For each `S in {256, 1024, 4096}` and each scramble seed in `{7, 17, 29, 41, 53}`, refit the Sobol approximation.
  - Report a compact table with mean, min, and max over seeds for the main metrics.
  - Keep plotting to the representative seed `7` to avoid clutter.
- **Concavity check**
  - Include a markdown derivation showing that with a fixed bank the Sobol log-normalizer is `logsumexp` of affine functions of `theta`, so the approximate negative log-likelihood remains convex.
  - Add one numerical line-slice sanity plot of the Sobol objective along a random direction through a fitted solution as an illustration only.

## Metrics and Acceptance Criteria

- Use these primary metrics:
  - `native normalization residual`
  - `fine-grid mass`
  - `held-out average log-likelihood`
  - `integrated absolute error to truth`
  - `integrated absolute difference to the midpoint baseline`
  - `|logZ_native - logZ_ref|`
- Define `native normalization residual` carefully:
  - baseline: `abs(sum_j p_hat(mid_j) * delta_j - 1)`
  - Sobol: `abs(mean_s p_hat(z_s) - 1)`
  - describe this as a method-native normalization check, not as exact Lebesgue-mass preservation
- Treat these as hard validity checks:
  - all native densities are nonnegative on the fine grid
  - all native normalization residuals are near machine precision
  - all fine-grid masses lie in `[0.95, 1.05]`
  - all held-out log-likelihoods are finite
- Treat these as decision metrics for whether the prototype is promising:
  - `S=1024` and `S=4096` should stay close to the midpoint baseline in held-out log-likelihood and density shape
  - `|logZ_native - logZ_ref|` should shrink materially as `S` increases
  - seed-to-seed variation should be modest relative to the midpoint-vs-Sobol gap

## Outputs and Conclusion

- Include one plot of truth vs native baseline vs native Sobol densities.
- Include one plot of truth vs reference-renormalized baseline vs reference-renormalized Sobol densities.
- Include one table for representative-seed runs with all primary metrics.
- Include one robustness table over scramble seeds for each bank size.
- End with a short markdown conclusion that answers:
  - whether fixed-bank Sobol produces a valid native estimator in this univariate case,
  - whether the approximation stays close enough to midpoint normalization to justify a package-level phase 2,
  - and which bank size looks like the smallest acceptable univariate demo setting.

## Assumptions

- The notebook remains univariate and uses only `basis_order=1` and `norm_constraint=65`.
- The baseline package estimator is treated as the reference implementation.
- The Sobol bank is fixed within each fit and never resampled during optimization.
- No claims about runtime advantage or multivariate scalability will be made from this notebook alone.
