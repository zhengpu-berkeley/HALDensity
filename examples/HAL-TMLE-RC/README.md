# HAL-TMLE Right-Censoring Examples

Start with [hal-tmle-rc-target-learning.ipynb](/Users/houyilong/GitHub/HALDensity/examples/HAL-TMLE-RC/hal-tmle-rc-target-learning.ipynb). It is the new unified demo notebook for the right-censored target-learning workflow.

What this notebook does:

- switches between `Survival`, `RMST`, `DensitySquare`, and `Entropy` with one `TARGET_TYPE` setting
- includes a batch section that demonstrates all four targets on the same simulated data set
- includes `KM` and integrated-`KM` comparators for `Survival` and `RMST`
- uses `CV-IPCW-L1MLE` as the initial estimator
- simulates censoring from `Unif[0, 1.2]`
- uses the updated automatic targeting rule that checks the initial score before taking any TMLE step
- uses the updated targeting `Gbar` floor of order `1 / (sqrt(n) log n)`
- ends with both a single-target MC summary and an all-four-target MC summary for bias, variance, MSE, and leave-one-out oracle coverage

Supporting code lives in [rc_target_learning_demo_utils.py](/Users/houyilong/GitHub/HALDensity/examples/HAL-TMLE-RC/rc_target_learning_demo_utils.py).

The older notebooks in this folder are still here for reference, but they are now best treated as legacy working notes rather than the main entry point.
