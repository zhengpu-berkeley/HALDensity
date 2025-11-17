#!/usr/bin/env python
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, "src")

from haldensity.censoring import KaplanMeier, compute_ipcw_weights, WeightedCVXPYEstimator

print("Generating data...")
rng = np.random.default_rng(12776)
T = truncnorm.rvs(-5, 5, loc=0.5, scale=0.1, size=1000, random_state=rng)
C = rng.uniform(0, 1, 1000)
T_tilde = np.minimum(T, C)
delta = (T <= C).astype(int)
data = pd.DataFrame({"T": T_tilde, "Delta": delta})
print(f"Generated {len(data)} samples, {delta.sum()} uncensored")

print("\nFitting KM...")
km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
print("✓ KM fitted")

print("\nComputing IPCW weights...")
w = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)
print(f"✓ Computed weights, mean={w.mean():.3f}")

print("\nPreparing uncensored data...")
mask_unc = data["Delta"].values == 1
df_unc = pd.DataFrame({"W1": data.loc[mask_unc, "T"].values})
w_unc = w[mask_unc]
print(f"✓ Prepared {len(df_unc)} uncensored points")

print("\nFitting IPCW-HAL with ECOS...")
start = time.perf_counter()
ipcw_est = WeightedCVXPYEstimator(
    norm_constraint=70.0,
    basis_order=0,
    n_grid_points=200,
    solver="ECOS",
    use_secondary_solver=False,
    legacy_mode=True,
    include_intercept_in_constraint=True,
).fit(df_unc, sample_weights=w_unc)
elapsed = time.perf_counter() - start
print(f"✓ IPCW fitted in {elapsed:.2f}s, theta shape: {ipcw_est.theta_hat.shape}")

print("\n✓✓✓ All tests passed!")

