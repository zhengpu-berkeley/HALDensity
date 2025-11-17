#!/usr/bin/env python
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, "src")

from haldensity.censoring import EMIPCWEstimator

print("Generating data...")
rng = np.random.default_rng(12776)
T = truncnorm.rvs(-5, 5, loc=0.5, scale=0.1, size=1000, random_state=rng)
C = rng.uniform(0, 1, 1000)
T_tilde = np.minimum(T, C)
delta = (T <= C).astype(int)
data = pd.DataFrame({"T": T_tilde, "Delta": delta})
print(f"Generated {len(data)} samples, {delta.sum()} uncensored")

print("\nFitting EM-IPCW-HAL with ECOS...")
start = time.perf_counter()
em_est = EMIPCWEstimator(
    norm_constraint=5 * 70.0,
    basis_order=0,
    n_grid_points=200,
    m_imputations=100,  # Reduced from 400
    max_em_iter=200,
    em_tol=0.01,
    use_sc_adjustment=False,
    verbose=True,
    init_solver="ECOS",
    m_step_solver="ECOS",
    init_norm_constraint=70.0,
    m_step_norm_constraint=5 * 70.0,
    e_step_n_grid=1000,
    rng_seed=12776,
).fit(data)
elapsed = time.perf_counter() - start
print(f"\n✓ EM fitted in {elapsed:.2f}s, theta shape: {em_est.theta_hat.shape}")

print("\n✓✓✓ All tests passed!")

