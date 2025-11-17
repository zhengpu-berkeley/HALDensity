#!/usr/bin/env python
"""Complete test for censored data estimation matching legacy notebook."""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, "src")

from haldensity.censoring import (
    KaplanMeier,
    compute_ipcw_weights,
    WeightedCVXPYEstimator,
    EMIPCWEstimator,
    kl_divergence,
)
from haldensity.censoring.metrics import incomplete_loglik
from haldensity.utils.density_computations import generic_compute_survival_from_density, generic_compute_cdf_from_density

warnings.filterwarnings('ignore', category=UserWarning)

# DGP parameters
MEAN, STD, LOWER, UPPER = 0.5, 0.1, 0.0, 1.0
SEED, N_SAMPLES = 12776, 1000
BEST_LAMBDA, M_STEP_NORM = 70.0, 350.0
M_IMPUTATIONS, MAX_EM_ITER, EM_TOL = 50, 200, 0.01

start = time.perf_counter()
print("=" * 60)
print("CENSORED DATA COMPREHENSIVE TEST")
print("=" * 60)

# Generate data
rng = np.random.default_rng(SEED)
a, b = (LOWER - MEAN) / STD, (UPPER - MEAN) / STD
T = truncnorm.rvs(a, b, loc=MEAN, scale=STD, size=N_SAMPLES, random_state=rng)
C = rng.uniform(LOWER, UPPER, N_SAMPLES)
T_tilde, delta = np.minimum(T, C), (T <= C).astype(int)
data = pd.DataFrame({"T": T_tilde, "Delta": delta})
print(f"\n✓ Generated {N_SAMPLES} samples ({delta.sum()} uncensored) at {time.perf_counter()-start:.2f}s")

# IPCW
km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
w = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)
df_unc = pd.DataFrame({"W1": data.loc[delta == 1, "T"].values})
w_unc = w[delta == 1]

print(f"\nFitting IPCW-HAL-MLE...")
ipcw_est = WeightedCVXPYEstimator(
    norm_constraint=BEST_LAMBDA, basis_order=0, n_grid_points=200,
    solver="SCS", use_secondary_solver=False, legacy_mode=True,
    include_intercept_in_constraint=True,
).fit(df_unc, sample_weights=w_unc)
print(f"✓ IPCW fitted at {time.perf_counter()-start:.2f}s")

# EM
print(f"\nFitting EM-IPCW-HAL-MLE (m={M_IMPUTATIONS})...")
em_est = EMIPCWEstimator(
    norm_constraint=M_STEP_NORM, basis_order=0, n_grid_points=200,
    m_imputations=M_IMPUTATIONS, max_em_iter=10,
    em_tol=EM_TOL,
    use_sc_adjustment=False, verbose=True,
    init_solver="SCS", m_step_solver="ECOS",
    init_norm_constraint=BEST_LAMBDA, m_step_norm_constraint=M_STEP_NORM,
    e_step_n_grid=1000, rng_seed=SEED,
).fit(data)
print(f"✓ EM fitted at {time.perf_counter()-start:.2f}s")

# Metrics
ipcw_ll = incomplete_loglik(ipcw_est, data, time_col="T", delta_col="Delta")
em_ll = incomplete_loglik(em_est, data, time_col="T", delta_col="Delta")
print(f"\nIncomplete-data log-likelihood (IPCW): {ipcw_ll:.6f}")
print(f"Incomplete-data log-likelihood (EM):   {em_ll:.6f}")

true_pdf = lambda x: truncnorm.pdf(x, a, b, loc=MEAN, scale=STD)
grid_ipcw, f_ipcw = ipcw_est.get_density()
grid_em, f_em = em_est.get_density()
kl_ipcw = kl_divergence(true_pdf, grid_ipcw, f_ipcw)
kl_em = kl_divergence(true_pdf, grid_em, f_em)
print(f"KL Divergence (True vs. IPCW): {kl_ipcw:.6f}")
print(f"KL Divergence (True vs. EM):   {kl_em:.6f}")

assert kl_ipcw < 0.1, f"KL(IPCW)={kl_ipcw:.4f} too large"
assert kl_em < 0.1, f"KL(EM)={kl_em:.4f} too large"

elapsed = time.perf_counter() - start
print(f"\n{'='*60}")
print(f"✓✓✓ ALL TESTS PASSED in {elapsed:.2f}s")
print(f"{'='*60}")

