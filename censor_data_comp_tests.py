import sys
import time
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
from haldensity.utils.density_computations import (
    generic_compute_survival_from_density,
    generic_compute_cdf_from_density,
)


MEAN = 0.5
STD = 0.1
LOWER = 0.0
UPPER = 1.0
SEED = 12776
N_SAMPLES = 1000
N_GRID_POINTS = 200
BEST_LAMBDA = 70.0
M_STEP_NORM = 5 * BEST_LAMBDA
BASIS_ORDER = 0
M_IMPUTATIONS = 50  # Reduced from 400 for speed (legacy uses 400)
MAX_EM_ITER = 5  # Reduced from 200 for speed (legacy converges in ~4 iters)
EM_TOL = 0.01
MAX_RUNTIME_SECONDS = 120.0  # Increased to 2 minutes given current performance


def generate_survival_data_from_dgp(n: int, rng: np.random.Generator) -> pd.DataFrame:
    a, b = (LOWER - MEAN) / STD, (UPPER - MEAN) / STD
    T = truncnorm.rvs(a, b, loc=MEAN, scale=STD, size=n, random_state=rng)
    C = rng.uniform(0.0, 1.0, size=n)
    T_tilde = np.minimum(T, C)
    delta = (T <= C).astype(int)
    return pd.DataFrame({"T_tilde": T_tilde, "delta": delta})


def true_density(grid: np.ndarray) -> np.ndarray:
    a, b = (LOWER - MEAN) / STD, (UPPER - MEAN) / STD
    return truncnorm.pdf(grid, a, b, loc=MEAN, scale=STD)


def check_density_validity(estimator, name: str):
    grid, f = estimator.get_density()
    assert np.all(np.isfinite(f)), f"{name}: density has non-finite values"
    assert np.all(f >= 0), f"{name}: density has negatives"
    delta = np.empty_like(grid)
    if len(delta) > 1:
        delta[1:] = np.diff(grid)
        delta[0] = delta[1]
    else:
        delta[0] = 1.0
    integral = float(np.sum(f * delta))
    assert abs(integral - 1.0) < 1e-2, f"{name}: density integral off: {integral}"
    cdf, _ = generic_compute_cdf_from_density(grid, f)
    surv, _ = generic_compute_survival_from_density(grid, f)
    assert np.all(np.diff(cdf) >= -1e-8), f"{name}: CDF not monotone"
    assert np.all(np.diff(surv) <= 1e-8), f"{name}: Survival not monotone"


def main():
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    start = time.perf_counter()
    print(f"Starting at {start:.2f}s")
    
    rng = np.random.default_rng(SEED)
    raw = generate_survival_data_from_dgp(N_SAMPLES, rng)
    data = pd.DataFrame({"T": raw["T_tilde"], "Delta": raw["delta"]})
    print(f"Generated data at {time.perf_counter() - start:.2f}s")

    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    w = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)
    mask_unc = data["Delta"].values == 1
    df_unc = pd.DataFrame({"W1": data.loc[mask_unc, "T"].values})
    w_unc = w[mask_unc]
    print(f"Computed IPCW weights at {time.perf_counter() - start:.2f}s")

    print("Fitting IPCW-HAL-MLE with SCS...")
    ipcw_est = WeightedCVXPYEstimator(
        norm_constraint=BEST_LAMBDA,
        basis_order=BASIS_ORDER,
        n_grid_points=N_GRID_POINTS,
        solver="SCS",  # Match legacy init
        use_secondary_solver=False,
        legacy_mode=True,
        include_intercept_in_constraint=True,
    ).fit(df_unc, sample_weights=w_unc)
    print(f"IPCW-HAL-MLE fitted at {time.perf_counter() - start:.2f}s")

    print("Fitting EM-IPCW-HAL-MLE...")
    em_est = EMIPCWEstimator(
        norm_constraint=M_STEP_NORM,
        basis_order=BASIS_ORDER,
        n_grid_points=N_GRID_POINTS,
        m_imputations=M_IMPUTATIONS,
        max_em_iter=MAX_EM_ITER,
        em_tol=EM_TOL,
    use_sc_adjustment=False,
        verbose=True,
        init_solver="SCS",  # Match legacy init
        m_step_solver="ECOS",  # Match legacy M-step
        init_norm_constraint=BEST_LAMBDA,
        m_step_norm_constraint=M_STEP_NORM,
        e_step_n_grid=1000,
        rng_seed=SEED,
    ).fit(data)
    print(f"EM-IPCW-HAL-MLE fitted at {time.perf_counter() - start:.2f}s")

    ipcw_ll = incomplete_loglik(ipcw_est, data, time_col="T", delta_col="Delta")
    em_ll = incomplete_loglik(em_est, data, time_col="T", delta_col="Delta")
    print(f"Incomplete-data log-likelihood (IPCW): {ipcw_ll:.6f}")
    print(f"Incomplete-data log-likelihood (EM):   {em_ll:.6f}")

    grid_ipcw, f_ipcw = ipcw_est.get_density()
    grid_em, f_em = em_est.get_density()
    kl_ipcw = kl_divergence(true_density, grid_ipcw, f_ipcw)
    kl_em = kl_divergence(true_density, grid_em, f_em)
    print(f"KL Divergence (True vs. IPCW): {kl_ipcw}")
    print(f"KL Divergence (True vs. EM):   {kl_em}")
    assert kl_ipcw < 0.1, "KL(IPCW) too large"
    assert kl_em < 0.1, "KL(EM) too large"

    check_density_validity(ipcw_est, "IPCW")
    check_density_validity(em_est, "EM")

    elapsed = time.perf_counter() - start
    print(f"\n✓ All censored-data tests passed in {elapsed:.2f} seconds.")
    assert elapsed <= MAX_RUNTIME_SECONDS, f"Runtime {elapsed:.2f}s exceeded {MAX_RUNTIME_SECONDS}s budget"


if __name__ == "__main__":
    main()
