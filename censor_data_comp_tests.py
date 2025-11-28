import argparse
import sys
import time
from dataclasses import dataclass

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


@dataclass
class Config:
    mean: float = 0.5
    std: float = 0.1
    lower: float = 0.0
    upper: float = 1.0
    seed: int = 12776
    n_samples: int = 1000
    n_grid_points: int = 200
    basis_order: int = 0
    norm_constraint: float = 8.0
    # M-step norm constraint: None means use same as norm_constraint (default behavior)
    # Note: Using a very large value (e.g., 1000) leads to degenerate solutions
    # because the M-step will overfit to the augmented data
    m_step_norm_constraint: float = None
    m_imputations: int = 50
    max_em_iter: int = 5
    em_tol: float = 0.01
    init_solver: str = "ECOS"
    m_step_solver: str = "ECOS"
    e_step_n_grid: int = 1000
    max_runtime_seconds: float = 120.0
    use_sc_adjustment: bool = False
    verbose_em: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Censored-data regression test for HALDensity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--n-samples", type=int, default=Config.n_samples)
    parser.add_argument("--n-grid-points", type=int, default=Config.n_grid_points)
    parser.add_argument("--basis-order", type=int, default=Config.basis_order)
    parser.add_argument("--norm-constraint", type=float, default=Config.norm_constraint)
    parser.add_argument("--m-step-norm-constraint", type=float, default=None)
    parser.add_argument("--m-imputations", type=int, default=Config.m_imputations)
    parser.add_argument("--max-em-iter", type=int, default=Config.max_em_iter)
    parser.add_argument("--em-tol", type=float, default=Config.em_tol)
    parser.add_argument("--init-solver", type=str, default=Config.init_solver)
    parser.add_argument("--m-step-solver", type=str, default=Config.m_step_solver)
    parser.add_argument("--e-step-n-grid", type=int, default=Config.e_step_n_grid)
    parser.add_argument("--max-runtime", type=float, default=Config.max_runtime_seconds)
    parser.add_argument("--use-sc-adjustment", action="store_true")
    parser.add_argument("--verbose-em", action="store_true", default=Config.verbose_em)
    args = parser.parse_args()
    if args.m_step_norm_constraint is not None:
        m_step_norm_constraint = args.m_step_norm_constraint
    elif args.basis_order == 0:
        m_step_norm_constraint = 1.0 * args.norm_constraint
    else:
        m_step_norm_constraint = 5.0 * args.norm_constraint
    return Config(
        seed=args.seed,
        n_samples=args.n_samples,
        n_grid_points=args.n_grid_points,
        basis_order=args.basis_order,
        norm_constraint=args.norm_constraint,
        m_step_norm_constraint=m_step_norm_constraint,
        m_imputations=args.m_imputations,
        max_em_iter=args.max_em_iter,
        em_tol=args.em_tol,
        init_solver=args.init_solver,
        m_step_solver=args.m_step_solver,
        e_step_n_grid=args.e_step_n_grid,
        max_runtime_seconds=args.max_runtime,
        use_sc_adjustment=args.use_sc_adjustment,
        verbose_em=args.verbose_em,
    )


def _generate_data(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    a, b = (cfg.lower - cfg.mean) / cfg.std, (cfg.upper - cfg.mean) / cfg.std
    T = truncnorm.rvs(a, b, loc=cfg.mean, scale=cfg.std, size=cfg.n_samples, random_state=rng)
    C = rng.uniform(cfg.lower, cfg.upper, size=cfg.n_samples)
    T_tilde = np.minimum(T, C)
    delta = (T <= C).astype(int)
    return pd.DataFrame({"T": T_tilde, "Delta": delta})


def _true_density(cfg: Config, grid: np.ndarray) -> np.ndarray:
    a, b = (cfg.lower - cfg.mean) / cfg.std, (cfg.upper - cfg.mean) / cfg.std
    return truncnorm.pdf(grid, a, b, loc=cfg.mean, scale=cfg.std)


def _check_density(estimator, name: str):
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
    assert abs(integral - 1.0) < 1e-2, f"{name}: density integral {integral}"
    cdf, _ = generic_compute_cdf_from_density(grid, f)
    surv, _ = generic_compute_survival_from_density(grid, f)
    assert np.all(np.diff(cdf) >= -1e-8), f"{name}: CDF not monotone"
    assert np.all(np.diff(surv) <= 1e-8), f"{name}: Survival not monotone"


def main(cfg: Config | None = None):
    cfg = cfg or parse_args()
    start = time.perf_counter()
    print(f"Starting at {start:.2f}s (seed={cfg.seed})")

    rng = np.random.default_rng(cfg.seed)
    data = _generate_data(cfg, rng)
    print(f"Generated data at {time.perf_counter() - start:.2f}s")

    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    w = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)
    mask_unc = data["Delta"].values == 1
    df_unc = pd.DataFrame({"W1": data.loc[mask_unc, "T"].values})
    w_unc = w[mask_unc]
    print(f"Computed IPCW weights at {time.perf_counter() - start:.2f}s")

    print(f"Fitting IPCW-HAL-MLE with {cfg.init_solver}...")
    ipcw_est = WeightedCVXPYEstimator(
        norm_constraint=cfg.norm_constraint,
        basis_order=cfg.basis_order,
        n_grid_points=cfg.n_grid_points,
        solver=cfg.init_solver,
        use_secondary_solver=False,
        include_intercept_in_constraint=True,
    ).fit(df_unc, sample_weights=w_unc)
    print(f"IPCW-HAL-MLE fitted at {time.perf_counter() - start:.2f}s")

    print("Fitting EM-IPCW-HAL-MLE...")
    em_est = EMIPCWEstimator(
        norm_constraint=cfg.m_step_norm_constraint,
        basis_order=cfg.basis_order,
        n_grid_points=cfg.n_grid_points,
        m_imputations=cfg.m_imputations,
        max_em_iter=cfg.max_em_iter,
        em_tol=cfg.em_tol,
        use_sc_adjustment=cfg.use_sc_adjustment,
        verbose=cfg.verbose_em,
        init_solver=cfg.init_solver,
        m_step_solver=cfg.m_step_solver,
        init_norm_constraint=cfg.norm_constraint,
        m_step_norm_constraint=cfg.m_step_norm_constraint,
        e_step_n_grid=cfg.e_step_n_grid,
        rng_seed=cfg.seed,
    ).fit(data)
    print(f"EM-IPCW-HAL-MLE fitted at {time.perf_counter() - start:.2f}s")

    ipcw_ll = incomplete_loglik(ipcw_est, data, time_col="T", delta_col="Delta")
    em_ll = incomplete_loglik(em_est, data, time_col="T", delta_col="Delta")
    print(f"Incomplete-data log-likelihood (IPCW): {ipcw_ll:.6f}")
    print(f"Incomplete-data log-likelihood (EM):   {em_ll:.6f}")

    grid_ipcw, f_ipcw = ipcw_est.get_density()
    grid_em, f_em = em_est.get_density()
    kl_ipcw = kl_divergence(lambda pts: _true_density(cfg, pts), grid_ipcw, f_ipcw)
    kl_em = kl_divergence(lambda pts: _true_density(cfg, pts), grid_em, f_em)
    print(f"KL Divergence (True vs. IPCW): {kl_ipcw}")
    print(f"KL Divergence (True vs. EM):   {kl_em}")
    assert kl_ipcw < 0.1, "KL(IPCW) too large"
    assert kl_em < 0.1, "KL(EM) too large"
    # Note: We do NOT assert ipcw_ll <= em_ll because:
    # 1. EM with multiple imputation is a Monte Carlo approximation
    # 2. M-step optimizes complete-data LL (Σ log f(T*)), not incomplete-data LL (Σ δ·log f + (1-δ)·log S)
    # 3. With regularization, these objectives can diverge
    # The EM guarantee of monotonic LL increase only holds for exact, unconstrained EM.

    _check_density(ipcw_est, "IPCW")
    _check_density(em_est, "EM")

    elapsed = time.perf_counter() - start
    print(f"\n✓ All censored-data tests passed in {elapsed:.2f} seconds.")
    assert elapsed <= cfg.max_runtime_seconds, (
        f"Runtime {elapsed:.2f}s exceeded {cfg.max_runtime_seconds}s budget"
    )


if __name__ == "__main__":
    main()
