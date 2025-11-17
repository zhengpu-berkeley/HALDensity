import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, "src")

from haldensity.censoring import CensoredOptunaHyperparameterTuner
from haldensity.censoring.metrics import (
    incomplete_loglik,
    mi_complete_loglik,
    kl_divergence,
)


@dataclass
class Config:
    seed: int = 2025
    n_samples: int = 400
    n_trials: int = 8
    cv_folds: int = 3
    n_grid_points: int = 200
    max_runtime_seconds: float = 240.0
    true_mean: float = 0.5
    true_std: float = 0.12
    lower: float = 0.0
    upper: float = 1.0
    orders: tuple[int, ...] = (0, 1, 2)
    kl_tol: float = 0.1


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Optuna-based censoring tests for HALDensity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--n-samples", type=int, default=Config.n_samples)
    parser.add_argument("--n-trials", type=int, default=Config.n_trials)
    parser.add_argument("--cv-folds", type=int, default=Config.cv_folds)
    parser.add_argument("--n-grid-points", type=int, default=Config.n_grid_points)
    parser.add_argument("--max-runtime", type=float, default=Config.max_runtime_seconds)
    parser.add_argument("--orders", type=str, default="0,1,2", help="Comma-separated basis orders to test")
    parser.add_argument("--kl-tol", type=float, default=Config.kl_tol)
    args = parser.parse_args()
    if args.orders.strip():
        orders = tuple(sorted({int(o) for o in args.orders.split(",") if o.strip()}))
    else:
        orders = Config.orders
    return Config(
        seed=args.seed,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        n_grid_points=args.n_grid_points,
        max_runtime_seconds=args.max_runtime,
        orders=orders,
        kl_tol=args.kl_tol,
    )


def _simulate_data(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    mean, std = cfg.true_mean, cfg.true_std
    a, b = (cfg.lower - mean) / std, (cfg.upper - mean) / std
    T = truncnorm.rvs(a, b, loc=mean, scale=std, size=cfg.n_samples, random_state=rng)
    C = rng.uniform(cfg.lower, cfg.upper, size=cfg.n_samples)
    T_tilde = np.minimum(T, C)
    delta = (T <= C).astype(int)
    return pd.DataFrame({"T": T_tilde, "Delta": delta})


def _run_tuner(
    estimator_name: str,
    basis_order: int,
    metric: str,
    cfg: Config,
    data: pd.DataFrame,
    n_trials: int,
) -> tuple[object, float]:
    overrides: dict[str, object] = {
        "basis_order": [basis_order],
        "norm_constraint": {"low": 10.0, "high": 400.0, "log": True},
    }
    if estimator_name == "EMIPCWEstimator":
        overrides.update(
            {
                "m_imputations": {"low": 8, "high": 20},
                "max_em_iter": {"low": 3, "high": 6},
                "use_sc_adjustment": {"choices": [False, True]},
            }
        )
    tuner = CensoredOptunaHyperparameterTuner(
        estimator_name=estimator_name,
        data=data,
        cv_folds=cfg.cv_folds,
        metric=metric,
        random_state=cfg.seed + basis_order,
        n_grid_points=cfg.n_grid_points,
        param_overrides=overrides,
        silent=True,
    )
    tuner.optimize(n_trials=n_trials)
    est = tuner.fit_best_model()
    if metric == "incomplete":
        score = incomplete_loglik(est, data, time_col="T", delta_col="Delta")
    else:
        augmented = getattr(est, "uncensored_augmented_", None)
        score = mi_complete_loglik(est, augmented)
    if not np.isfinite(score):
        raise RuntimeError(f"Non-finite {metric} score for {estimator_name} (order={basis_order})")
    return est, score


def _true_density(cfg: Config, points: np.ndarray) -> np.ndarray:
    a, b = (cfg.lower - cfg.true_mean) / cfg.true_std, (cfg.upper - cfg.true_mean) / cfg.true_std
    return truncnorm.pdf(points, a, b, loc=cfg.true_mean, scale=cfg.true_std)


def _check_kl(estimator: object, cfg: Config, label: str) -> None:
    grid, density = estimator.get_density()
    kl = kl_divergence(lambda pts: _true_density(cfg, pts), grid, density)
    print(f"{label} KL(true || {label}) = {kl:.4f}")
    if kl > cfg.kl_tol or not np.isfinite(kl):
        raise RuntimeError(f"{label} KL {kl:.4f} exceeds tolerance {cfg.kl_tol}")


def main(cfg: Config | None = None) -> None:
    cfg = cfg or parse_args()
    start = time.perf_counter()
    print(f"Starting Optuna tuner tests (seed={cfg.seed}, trials={cfg.n_trials})")
    data = _simulate_data(cfg)

    weighted_orders = cfg.orders if cfg.orders else (0,)
    for order in weighted_orders:
        est, weighted_score = _run_tuner(
            "WeightedCVXPYEstimator",
            basis_order=order,
            metric="incomplete",
            cfg=cfg,
            data=data,
            n_trials=max(2, cfg.n_trials // 2) if order > 0 else cfg.n_trials,
        )
        print(
            f"WeightedCVXPYEstimator (order={order}) incomplete loglik: {weighted_score:.4f}"
        )
        _check_kl(est, cfg, f"WeightedCVXPYEstimator(order={order})")

    for basis_order in cfg.orders:
        est, inc_score = _run_tuner(
            "EMIPCWEstimator",
            basis_order=basis_order,
            metric="incomplete",
            cfg=cfg,
            data=data,
            n_trials=cfg.n_trials,
        )
        print(
            f"EMIPCWEstimator (order={basis_order}) incomplete loglik: {inc_score:.4f}"
        )
        _check_kl(est, cfg, f"EMIPCWEstimator(order={basis_order})")
        if basis_order == 0:
            _, mi_score = _run_tuner(
                "EMIPCWEstimator",
                basis_order=basis_order,
                metric="mi_complete",
                cfg=cfg,
                data=data,
                n_trials=max(2, cfg.n_trials // 2),
            )
            print(f"EMIPCWEstimator (order=0) MI-complete loglik: {mi_score:.4f}")

    elapsed = time.perf_counter() - start
    print(f"\n✓ Optuna tuner tests finished in {elapsed:.2f} seconds")
    if elapsed > cfg.max_runtime_seconds:
        raise RuntimeError(
            f"Optuna tuner tests exceeded runtime budget {cfg.max_runtime_seconds}s (took {elapsed:.2f}s)"
        )


if __name__ == "__main__":
    main(parse_args())
