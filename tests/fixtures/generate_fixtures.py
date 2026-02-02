"""Generate test fixtures for regression testing.

This script generates synthetic data and runs estimators to capture expected results
that will be used to validate the refactor preserves behavior.

Run with: uv run python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

# Estimators
from haldensity.censoring.right.estimators import RightCensoredInitEstimator, RightCensoredEMEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.metrics import incomplete_loglik, mi_complete_loglik
from haldensity.censoring.interval.estimators import IntervalCensoredInitEstimator, IntervalCensoredEMEstimator
from haldensity.censoring.interval.metrics import incomplete_loglik_interval
from haldensity.censoring.utils.common_metrics import kl_divergence

FIXTURES_DIR = Path(__file__).parent
SEED = 42
N_SAMPLES = 200

# Estimator parameters for reproducibility
RC_INIT_PARAMS = {
    "norm_constraint": 10.0,
    "n_grid_points": 100,
    "basis_order": 0,
    "solver": "ECOS",
    "use_secondary_solver": True,
}

RC_EM_PARAMS = {
    "norm_constraint": 10.0,
    "n_grid_points": 100,
    "basis_order": 0,
    "m_imputations": 20,
    "max_em_iter": 10,
    "em_tol": 1e-4,
    "rng_seed": SEED,
}

IC_INIT_PARAMS = {
    "norm_constraint": 10.0,
    "n_grid_points": 100,
    "basis_order": 0,
    "solver": "ECOS",
    "use_secondary_solver": True,
}

IC_EM_PARAMS = {
    "norm_constraint": 10.0,
    "n_grid_points": 100,
    "basis_order": 0,
    "m_imputations": 20,
    "max_em_iter": 10,
    "em_tol": 1e-4,
    "rng_seed": SEED,
}


def generate_right_censored_data(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic right-censored data.
    
    True event times ~ Beta(2, 5) scaled to [0, 1]
    Censoring times ~ Uniform(0, 1)
    """
    rng = np.random.default_rng(seed)
    
    # True event times from Beta(2, 5) - skewed distribution
    t_event = rng.beta(2, 5, size=n)
    
    # Censoring times from Uniform
    t_cens = rng.uniform(0.0, 1.0, size=n)
    
    # Observed time and event indicator
    t_obs = np.minimum(t_event, t_cens)
    delta = (t_event <= t_cens).astype(int)
    
    # Ensure mix of censored/uncensored
    if delta.sum() == 0 or delta.sum() == n:
        delta[:n // 10] = 1
        delta[-n // 10:] = 0
    
    return pd.DataFrame({
        "T": t_obs.astype(float),
        "Delta": delta.astype(int),
    })


def generate_interval_censored_data(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic interval-censored data.
    
    True event times ~ Beta(2, 5) scaled to [0, 1]
    Interval width ~ Uniform(0.05, 0.2)
    """
    rng = np.random.default_rng(seed)
    
    # True event times
    t_event = rng.beta(2, 5, size=n)
    
    # Random interval widths
    widths = rng.uniform(0.05, 0.2, size=n)
    
    # Left endpoint: shift left by random fraction of width
    left_shift = rng.uniform(0, 1, size=n) * widths
    L = np.clip(t_event - left_shift, 0.0, 1.0)
    R = np.clip(L + widths, 0.0, 1.0)
    
    # Ensure L < R
    R = np.maximum(R, L + 0.01)
    R = np.minimum(R, 1.0)
    
    return pd.DataFrame({
        "L": L.astype(float),
        "R": R.astype(float),
    })


def fit_rc_init_estimator(data: pd.DataFrame) -> tuple[RightCensoredInitEstimator, dict]:
    """Fit RC Init (IPCW) estimator and return results."""
    # Compute IPCW weights
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    T_vals = np.asarray(data["T"].values, dtype=float)
    Delta_vals = np.asarray(data["Delta"].values, dtype=int)
    weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
    
    # Fit on uncensored observations
    unc_mask = Delta_vals == 1
    df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
    w_unc = weights[unc_mask]
    
    est = RightCensoredInitEstimator(**RC_INIT_PARAMS)
    est.fit(df_unc, sample_weights=w_unc)
    
    return est, est.get_results()


def fit_rc_em_estimator(data: pd.DataFrame) -> tuple[RightCensoredEMEstimator, dict]:
    """Fit RC EM estimator and return results."""
    est = RightCensoredEMEstimator(**RC_EM_PARAMS)
    est.fit(data)
    return est, est.get_results()


def fit_ic_init_estimator(data: pd.DataFrame) -> tuple[IntervalCensoredInitEstimator, dict]:
    """Fit IC Init (Midpoint) estimator and return results."""
    est = IntervalCensoredInitEstimator(**IC_INIT_PARAMS)
    est.fit(data, L_col="L", R_col="R")
    return est, est.get_results()


def fit_ic_em_estimator(data: pd.DataFrame) -> tuple[IntervalCensoredEMEstimator, dict]:
    """Fit IC EM estimator and return results."""
    est = IntervalCensoredEMEstimator(**IC_EM_PARAMS)
    est.fit(data)
    return est, est.get_results()


def serialize_results(results: dict) -> dict:
    """Convert numpy arrays to lists for JSON serialization."""
    serialized = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, dict):
            serialized[key] = serialize_results(value)
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value) if isinstance(value, np.floating) else int(value)
        else:
            serialized[key] = value
    return serialized


def save_json(data: dict, filename: str) -> None:
    """Save dict to JSON file."""
    filepath = FIXTURES_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Generating regression test fixtures")
    print("=" * 60)
    
    # Generate and save synthetic data
    print("\n1. Generating synthetic data...")
    rc_data = generate_right_censored_data()
    ic_data = generate_interval_censored_data()
    
    save_json({"T": rc_data["T"].tolist(), "Delta": rc_data["Delta"].tolist()}, 
              "synthetic_right_censored.json")
    save_json({"L": ic_data["L"].tolist(), "R": ic_data["R"].tolist()}, 
              "synthetic_interval_censored.json")
    
    print(f"   RC data: n={len(rc_data)}, censored={len(rc_data) - rc_data['Delta'].sum()}")
    print(f"   IC data: n={len(ic_data)}")
    
    # Fit RC estimators and save results
    print("\n2. Fitting RC Init (IPCW) estimator...")
    rc_init_est, rc_init_results = fit_rc_init_estimator(rc_data)
    
    print("\n3. Fitting RC EM estimator...")
    rc_em_est, rc_em_results = fit_rc_em_estimator(rc_data)
    
    # Compute RC metrics
    print("\n4. Computing RC metrics...")
    rc_init_ll = float(incomplete_loglik(rc_init_est, rc_data, time_col="T", delta_col="Delta"))
    rc_em_ll = float(incomplete_loglik(rc_em_est, rc_data, time_col="T", delta_col="Delta"))
    
    # Get density at evaluation points
    eval_points = np.linspace(0.01, 0.99, 50)
    rc_init_density = rc_init_est.get_density_at_points(eval_points).tolist()
    rc_em_density = rc_em_est.get_density_at_points(eval_points).tolist()
    
    # Save RC results
    rc_expected = {
        "params": {
            "init": RC_INIT_PARAMS,
            "em": RC_EM_PARAMS,
        },
        "init": {
            "results": serialize_results(rc_init_results),
            "incomplete_loglik": rc_init_ll,
            "density_at_eval_points": rc_init_density,
            "eval_points": eval_points.tolist(),
        },
        "em": {
            "results": serialize_results(rc_em_results),
            "incomplete_loglik": rc_em_ll,
            "density_at_eval_points": rc_em_density,
            "eval_points": eval_points.tolist(),
            "em_iterations": rc_em_results.get("em_iterations"),
            "em_converged": rc_em_results.get("em_converged"),
        },
    }
    save_json(rc_expected, "expected_results_rc.json")
    
    print(f"   RC Init: n_knots={rc_init_results.get('n_selected_knots')}, LL={rc_init_ll:.4f}")
    print(f"   RC EM: n_knots={rc_em_results.get('n_selected_knots')}, LL={rc_em_ll:.4f}, "
          f"iters={rc_em_results.get('em_iterations')}")
    
    # Fit IC estimators and save results
    print("\n5. Fitting IC Init (Midpoint) estimator...")
    ic_init_est, ic_init_results = fit_ic_init_estimator(ic_data)
    
    print("\n6. Fitting IC EM estimator...")
    ic_em_est, ic_em_results = fit_ic_em_estimator(ic_data)
    
    # Compute IC metrics
    print("\n7. Computing IC metrics...")
    ic_init_ll = float(incomplete_loglik_interval(ic_init_est, ic_data, L_col="L", R_col="R"))
    ic_em_ll = float(incomplete_loglik_interval(ic_em_est, ic_data, L_col="L", R_col="R"))
    
    # Get density at evaluation points
    ic_init_density = ic_init_est.get_density_at_points(eval_points).tolist()
    ic_em_density = ic_em_est.get_density_at_points(eval_points).tolist()
    
    # Save IC results
    ic_expected = {
        "params": {
            "init": IC_INIT_PARAMS,
            "em": IC_EM_PARAMS,
        },
        "init": {
            "results": serialize_results(ic_init_results),
            "incomplete_loglik": ic_init_ll,
            "density_at_eval_points": ic_init_density,
            "eval_points": eval_points.tolist(),
        },
        "em": {
            "results": serialize_results(ic_em_results),
            "incomplete_loglik": ic_em_ll,
            "density_at_eval_points": ic_em_density,
            "eval_points": eval_points.tolist(),
            "em_iterations": ic_em_results.get("em_iterations"),
            "em_converged": ic_em_results.get("em_converged"),
        },
    }
    save_json(ic_expected, "expected_results_ic.json")
    
    print(f"   IC Init: n_knots={ic_init_results.get('n_selected_knots')}, LL={ic_init_ll:.4f}")
    print(f"   IC EM: n_knots={ic_em_results.get('n_selected_knots')}, LL={ic_em_ll:.4f}, "
          f"iters={ic_em_results.get('em_iterations')}")
    
    print("\n" + "=" * 60)
    print("Fixture generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
