import numpy as np
import pandas as pd

from haldensity.censoring import (
    IntervalCensoredEMTuner,
    IntervalCensoredInitEstimator,
    KaplanMeier,
    RightCensoredEMTuner,
    RightCensoredInitEstimator,
    compute_ipcw_weights,
)
from haldensity.utils import TruncatedGMM


def _make_sampler() -> TruncatedGMM:
    return TruncatedGMM(
        components=[
            {"mean": 0.2, "std": 0.05, "lower": 0.0, "upper": 1.0},
            {"mean": 0.5, "std": 0.05, "lower": 0.0, "upper": 1.0},
            {"mean": 0.8, "std": 0.05, "lower": 0.0, "upper": 1.0},
        ],
        weights=[0.33, 0.34, 0.33],
    )


def _make_right_censored_data(n: int = 120, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampler = _make_sampler()
    t_event = sampler.generate_samples(n)
    t_cens = rng.uniform(0.0, 1.0, size=n)
    t_obs = np.minimum(t_event, t_cens)
    delta = (t_event <= t_cens).astype(int)
    return pd.DataFrame({"T": t_obs, "Delta": delta})


def _make_interval_censored_data(n: int = 120, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampler = _make_sampler()
    t_event = sampler.generate_samples(n)
    widths = rng.uniform(0.05, 0.15, size=n)
    left_shift = rng.uniform(0.0, 1.0, size=n) * widths
    left = np.clip(t_event - left_shift, 0.0, 1.0)
    right = np.clip(left + widths, 0.0, 1.0)
    right = np.maximum(right, left + 0.01)
    right = np.minimum(right, 1.0)
    return pd.DataFrame({"L": left, "R": right})


def test_right_censored_em_runs_with_zero_selected_knots() -> None:
    data = _make_right_censored_data()

    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    times = np.asarray(data["T"].values, dtype=float)
    delta = np.asarray(data["Delta"].values, dtype=int)
    weights = compute_ipcw_weights(times, delta, lambda t: np.atleast_1d(km.predict(t)))
    unc = delta == 1

    # Force Stage 1 to select no knots by pruning all knot coefficients.
    stage1 = RightCensoredInitEstimator(
        norm_constraint=10.0,
        n_grid_points=60,
        basis_order=2,   # {1, x, x^2, ...}
        tol=1e9,         # prune all knot terms
        solver="ECOS",
        use_secondary_solver=True,
    ).fit(pd.DataFrame({"W1": times[unc]}), sample_weights=weights[unc])

    assert stage1.grid_points_hal_selected is not None
    assert np.asarray(stage1.grid_points_hal_selected).size == 0

    # Stage 2 must remain parametric-only (no knots) and must not crash.
    result = RightCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=False,
        em_m_imputations=10,
        em_max_em_iter=3,
        em_tol=1e-4,
        em_norm_factor=1.5,  # treated as config; should not affect the "no knots" consistency
        n_grid_points=60,
        random_state=42,
        silent=True,
    ).optimize()

    theta = np.asarray(result.estimator.theta_hat, dtype=float)
    assert theta.shape[0] == 3  # intercept + x + x^2
    sel = getattr(result.estimator, "grid_points_hal_selected", None)
    assert sel is not None
    assert np.asarray(sel).size == 0


def test_interval_censored_em_runs_with_zero_selected_knots() -> None:
    data = _make_interval_censored_data()

    stage1 = IntervalCensoredInitEstimator(
        norm_constraint=10.0,
        n_grid_points=60,
        basis_order=2,
        tol=1e9,
        solver="ECOS",
        use_secondary_solver=True,
        include_intercept_in_constraint=False,
    ).fit(data, L_col="L", R_col="R")

    assert stage1.grid_points_hal_selected is not None
    assert np.asarray(stage1.grid_points_hal_selected).size == 0

    result = IntervalCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=False,
        em_m_imputations=10,
        em_max_em_iter=3,
        em_tol=1e-4,
        em_norm_factor=1.5,
        n_grid_points=60,
        random_state=42,
        silent=True,
        L_col="L",
        R_col="R",
    ).optimize()

    theta = np.asarray(result.estimator.theta_hat, dtype=float)
    assert theta.shape[0] == 3
    sel = getattr(result.estimator, "grid_points_hal_selected", None)
    assert sel is not None
    assert np.asarray(sel).size == 0


