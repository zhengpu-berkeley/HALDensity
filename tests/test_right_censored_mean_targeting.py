import numpy as np
import pandas as pd

from haldensity.censoring.right.estimators import RightCensoredInitEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.targeting import (
    RightCensoredMeanTargetLearner,
    right_censored_mean_estimand_variance,
    right_censored_mean_targeting_M_step,
)
from haldensity.targeting.right_censored_mean.learner import _compute_mean_direction_on_grid
from haldensity.targeting.right_censored_survival.learner import (
    RCCensoringCache,
    RCTargetGrid,
)


def _make_observed_data(n: int = 64, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    event = np.clip(rng.beta(2.2, 2.5, size=n), 1e-3, 0.999)
    censor = rng.uniform(0.05, 0.95, size=n)
    observed = np.minimum(event, censor)
    delta = (event <= censor).astype(int)
    data = pd.DataFrame({"T": observed, "Delta": delta})
    assert int(delta.sum()) > 8
    return data


def _fit_initial_estimator(data: pd.DataFrame) -> tuple[RightCensoredInitEstimator, KaplanMeier]:
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    t_vals = data["T"].to_numpy(dtype=float)
    delta_vals = data["Delta"].to_numpy(dtype=int)
    weights = compute_ipcw_weights(
        t_vals,
        delta_vals,
        lambda t: np.atleast_1d(km.predict(t)),
        clip=1e-6,
    )
    uncensored_mask = delta_vals == 1
    uncensored_df = pd.DataFrame({"W1": t_vals[uncensored_mask]})

    estimator = RightCensoredInitEstimator(
        tol=1e-8,
        norm_constraint=25.0,
        n_grid_points=60,
        basis_order=1,
        solver="SCS",
        use_secondary_solver=True,
    )
    estimator.fit(uncensored_df, sample_weights=weights[uncensored_mask])
    return estimator, km


class _FakeKM:
    def __init__(self, jump_times: np.ndarray, gbar_right: np.ndarray) -> None:
        self.jump_times = np.asarray(jump_times, dtype=float)
        self.gbar_right = np.asarray(gbar_right, dtype=float)

    def predict(self, t):
        ts = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.jump_times, ts, side="right") - 1
        idx = np.clip(idx, -1, len(self.gbar_right) - 1)
        out = np.where(idx >= 0, self.gbar_right[idx], 1.0)
        return out if isinstance(t, (list, np.ndarray)) else float(out)


def test_right_censored_mean_targeting_exports_and_shape():
    data = _make_observed_data()
    estimator, km = _fit_initial_estimator(data)

    learner = RightCensoredMeanTargetLearner()
    assert isinstance(learner, RightCensoredMeanTargetLearner)

    targeted = right_censored_mean_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        km=km,
        mode="one_step",
        store_pointwise_arrays=False,
    )

    assert len(targeted["summary"]) == 1
    assert len(targeted["pointwise_fits"]) == 1
    assert targeted["metadata"]["api_version"] == "mean_v1"
    assert targeted["metadata"]["target"] == "mean"
    assert targeted["censoring_cache"]["km"] is km

    summary = targeted["summary"].iloc[0]
    assert 0.0 <= summary["mu_init"] <= 1.0
    assert 0.0 <= summary["mu_one_step"] <= 1.0
    assert 0.0 <= summary["mu_final"] <= 1.0
    assert np.isfinite(summary["standard_error"])
    assert np.isfinite(summary["estimand_variance"])
    assert targeted["pointwise_fits"][0]["target"] == "mean"


def test_right_censored_mean_targeting_store_arrays_and_initial_ci_convention():
    data = _make_observed_data(seed=103)
    estimator, km = _fit_initial_estimator(data)

    targeted = right_censored_mean_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        km=km,
        mode="one_step",
        clip=0.2,
        store_pointwise_arrays=True,
    )
    summary = targeted["summary"].iloc[0]
    pointwise_fit = targeted["pointwise_fits"][0]

    assert "estimated_density" in pointwise_fit
    assert "raw_direction" in pointwise_fit
    assert "centered_direction" in pointwise_fit
    assert "eic_values" in pointwise_fit
    assert "clip_active_eic_mask" in pointwise_fit
    density_mass = float(np.sum(pointwise_fit["estimated_density"] * pointwise_fit["delta_j"]))
    assert np.isclose(density_mass, 1.0, atol=1e-6)

    assert np.isclose(summary["estimand_variance"], summary["estimand_variance_initial_stage"])
    assert np.isclose(summary["standard_error"], summary["standard_error_initial_stage"])
    expected_lower = max(0.0, summary["psi_star"] - 1.96 * summary["standard_error_initial_stage"])
    expected_upper = min(1.0, summary["psi_star"] + 1.96 * summary["standard_error_initial_stage"])
    assert np.isclose(summary["ci_lower"], expected_lower)
    assert np.isclose(summary["ci_upper"], expected_upper)
    assert int(summary["clip_active_eic_excluded_count"]) > 0
    assert int(summary["clip_active_eic_included_count"]) + int(
        summary["clip_active_eic_excluded_count"]
    ) == len(data)
    assert int(summary["positivity_eic_excluded_count"]) == int(
        summary["clip_active_eic_excluded_count"]
    )
    assert np.isclose(
        summary["positivity_eic_excluded_rate"],
        summary["clip_active_eic_excluded_rate"],
    )
    eic_values = np.asarray(pointwise_fit["eic_values"], dtype=float)
    include_mask = np.asarray(pointwise_fit["clip_active_eic_include_mask"], dtype=bool)
    expected_filtered_se = np.sqrt(np.var(eic_values[include_mask], ddof=1) / include_mask.sum())
    assert np.isclose(summary["standard_error_final"], expected_filtered_se)
    assert summary["standard_error_all_eic_final"] >= summary["standard_error_final"]

    default_var = right_censored_mean_estimand_variance(targeted, observed_data=data)
    initial_var = right_censored_mean_estimand_variance(
        targeted,
        observed_data=data,
        which="initial_stage",
    )
    one_step_var = right_censored_mean_estimand_variance(
        targeted,
        observed_data=data,
        which="one_step",
    )
    final_var = right_censored_mean_estimand_variance(
        targeted,
        observed_data=data,
        which="final",
    )
    assert np.allclose(default_var, [summary["estimand_variance_initial_stage"]])
    assert np.allclose(initial_var, [summary["estimand_variance_initial_stage"]])
    assert np.allclose(one_step_var, [summary["estimand_variance_one_step"]])
    assert np.allclose(final_var, [summary["estimand_variance_final"]])


def test_right_censored_mean_direction_no_censoring_reduces_to_centered_mean_gradient():
    grid = np.array([0.25, 0.75])
    delta_j = np.array([0.5, 0.5])
    density = np.array([1.0, 1.0])
    survival = np.array([1.0, 0.5])
    target_grid = RCTargetGrid(
        t0=float("nan"),
        grid_edges=np.array([0.0, 0.5, 1.0]),
        grid_midpoints=grid,
        delta_j=delta_j,
        density_grid=density,
        log_density_grid=np.log(density),
        survival_grid=survival,
        edge_survival=np.array([1.0, 0.5, 0.0]),
        t0_inserted=False,
    )
    cache = RCCensoringCache(
        km=_FakeKM(np.array([0.2, 0.8]), np.array([1.0, 1.0])),
        jump_times=np.array([0.2, 0.8]),
        gbar_right=np.array([1.0, 1.0]),
        gbar_left=np.array([1.0, 1.0]),
        jump_masses=np.array([0.0, 0.0]),
        clip=1e-6,
    )

    raw_direction, centered_direction, details = _compute_mean_direction_on_grid(
        target_grid,
        cache,
        survival_clip=1e-8,
    )

    expected_centered = grid - 0.5
    assert np.allclose(raw_direction, grid)
    assert np.allclose(centered_direction, expected_centered)
    assert np.isclose(details["raw_mean"], 0.5)


def test_right_censored_mean_direction_uses_km_jump_sum():
    grid = np.array([0.25, 0.75])
    delta_j = np.array([0.5, 0.5])
    density = np.array([1.0, 1.0])
    survival = np.array([1.0, 0.5])
    target_grid = RCTargetGrid(
        t0=float("nan"),
        grid_edges=np.array([0.0, 0.5, 1.0]),
        grid_midpoints=grid,
        delta_j=delta_j,
        density_grid=density,
        log_density_grid=np.log(density),
        survival_grid=survival,
        edge_survival=np.array([1.0, 0.5, 0.0]),
        t0_inserted=False,
    )
    cache = RCCensoringCache(
        km=_FakeKM(np.array([0.5]), np.array([0.8])),
        jump_times=np.array([0.5]),
        gbar_right=np.array([0.8]),
        gbar_left=np.array([1.0]),
        jump_masses=np.array([0.2]),
        clip=1e-6,
    )

    raw_direction, _, details = _compute_mean_direction_on_grid(
        target_grid,
        cache,
        survival_clip=1e-8,
    )

    expected_increment = 0.75 * 0.2 / (0.8**2)
    expected_cumulative = np.array([0.0, expected_increment])
    expected_raw = np.array([0.25 / 1.0, 0.75 / 0.8]) - expected_cumulative

    assert np.allclose(details["tail_mean_jump"], [0.75])
    assert np.allclose(details["jump_increments"], [expected_increment])
    assert np.allclose(details["cumulative_jump_term_grid"], expected_cumulative)
    assert np.allclose(raw_direction, expected_raw)


def test_right_censored_mean_one_step_matches_iterative_with_one_iteration():
    data = _make_observed_data(seed=107)
    estimator, km = _fit_initial_estimator(data)

    one_step = right_censored_mean_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        km=km,
        mode="one_step",
        store_pointwise_arrays=False,
    )
    iterative_one = right_censored_mean_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        km=km,
        mode="iterative",
        max_iter=1,
        store_pointwise_arrays=False,
    )

    one_summary = one_step["summary"].iloc[0]
    iter_summary = iterative_one["summary"].iloc[0]
    assert np.isclose(one_summary["psi_star"], iter_summary["psi_star"])
    assert np.isclose(one_summary["eif_mean_final"], iter_summary["eif_mean_final"])
    assert int(iter_summary["n_iterations"]) == 1


def test_right_censored_mean_auto_gate_can_stop_after_one_step():
    data = _make_observed_data(seed=109)
    estimator, km = _fit_initial_estimator(data)

    targeted = right_censored_mean_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        km=km,
        mode="auto",
        min_score_tol=np.inf,
        max_iter=5,
        store_pointwise_arrays=False,
    )
    summary = targeted["summary"].iloc[0]

    assert int(summary["n_iterations"]) == 0
    assert not bool(summary["continued_past_one_step"])
    assert not bool(summary["used_iterative"])
    assert summary["status_one_step"] == "skipped_initial_gate"
    assert np.isclose(summary["mu_init"], summary["mu_final"])
    assert np.isclose(summary["eif_mean_initial_stage"], summary["eif_mean_final"])
