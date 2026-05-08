import numpy as np
import pandas as pd

from haldensity.censoring.right.estimators import RightCensoredInitEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.targeting import (
    RightCensoredEntropyTargetLearner,
    right_censored_entropy_estimand_variance,
    right_censored_entropy_targeting_M_step,
)
from haldensity.targeting.right_censored_entropy.learner import (
    _compute_entropy_direction_on_grid,
)
from haldensity.targeting.right_censored_survival.learner import (
    RCCensoringCache,
    RCTargetGrid,
)


def _make_observed_data(n: int = 64, seed: int = 401) -> pd.DataFrame:
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


def test_right_censored_entropy_targeting_exports_and_shape():
    data = _make_observed_data()
    estimator, km = _fit_initial_estimator(data)

    learner = RightCensoredEntropyTargetLearner()
    assert isinstance(learner, RightCensoredEntropyTargetLearner)

    targeted = right_censored_entropy_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        km=km,
        mode="one_step",
        store_pointwise_arrays=False,
    )

    assert len(targeted["summary"]) == 1
    assert len(targeted["pointwise_fits"]) == 1
    assert targeted["metadata"]["api_version"] == "entropy_v1"
    assert targeted["metadata"]["target"] == "entropy"
    assert targeted["censoring_cache"]["km"] is km

    summary = targeted["summary"].iloc[0]
    assert np.isfinite(summary["entropy_init"])
    assert np.isfinite(summary["entropy_final"])
    assert np.isfinite(summary["standard_error"])
    assert np.isclose(summary["exact_eif_mean_initial_stage"], summary["eif_mean_initial_stage"])
    assert np.isclose(summary["exact_threshold_initial"], summary["threshold_initial"])
    assert np.isclose(summary["exact_eif_mean_final"], summary["eif_mean_final"])
    assert np.isclose(summary["exact_threshold_final"], summary["threshold_final"])

    variances = right_censored_entropy_estimand_variance(targeted, observed_data=data)
    assert variances.shape == (1,)
    assert np.all(np.isfinite(variances))


def test_entropy_direction_no_censoring_reduces_to_centered_entropy_gradient():
    grid = np.array([0.25, 0.75])
    delta_j = np.array([0.5, 0.5])
    density = np.array([0.5, 1.5])
    target_grid = RCTargetGrid(
        t0=float("nan"),
        grid_edges=np.array([0.0, 0.5, 1.0]),
        grid_midpoints=grid,
        delta_j=delta_j,
        density_grid=density,
        log_density_grid=np.log(density),
        survival_grid=np.array([1.0, 0.75]),
        edge_survival=np.array([1.0, 0.75, 0.0]),
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

    raw_direction, centered_direction, _ = _compute_entropy_direction_on_grid(
        target_grid,
        cache,
        survival_clip=1e-8,
        targeting_gbar_floor=1e-6,
    )

    psi = np.sum((-np.log(density)) * density * delta_j)
    expected_raw = -np.log(density) - psi
    expected_centered = expected_raw
    assert np.allclose(raw_direction, expected_raw)
    assert np.allclose(centered_direction, expected_centered)


def test_entropy_direction_uses_target_constant_jump_correction():
    grid = np.array([0.25, 0.75])
    delta_j = np.array([0.5, 0.5])
    density = np.array([0.5, 1.5])
    target_grid = RCTargetGrid(
        t0=float("nan"),
        grid_edges=np.array([0.0, 0.5, 1.0]),
        grid_midpoints=grid,
        delta_j=delta_j,
        density_grid=density,
        log_density_grid=np.log(density),
        survival_grid=np.array([1.0, 0.75]),
        edge_survival=np.array([1.0, 0.75, 0.0]),
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

    raw_direction, _, details = _compute_entropy_direction_on_grid(
        target_grid,
        cache,
        survival_clip=1e-8,
        targeting_gbar_floor=1e-6,
    )

    phi_grid = -np.log(density)
    psi = np.sum(phi_grid * density * delta_j)
    expected_tail_entropy = phi_grid[1]
    expected_increment = expected_tail_entropy * 0.2 / (0.8**2) - psi * 0.2 / 0.8
    expected_cumulative = np.array([0.0, expected_increment])
    expected_raw = np.array([phi_grid[0] / 1.0 - psi, phi_grid[1] / 0.8 - psi]) - expected_cumulative

    assert np.allclose(details["tail_entropy_jump"], [expected_tail_entropy])
    assert np.allclose(details["target_constant_jump_correction"], [psi * 0.2 / 0.8])
    assert np.allclose(details["jump_increments"], [expected_increment])
    assert np.allclose(details["cumulative_jump_term_grid"], expected_cumulative)
    assert np.allclose(raw_direction, expected_raw)


def test_right_censored_entropy_auto_gate_can_skip_initial_targeting():
    data = _make_observed_data(seed=403)
    estimator, km = _fit_initial_estimator(data)

    targeted = right_censored_entropy_targeting_M_step(
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
    assert summary["status_one_step"] == "skipped_initial_gate"
    assert np.isclose(summary["entropy_init"], summary["entropy_final"])
    assert np.isclose(summary["eif_mean_initial_stage"], summary["eif_mean_final"])
