import numpy as np
import pandas as pd

from haldensity.censoring.right.estimators import RightCensoredInitEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.targeting import (
    RightCensoredRMSTTargetLearner,
    right_censored_rmst_estimand_variance,
    right_censored_rmst_targeting_M_step,
)
from haldensity.targeting.right_censored_rmst.learner import _compute_rmst_direction_on_grid
from haldensity.targeting.right_censored_survival.learner import (
    RCCensoringCache,
    RCTargetGrid,
)


def _make_observed_data(n: int = 64, seed: int = 301) -> pd.DataFrame:
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


def test_right_censored_rmst_targeting_exports_and_shape():
    data = _make_observed_data()
    estimator, km = _fit_initial_estimator(data)

    learner = RightCensoredRMSTTargetLearner()
    assert isinstance(learner, RightCensoredRMSTTargetLearner)

    targeted = right_censored_rmst_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=np.array([0.25, 0.5, 1.0]),
        km=km,
        mode="one_step",
        store_pointwise_arrays=False,
    )

    assert targeted["targeting_points"].shape == (3,)
    assert len(targeted["summary"]) == 3
    assert len(targeted["pointwise_fits"]) == 3
    assert targeted["metadata"]["api_version"] == "rmst_v1"
    assert targeted["metadata"]["target"] == "rmst"
    assert targeted["censoring_cache"]["km"] is km

    summary = targeted["summary"].sort_values("tau").reset_index(drop=True)
    assert np.all(summary["rmst_init"].to_numpy(dtype=float) > 0.0)
    assert np.all(summary["rmst_final"].to_numpy(dtype=float) > 0.0)
    assert np.all(np.isfinite(summary["standard_error"].to_numpy(dtype=float)))

    variances = right_censored_rmst_estimand_variance(targeted, observed_data=data)
    assert variances.shape == (3,)
    assert np.all(np.isfinite(variances))


def test_rmst_direction_no_censoring_reduces_to_centered_bounded_mean_gradient():
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

    raw_direction, centered_direction, details = _compute_rmst_direction_on_grid(
        target_grid,
        cache,
        tau=0.5,
        survival_clip=1e-8,
        targeting_gbar_floor=1e-6,
    )

    expected_raw = np.array([0.25, 0.5])
    expected_centered = expected_raw - 0.375
    assert np.allclose(raw_direction, expected_raw)
    assert np.allclose(centered_direction, expected_centered)
    assert np.isclose(details["psi"], 0.375)


def test_right_censored_rmst_auto_gate_can_skip_initial_targeting():
    data = _make_observed_data(seed=303)
    estimator, km = _fit_initial_estimator(data)

    targeted = right_censored_rmst_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=np.array([0.3, 0.6]),
        km=km,
        mode="auto",
        min_score_tol=np.inf,
        max_iter=5,
        store_pointwise_arrays=False,
    )
    summary = targeted["summary"].sort_values("tau").reset_index(drop=True)

    assert np.all(summary["n_iterations"].to_numpy(dtype=int) == 0)
    assert np.all(summary["status_one_step"] == "skipped_initial_gate")
    assert np.allclose(summary["rmst_init"].to_numpy(), summary["rmst_final"].to_numpy())
    assert np.allclose(
        summary["eif_mean_initial_stage"].to_numpy(),
        summary["eif_mean_final"].to_numpy(),
    )
