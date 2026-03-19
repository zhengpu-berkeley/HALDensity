import numpy as np
import pandas as pd

from haldensity.censoring.right.estimators import RightCensoredInitEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.targeting import (
    right_censored_survival_estimand_variance,
    right_censored_survival_targeting_M_step,
    right_censored_survival_targeting_M_step_v2,
)


def _make_observed_data(n: int = 64, seed: int = 7) -> pd.DataFrame:
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
    T_vals = data["T"].to_numpy(dtype=float)
    delta_vals = data["Delta"].to_numpy(dtype=int)
    weights = compute_ipcw_weights(
        T_vals,
        delta_vals,
        lambda t: np.atleast_1d(km.predict(t)),
        clip=1e-6,
    )
    uncensored_mask = delta_vals == 1
    uncensored_df = pd.DataFrame({"W1": T_vals[uncensored_mask]})

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


def test_right_censored_survival_targeting_scalar_and_vector_shapes():
    data = _make_observed_data()
    estimator, km = _fit_initial_estimator(data)

    scalar_fit = right_censored_survival_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=0.5,
        km=km,
        store_pointwise_arrays=False,
    )
    assert scalar_fit["targeting_points"].shape == (1,)
    assert len(scalar_fit["summary"]) == 1
    assert len(scalar_fit["pointwise_fits"]) == 1
    assert scalar_fit["censoring_cache"]["km"] is km
    assert scalar_fit["metadata"]["store_pointwise_arrays"] is False

    pointwise_fit = scalar_fit["pointwise_fits"][0]
    assert "estimated_density" not in pointwise_fit
    assert "diagnostics" not in pointwise_fit
    assert pointwise_fit["target_grid_augmented_with_t0"] in (True, False)

    vector_fit = right_censored_survival_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=np.array([0.2, 0.4, 0.6, 0.8]),
        store_pointwise_arrays=False,
    )
    assert vector_fit["targeting_points"].shape == (4,)
    assert len(vector_fit["summary"]) == 4
    assert len(vector_fit["pointwise_fits"]) == 4

    variances = right_censored_survival_estimand_variance(vector_fit, observed_data=data)
    assert variances.shape == (4,)
    assert np.all(np.isfinite(variances))


def test_right_censored_survival_targeting_store_arrays_and_numerics():
    data = _make_observed_data(seed=11)
    estimator, km = _fit_initial_estimator(data)

    targeted = right_censored_survival_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=np.array([0.25, 0.5]),
        km=km,
        store_pointwise_arrays=True,
    )

    for pointwise_fit in targeted["pointwise_fits"]:
        assert "estimated_density" in pointwise_fit
        assert "targeted_survival_grid" in pointwise_fit
        assert "raw_direction" in pointwise_fit
        assert "centered_direction" in pointwise_fit
        assert "eic_values" in pointwise_fit
        assert "diagnostics" in pointwise_fit
        assert "km" not in pointwise_fit
        assert "estimator" not in pointwise_fit
        assert "observed_data" not in pointwise_fit

        density_mass = float(
            np.sum(pointwise_fit["estimated_density"] * pointwise_fit["delta_j"])
        )
        assert np.isclose(density_mass, 1.0, atol=1e-6)
        assert 0.0 <= pointwise_fit["psi_init"] <= 1.0
        assert 0.0 <= pointwise_fit["psi_star"] <= 1.0
        assert np.isfinite(pointwise_fit["standard_error"])
        assert np.isfinite(pointwise_fit["estimand_variance"])
        if pointwise_fit["solve_method"] == "score_root":
            assert abs(pointwise_fit["score_at_solution"]) < 1e-6


def test_right_censored_survival_targeting_v2_shape_and_stage_fields():
    data = _make_observed_data(seed=13)
    estimator, km = _fit_initial_estimator(data)

    targeted_v2 = right_censored_survival_targeting_M_step_v2(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=np.array([0.25, 0.5]),
        km=km,
        mode="auto",
        one_step_eif_gate=1e-8,
        store_pointwise_arrays=False,
    )

    assert targeted_v2["targeting_points"].shape == (2,)
    assert len(targeted_v2["summary"]) == 2
    assert len(targeted_v2["pointwise_fits"]) == 2
    assert targeted_v2["metadata"]["api_version"] == "v2"
    assert targeted_v2["metadata"]["mode"] == "auto"

    required_summary_cols = {
        "psi_one_step",
        "psi_final",
        "eif_mean_one_step",
        "eif_mean_final",
        "used_iterative",
        "decision_reason",
        "n_iterations",
        "stop_reason",
    }
    assert required_summary_cols.issubset(set(targeted_v2["summary"].columns))

    for pointwise_fit in targeted_v2["pointwise_fits"]:
        assert "initial_stage" in pointwise_fit
        assert "one_step_stage" in pointwise_fit
        assert "final_stage" in pointwise_fit
        assert "used_iterative" in pointwise_fit
        assert "decision_reason" in pointwise_fit
        assert "iteration_history" in pointwise_fit


def test_right_censored_survival_targeting_v2_mode_one_step_matches_v1():
    data = _make_observed_data(seed=17)
    estimator, km = _fit_initial_estimator(data)
    target_points = np.array([0.2, 0.4, 0.6])

    targeted_v1 = right_censored_survival_targeting_M_step(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=target_points,
        km=km,
        store_pointwise_arrays=False,
    )
    targeted_v2 = right_censored_survival_targeting_M_step_v2(
        initial_estimator=estimator,
        observed_data=data,
        targeting_points=target_points,
        km=km,
        mode="one_step",
        store_pointwise_arrays=False,
    )

    v1_summary = targeted_v1["summary"].sort_values("t0").reset_index(drop=True)
    v2_summary = targeted_v2["summary"].sort_values("t0").reset_index(drop=True)

    assert np.allclose(v2_summary["psi_star"].to_numpy(), v1_summary["psi_star"].to_numpy())
    assert np.allclose(v2_summary["eif_mean"].to_numpy(), v1_summary["eif_mean"].to_numpy())
    assert np.all(v2_summary["used_iterative"].to_numpy(dtype=bool) == np.array([False, False, False]))
