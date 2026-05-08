import numpy as np

from haldensity.censoring.right.comparison import (
    fit_right_censored_cv_ipcw_l1mle_plugin,
    fit_right_censored_cv_ipcw_l1mle_undersmoothed_plugin,
    simulate_beta_uniform_right_censored,
)


def _common_kwargs() -> dict:
    return {
        "cv_folds": 2,
        "n_trials": 2,
        "random_state": 7,
        "n_grid_points": 80,
        "stage1_param_overrides": {
            "basis_order": [0],
            "norm_constraint": {"low": 3.0, "high": 6.0, "log": False},
        },
        "norm_constraint_factors": [1.0, 1.1],
        "l1_kwargs": {
            "n_grid_points": 80,
            "learning_rate": 0.1,
            "n_iterations": 120,
            "ll_change_tol": 1e-3,
            "history_every": 40,
        },
    }


def test_cv_ipcw_l1mle_plugin_api_returns_no_undersmoothing_summary() -> None:
    sim = simulate_beta_uniform_right_censored(n=64, seed=17)
    data = sim["observed_data"]
    truth = sim["truth"]
    points = np.array([0.25, 0.5], dtype=float)

    result = fit_right_censored_cv_ipcw_l1mle_plugin(
        data,
        targeting_points=points,
        truth=truth,
        stage1_tuner_kwargs={
            "silent": True,
            "use_conservative_adjustment": False,
        },
        **_common_kwargs(),
    )

    assert np.isclose(result["no_undersmoothing_factor"], 1.0)
    summary = result["no_undersmoothing_summary"]
    assert len(summary) == len(points)
    assert {
        "t0",
        "psi_init",
        "estimated_density",
        "truth_survival",
        "truth_density",
        "abs_error_plugin",
        "density_abs_error",
        "norm_factor",
        "norm_constraint",
    }.issubset(summary.columns)
    assert np.all(np.isfinite(summary["psi_init"]))
    assert np.all(np.isfinite(summary["estimated_density"]))
    assert np.allclose(summary["norm_factor"].to_numpy(dtype=float), 1.0)


def test_cv_ipcw_l1mle_undersmoothed_api_returns_score_solving_diagnostics() -> None:
    sim = simulate_beta_uniform_right_censored(n=72, seed=23)
    data = sim["observed_data"]
    truth = sim["truth"]
    points = np.array([0.25, 0.5, 0.75], dtype=float)

    result = fit_right_censored_cv_ipcw_l1mle_undersmoothed_plugin(
        data,
        targeting_points=points,
        truth=truth,
        stage1_tuner_kwargs={
            "silent": True,
            "use_conservative_adjustment": True,
            "conservative_selection_rule": "legacy_percent_sd",
            "conservative_k_percent": 0.05,
        },
        **_common_kwargs(),
    )

    stage1_meta = result["metadata"]["stage1_metadata"]
    assert stage1_meta.get("conservative_selection_rule") == "legacy_percent_sd"
    assert stage1_meta.get("conservative_params") is not None

    # Both estimators are available separately:
    assert "no_undersmoothing_summary" in result
    assert "undersmoothed_summary" in result

    selected = result["undersmoothed_summary"]
    assert len(selected) == len(points)
    assert {
        "t0",
        "selected_norm_factor",
        "psi_init",
        "psi_star_targeted_diagnostic",
        "eif_mean_final",
        "threshold_final",
        "passes_final",
        "selection_status",
    }.issubset(selected.columns)
    assert selected["selected_norm_factor"].between(1.0, 1.1).all()
    assert set(selected["selection_status"]).issubset(
        {"smallest_passing_factor", "closest_failing_factor"}
    )

    long_df = result["undersmoothed_targeting_summary_long"]
    assert len(long_df) == len(points) * 2
    assert {
        "norm_factor",
        "psi_init",
        "psi_star",
        "eif_mean_final",
        "threshold_final",
        "passes_final",
    }.issubset(long_df.columns)
