import numpy as np

from haldensity.censoring.right.comparison import (
    fit_right_censored_initial_estimator,
    run_right_censored_initial_estimator_experiment,
    simulate_beta_uniform_right_censored,
)
from haldensity.censoring.right.observed_mle import (
    RightCensoredObservedFISTAEstimator,
    RightCensoredObservedFPGDEstimator,
)


def _make_data(n: int = 48, seed: int = 17):
    sim = simulate_beta_uniform_right_censored(n=n, seed=seed)
    return sim["observed_data"], sim["truth"]


def test_right_censored_observed_estimators_fit_valid_density():
    data, _ = _make_data()

    fista = RightCensoredObservedFISTAEstimator(
        lam=0.02,
        n_iterations=60,
        ll_change_tol=1e-4,
        n_grid_points=80,
        basis_order=0,
    ).fit(data)
    fpgd = RightCensoredObservedFPGDEstimator(
        norm_constraint=8.0,
        n_iterations=80,
        ll_change_tol=1e-4,
        learning_rate=0.1,
        n_grid_points=80,
        basis_order=0,
    ).fit(data)

    for estimator in (fista, fpgd):
        _, density = estimator.get_density()
        density_mass = float(np.sum(density * estimator.delta_j))
        assert np.isclose(density_mass, 1.0, atol=1e-5)
        results = estimator.get_results()
        assert int(results["n_iterations_run"]) > 0
        assert "optimization_history" in results


def test_right_censored_method_specific_fit_interface():
    data, _ = _make_data(seed=21)
    method_overrides = {
        "M1": {"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
        "M2": {"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
        "M3": {"basis_order": [0], "lam": {"low": 0.01, "high": 0.05, "log": False}},
        "M4": {"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
    }

    fit_m1 = fit_right_censored_initial_estimator(
        "M1",
        data,
        cv_folds=2,
        n_trials=2,
        n_grid_points=80,
        random_state=2,
        param_overrides=method_overrides["M1"],
        init_tuner_kwargs={"silent": True, "use_conservative_adjustment": False},
    )
    fit_m2 = fit_right_censored_initial_estimator(
        "M2",
        data,
        cv_folds=2,
        n_trials=2,
        n_grid_points=80,
        random_state=2,
        param_overrides=method_overrides["M2"],
        init_tuner_kwargs={"silent": True, "use_conservative_adjustment": False},
    )

    assert fit_m1.metadata["validation_metric"] == "observed_loglik"
    assert fit_m2.metadata["validation_metric"] == "ipcw_loglik"
    assert fit_m1.estimator.is_fitted
    assert fit_m2.estimator.is_fitted


def test_right_censored_experiment_runner_supports_tmle_for_m1_to_m4():
    data, truth = _make_data(seed=31)
    target_times = np.array([0.25, 0.5])
    method_overrides = {
        "M1": {"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
        "M2": {"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
        "M3": {"basis_order": [0], "lam": {"low": 0.01, "high": 0.05, "log": False}},
        "M4": {"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
    }

    result = run_right_censored_initial_estimator_experiment(
        data,
        truth=truth,
        target_times=target_times,
        cv_folds=2,
        n_trials=2,
        random_state=3,
        n_grid_points=80,
        method_param_overrides=method_overrides,
        init_tuner_kwargs={"silent": True, "use_conservative_adjustment": False},
        fista_kwargs={"n_iterations": 60, "ll_change_tol": 1e-4},
        fpgd_kwargs={"n_iterations": 80, "ll_change_tol": 1e-4, "learning_rate": 0.1},
        run_tmle=True,
        tmle_kwargs={"store_pointwise_arrays": False},
    )

    initial_summary = result["initial_summary"]
    tmle_summary = result["tmle_summary"]
    assert set(initial_summary["method"]) == {"M1", "M2", "M3", "M4"}
    assert set(tmle_summary["method"]) == {"M1", "M2", "M3", "M4"}
    assert len(tmle_summary) == 8
    assert np.all(np.isfinite(tmle_summary["psi_init"]))
    assert np.all(np.isfinite(tmle_summary["psi_star"]))
    assert np.all(np.isfinite(tmle_summary["epsilon"]))
    assert np.all(np.isfinite(tmle_summary["standard_error"]))
    assert np.all(np.isfinite(tmle_summary["targeted_abs_error"]))
