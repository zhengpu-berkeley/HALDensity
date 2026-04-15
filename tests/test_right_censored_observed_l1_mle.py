import numpy as np
import pandas as pd
import pytest

from haldensity.censoring.right import (
    KaplanMeier,
    RightCensoredInitEstimator,
    RightCensoredObservedL1MLE,
    compute_ipcw_weights,
    right_censored_observed_loglik_and_gradient,
)
from haldensity.censoring.right.comparison import simulate_beta_uniform_right_censored


def _make_data(n: int = 64, seed: int = 13) -> pd.DataFrame:
    sim = simulate_beta_uniform_right_censored(n=n, seed=seed)
    return sim["observed_data"]


def _fit_stage1_like_estimator(
    data: pd.DataFrame,
    *,
    norm_constraint: float = 8.0,
    n_grid_points: int = 80,
) -> RightCensoredInitEstimator:
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    t_vals = np.asarray(data["T"].values, dtype=float)
    delta_vals = np.asarray(data["Delta"].values, dtype=int)
    weights = compute_ipcw_weights(t_vals, delta_vals, lambda t: np.atleast_1d(km.predict(t)))

    uncensored = delta_vals == 1
    est = RightCensoredInitEstimator(
        norm_constraint=norm_constraint,
        n_grid_points=n_grid_points,
        basis_order=0,
        include_intercept_in_constraint=False,
        use_secondary_solver=True,
    )
    est.fit(pd.DataFrame({"W1": t_vals[uncensored]}), sample_weights=weights[uncensored])
    support = np.asarray(est.grid_points_hal_selected, dtype=float)
    if support.size == 0:
        raise AssertionError("test setup produced an empty Stage 1 support")
    return est


def _compress_theta_to_selected_support(estimator: RightCensoredInitEstimator) -> np.ndarray:
    support = np.asarray(estimator.grid_points_hal_selected, dtype=float)
    all_knots = np.asarray(estimator._grid_points_hal, dtype=float)
    theta_full = np.asarray(estimator.theta_hat, dtype=float)
    basis_order = int(estimator.basis_order)
    poly_cols = basis_order if basis_order > 0 else 0
    knot_start = 1 + poly_cols

    theta_selected = np.zeros(knot_start + support.size, dtype=float)
    theta_selected[:knot_start] = theta_full[:knot_start]
    for i, knot in enumerate(support):
        matches = np.where(np.isclose(all_knots, knot, atol=1e-10, rtol=0.0))[0]
        if matches.size == 0:
            raise AssertionError("selected support knot was not found in the full Stage 1 grid")
        theta_selected[knot_start + i] = theta_full[knot_start + int(matches[0])]
    return theta_selected


def test_right_censored_observed_l1_fit_preserves_fixed_support() -> None:
    data = _make_data()
    stage1 = _fit_stage1_like_estimator(data)
    working_grid = np.asarray(stage1.grid_points_hal_selected, dtype=float).copy()
    warm_start = _compress_theta_to_selected_support(stage1)

    est = RightCensoredObservedL1MLE(
        working_grid_points=working_grid,
        norm_constraint=float(stage1.norm_constraint),
        basis_order=int(stage1.basis_order),
        n_grid_points=80,
        learning_rate=0.1,
        n_iterations=150,
        ll_change_tol=1e-4,
        include_intercept_in_constraint=False,
        warm_start_theta=warm_start,
    ).fit(data)

    _, density = est.get_density()
    density_mass = float(np.sum(density * est.delta_j))
    assert np.isclose(density_mass, 1.0, atol=1e-5)
    assert np.array_equal(np.asarray(est._grid_points_hal, dtype=float), working_grid)
    assert np.array_equal(np.asarray(est.grid_points_hal_selected, dtype=float), working_grid)

    results = est.get_results()
    assert int(results["n_iterations_run"]) > 0
    assert "optimization_history" in results
    assert float(results["penalized_l1_norm"]) <= float(stage1.norm_constraint) + 1e-6

    helper_ll, helper_grad = right_censored_observed_loglik_and_gradient(
        est.theta_hat,
        data,
        working_grid_points=working_grid,
        basis_order=int(est.basis_order),
        n_grid_points=int(est.n_grid_points),
    )
    method_ll, method_grad = est.observed_loglik_and_gradient(est.theta_hat, compute_grad=True)
    assert np.isclose(helper_ll, method_ll)
    assert np.allclose(helper_grad, method_grad)


def test_right_censored_observed_l1_keeps_full_support_when_coefficients_are_zero() -> None:
    data = _make_data(seed=29)
    working_grid = np.array([0.15, 0.35, 0.55, 0.75], dtype=float)

    est = RightCensoredObservedL1MLE(
        working_grid_points=working_grid,
        norm_constraint=0.0,
        basis_order=0,
        n_grid_points=80,
        learning_rate=0.1,
        n_iterations=80,
        ll_change_tol=1e-4,
        include_intercept_in_constraint=False,
    ).fit(data)

    results = est.get_results()
    assert np.array_equal(np.asarray(est._grid_points_hal, dtype=float), working_grid)
    assert np.array_equal(np.asarray(results["grid_points_hal_selected"], dtype=float), working_grid)
    assert np.all(est.theta_hat[1:] == 0.0)
    assert int(results["n_exact_zero_coeffs"]) >= working_grid.size
    assert int(results["n_active_knot_coeffs"]) == 0


def test_right_censored_observed_l1_allows_empty_fixed_support() -> None:
    data = _make_data(seed=31)
    working_grid = np.array([], dtype=float)
    warm_start = np.array([0.0], dtype=float)

    est = RightCensoredObservedL1MLE(
        working_grid_points=working_grid,
        norm_constraint=0.0,
        basis_order=0,
        n_grid_points=80,
        learning_rate=0.1,
        n_iterations=80,
        ll_change_tol=1e-4,
        include_intercept_in_constraint=False,
        warm_start_theta=warm_start,
    ).fit(data)

    _, density = est.get_density()
    density_mass = float(np.sum(density * est.delta_j))
    assert np.isclose(density_mass, 1.0, atol=1e-5)
    assert est.theta_hat.shape == (1,)
    assert np.asarray(est._grid_points_hal, dtype=float).size == 0
    assert np.asarray(est.grid_points_hal_selected, dtype=float).size == 0

    loglik, grad = right_censored_observed_loglik_and_gradient(
        est.theta_hat,
        data,
        working_grid_points=working_grid,
        basis_order=0,
        n_grid_points=120,
        compute_grad=True,
    )
    assert np.isfinite(loglik)
    assert grad is not None
    assert grad.shape == (1,)


def test_right_censored_observed_loglik_gradient_matches_finite_differences() -> None:
    data = _make_data(n=36, seed=5)
    working_grid = np.array([0.2, 0.5, 0.8], dtype=float)
    theta = np.array([0.15, -0.04, 0.03, 0.08], dtype=float)

    loglik, grad = right_censored_observed_loglik_and_gradient(
        theta,
        data,
        working_grid_points=working_grid,
        basis_order=0,
        n_grid_points=120,
        compute_grad=True,
    )
    assert np.isfinite(loglik)
    assert grad is not None

    eps = 1e-6
    fd = np.zeros_like(theta)
    for j in range(theta.size):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] += eps
        theta_minus[j] -= eps
        ll_plus, _ = right_censored_observed_loglik_and_gradient(
            theta_plus,
            data,
            working_grid_points=working_grid,
            basis_order=0,
            n_grid_points=120,
            compute_grad=False,
        )
        ll_minus, _ = right_censored_observed_loglik_and_gradient(
            theta_minus,
            data,
            working_grid_points=working_grid,
            basis_order=0,
            n_grid_points=120,
            compute_grad=False,
        )
        fd[j] = (ll_plus - ll_minus) / (2.0 * eps)

    assert np.allclose(grad, fd, atol=1e-4, rtol=1e-4)


def test_right_censored_observed_l1_warm_start_is_accepted_and_validated() -> None:
    data = _make_data(seed=17)
    stage1 = _fit_stage1_like_estimator(data, norm_constraint=10.0)
    working_grid = np.asarray(stage1.grid_points_hal_selected, dtype=float).copy()
    warm_start = _compress_theta_to_selected_support(stage1)

    est = RightCensoredObservedL1MLE(
        working_grid_points=working_grid,
        norm_constraint=float(stage1.norm_constraint),
        basis_order=int(stage1.basis_order),
        n_grid_points=80,
        learning_rate=0.1,
        n_iterations=120,
        ll_change_tol=1e-4,
        include_intercept_in_constraint=False,
        warm_start_theta=warm_start,
    ).fit(data)
    assert est.theta_hat.shape[0] == warm_start.shape[0]

    with pytest.raises(ValueError, match="warm_start_theta must have length"):
        RightCensoredObservedL1MLE(
            working_grid_points=working_grid,
            norm_constraint=float(stage1.norm_constraint),
            basis_order=int(stage1.basis_order),
            n_grid_points=80,
            learning_rate=0.1,
            n_iterations=120,
            ll_change_tol=1e-4,
            include_intercept_in_constraint=False,
            warm_start_theta=np.zeros(warm_start.size + 1, dtype=float),
        )

    est_no_warm = RightCensoredObservedL1MLE(
        working_grid_points=working_grid,
        norm_constraint=float(stage1.norm_constraint),
        basis_order=int(stage1.basis_order),
        n_grid_points=80,
        learning_rate=0.1,
        n_iterations=120,
        ll_change_tol=1e-4,
        include_intercept_in_constraint=False,
    )
    with pytest.raises(ValueError, match="warm_start_theta must have length"):
        est_no_warm.fit(data, warm_start_theta=np.zeros(warm_start.size + 2, dtype=float))
