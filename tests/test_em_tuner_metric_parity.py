import numpy as np
import pandas as pd
import pytest

from haldensity.censoring import (
    IntervalCensoredEMTuner,
    IntervalCensoredInitEstimator,
    KaplanMeier,
    RightCensoredEMTuner,
    RightCensoredInitEstimator,
    compute_ipcw_weights,
    incomplete_loglik_interval,
    interval_censor_inspection_uniform,
    kl_divergence,
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


def _make_right_censored_data(n: int = 220, seed: int = 42) -> tuple[pd.DataFrame, TruncatedGMM]:
    np.random.seed(seed)
    sampler = _make_sampler()
    t_event = sampler.generate_samples(n)
    t_cens = np.random.uniform(0.0, 1.0, size=n)
    t_obs = np.minimum(t_event, t_cens)
    delta = (t_event <= t_cens).astype(int)
    return pd.DataFrame({"T": t_obs, "Delta": delta}), sampler


def _make_interval_censored_data(n: int = 260, seed: int = 42) -> tuple[pd.DataFrame, TruncatedGMM]:
    np.random.seed(seed)
    sampler = _make_sampler()
    t_event = sampler.generate_samples(n)
    data = interval_censor_inspection_uniform(t_event, n_inspections=8, random_state=seed)
    return data, sampler


def _fit_stage1_right(data: pd.DataFrame) -> RightCensoredInitEstimator:
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    times = np.asarray(data["T"].values, dtype=float)
    delta = np.asarray(data["Delta"].values, dtype=int)
    weights = compute_ipcw_weights(times, delta, lambda t: np.atleast_1d(km.predict(t)))
    unc = delta == 1

    est = RightCensoredInitEstimator(
        norm_constraint=10.0,
        n_grid_points=100,
        basis_order=0,
        solver="ECOS",
        use_secondary_solver=True,
    )
    est.fit(pd.DataFrame({"W1": times[unc]}), sample_weights=weights[unc])
    return est


def _fit_stage1_interval(data: pd.DataFrame) -> IntervalCensoredInitEstimator:
    return IntervalCensoredInitEstimator(
        norm_constraint=10.0,
        n_grid_points=100,
        basis_order=0,
        solver="ECOS",
        use_secondary_solver=True,
        include_intercept_in_constraint=False,
    ).fit(data, L_col="L", R_col="R")


def _kl_to_truth(
    sampler: TruncatedGMM,
    estimator,
    grid: np.ndarray,
) -> float:
    est_density = np.asarray(estimator.get_density_at_points(grid), dtype=float)
    return kl_divergence(sampler.compute_density, grid, est_density)


def _selected_knots(estimator) -> np.ndarray:
    knots = estimator.get_results()["grid_points_hal_selected"]
    return np.asarray(knots, dtype=float)


def _assert_density_validity(estimator, label: str) -> None:
    grid = np.linspace(0.001, 0.999, 5000)
    density = np.asarray(estimator.get_density_at_points(grid), dtype=float)
    assert float(np.min(density)) >= -1e-12, f"{label}: negative density encountered"

    # External numerical integration on a fine grid.
    area_trapz = float(np.trapz(density, grid))
    assert 0.9 <= area_trapz <= 1.1, f"{label}: trapz area out of range ({area_trapz})"

    # Internal normalization consistency on the estimator's own integration grid.
    grid_mid = getattr(estimator, "grid_midpoints", None)
    delta_j = getattr(estimator, "delta_j", None)
    assert grid_mid is not None and delta_j is not None, f"{label}: missing normalization grid state"
    grid_mid = np.asarray(grid_mid, dtype=float)
    delta_j = np.asarray(delta_j, dtype=float)
    density_mid = np.asarray(estimator.get_density_at_points(grid_mid), dtype=float)
    area_internal = float(np.sum(density_mid * delta_j))
    assert abs(area_internal - 1.0) <= 1e-8, f"{label}: internal area not normalized ({area_internal})"

    # External vs internal integration should be broadly consistent.
    assert abs(area_trapz - area_internal) <= 0.12, (
        f"{label}: trapz/internal area mismatch ({area_trapz} vs {area_internal})"
    )


def test_right_censored_stage2b_improves_kl_and_preserves_knot_invariants() -> None:
    data, sampler = _make_right_censored_data()
    stage1 = _fit_stage1_right(data)

    stage2a = RightCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=False,
        em_m_imputations=30,
        em_max_em_iter=12,
        em_tol=1e-3,
        em_norm_factor=1.0,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    stage2b = RightCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=True,
        oversmooth_factors=[0.6, 0.8, 1.0],
        em_m_imputations=30,
        em_max_em_iter=12,
        em_tol=1e-3,
        em_norm_factor=1.5,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    grid = np.linspace(0.001, 0.999, 2500)
    kl_stage1 = _kl_to_truth(sampler, stage1, grid)
    kl_stage2a = _kl_to_truth(sampler, stage2a.estimator, grid)
    kl_stage2b = _kl_to_truth(sampler, stage2b.estimator, grid)

    assert kl_stage2b <= kl_stage1 + 1e-8
    assert kl_stage2b <= kl_stage2a + 1e-8

    n_knots_stage1 = int(stage1.get_results()["n_selected_knots"])
    n_knots_stage2a = int(stage2a.estimator.get_results()["n_selected_knots"])
    n_knots_stage2b = int(stage2b.estimator.get_results()["n_selected_knots"])
    assert n_knots_stage2a == n_knots_stage1
    assert n_knots_stage2b <= n_knots_stage1

    # Run-level knot invariance: Stage 2A must preserve Stage 1 knot set exactly.
    assert np.array_equal(_selected_knots(stage1), _selected_knots(stage2a.estimator))

    # Factor-level knot invariance: each Stage 2B EM run must preserve its own init knot set.
    init_by_factor = {float(rec.factor): rec for rec in stage2b.metadata["init_records"]}
    for em_rec in stage2b.metadata["em_records"]:
        init_knots = _selected_knots(init_by_factor[float(em_rec.factor)].estimator)
        em_knots = _selected_knots(em_rec.em_estimator)
        assert np.array_equal(init_knots, em_knots)

    _assert_density_validity(stage1, "right/stage1")
    _assert_density_validity(stage2a.estimator, "right/stage2a")
    _assert_density_validity(stage2b.estimator, "right/stage2b")


def test_interval_censored_stage2b_improves_kl_and_preserves_knot_invariants() -> None:
    data, sampler = _make_interval_censored_data()
    stage1 = _fit_stage1_interval(data)

    stage2a = IntervalCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=False,
        em_m_imputations=30,
        em_max_em_iter=12,
        em_tol=1e-3,
        em_norm_factor=1.0,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    stage2b = IntervalCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=True,
        oversmooth_factors=[0.6, 0.8, 1.0],
        em_m_imputations=30,
        em_max_em_iter=12,
        em_tol=1e-3,
        em_norm_factor=1.6,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    grid = np.linspace(0.001, 0.999, 2500)
    kl_stage1 = _kl_to_truth(sampler, stage1, grid)
    kl_stage2a = _kl_to_truth(sampler, stage2a.estimator, grid)
    kl_stage2b = _kl_to_truth(sampler, stage2b.estimator, grid)

    assert kl_stage2b <= kl_stage1 + 1e-8
    # For IC, KL to truth can favor Stage 2A while Stage 2B is selected by
    # incomplete-data likelihood. Require Stage 2B to dominate on its
    # optimization target and remain KL-improved vs Stage 1.
    ll_stage2a = float(incomplete_loglik_interval(stage2a.estimator, data))
    ll_stage2b = float(incomplete_loglik_interval(stage2b.estimator, data))
    assert ll_stage2b >= ll_stage2a - 1e-8

    n_knots_stage1 = int(stage1.get_results()["n_selected_knots"])
    n_knots_stage2a = int(stage2a.estimator.get_results()["n_selected_knots"])
    n_knots_stage2b = int(stage2b.estimator.get_results()["n_selected_knots"])
    assert n_knots_stage2a == n_knots_stage1
    assert n_knots_stage2b <= n_knots_stage1

    # Run-level knot invariance: Stage 2A must preserve Stage 1 knot set exactly.
    assert np.array_equal(_selected_knots(stage1), _selected_knots(stage2a.estimator))

    # Factor-level knot invariance: each Stage 2B EM run must preserve its own init knot set.
    init_by_factor = {float(rec.factor): rec for rec in stage2b.metadata["init_records"]}
    for em_rec in stage2b.metadata["em_records"]:
        init_knots = _selected_knots(init_by_factor[float(em_rec.factor)].estimator)
        em_knots = _selected_knots(em_rec.em_estimator)
        assert np.array_equal(init_knots, em_knots)

    _assert_density_validity(stage1, "interval/stage1")
    _assert_density_validity(stage2a.estimator, "interval/stage2a")
    _assert_density_validity(stage2b.estimator, "interval/stage2b")


@pytest.mark.slow_notebook_parity
def test_notebook_style_right_censored_metric_window() -> None:
    data, sampler = _make_right_censored_data(n=300, seed=42)
    stage1 = _fit_stage1_right(data)

    stage2a = RightCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=False,
        em_m_imputations=50,
        em_max_em_iter=20,
        em_tol=1e-3,
        em_norm_factor=1.0,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    stage2b = RightCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=True,
        oversmooth_factors=[0.6, 0.7, 0.8, 0.9, 1.0],
        em_m_imputations=50,
        em_max_em_iter=20,
        em_tol=1e-3,
        em_norm_factor=1.5,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    grid = np.linspace(0.001, 0.999, 4000)
    kl_stage1 = _kl_to_truth(sampler, stage1, grid)
    kl_stage2a = _kl_to_truth(sampler, stage2a.estimator, grid)
    kl_stage2b = _kl_to_truth(sampler, stage2b.estimator, grid)

    assert kl_stage2b <= kl_stage1 + 5e-3
    assert kl_stage2b <= kl_stage2a + 5e-3
