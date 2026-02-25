import numpy as np
import pandas as pd
import pytest

from haldensity.censoring import (
    IntervalCensoredCVOversmoothEMTuner,
    IntervalCensoredEMTuner,
    IntervalCensoredInitEstimator,
    KaplanMeier,
    RightCensoredCVOversmoothEMTuner,
    RightCensoredEMTuner,
    RightCensoredInitEstimator,
    compute_ipcw_weights,
    interval_censor_inspection_uniform,
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


def _make_right_censored_data(n: int = 140, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    sampler = _make_sampler()
    t_event = sampler.generate_samples(n)
    t_cens = np.random.uniform(0.0, 1.0, size=n)
    t_obs = np.minimum(t_event, t_cens)
    delta = (t_event <= t_cens).astype(int)
    return pd.DataFrame({"T": t_obs, "Delta": delta})


def _make_interval_censored_data(n: int = 180, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    sampler = _make_sampler()
    t_event = sampler.generate_samples(n)
    return interval_censor_inspection_uniform(t_event, n_inspections=8, random_state=seed)


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


def test_stage2_tuners_no_longer_accept_cv_folds() -> None:
    right_data = _make_right_censored_data()
    right_stage1 = _fit_stage1_right(right_data)
    with pytest.raises(TypeError):
        RightCensoredEMTuner(
            right_data,
            stage1_estimator=right_stage1,
            do_over_smooth=False,
            cv_folds=3,
        )

    interval_data = _make_interval_censored_data()
    interval_stage1 = _fit_stage1_interval(interval_data)
    with pytest.raises(TypeError):
        IntervalCensoredEMTuner(
            interval_data,
            stage1_estimator=interval_stage1,
            do_over_smooth=False,
            cv_folds=3,
        )


def test_no_oversmooth_mode_returns_single_factor_and_standard_params() -> None:
    data = _make_right_censored_data()
    stage1 = _fit_stage1_right(data)

    result = RightCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=False,
        em_m_imputations=20,
        em_max_em_iter=8,
        em_tol=1e-3,
        em_norm_factor=1.0,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    assert result.metadata["mode"] == "no_oversmooth"
    assert set(result.best_params.keys()) == {"oversmooth_factor", "em_norm_factor"}
    assert result.best_params["oversmooth_factor"] == pytest.approx(1.0)
    assert result.best_params["em_norm_factor"] == pytest.approx(1.0)
    assert len(result.metadata["em_records"]) == 1
    assert result.metadata["em_records"][0].factor == pytest.approx(1.0)


def test_no_oversmooth_warns_for_non_unit_em_norm_factor() -> None:
    data = _make_right_censored_data()
    stage1 = _fit_stage1_right(data)

    with pytest.warns(UserWarning, match="em_norm_factor"):
        RightCensoredEMTuner(
            data,
            stage1_estimator=stage1,
            do_over_smooth=False,
            em_m_imputations=20,
            em_max_em_iter=8,
            em_tol=1e-3,
            em_norm_factor=1.3,
            n_grid_points=100,
            random_state=42,
            silent=True,
        ).optimize()


def test_oversmooth_reuses_stage1_estimator_for_factor_one() -> None:
    data = _make_right_censored_data()
    stage1 = _fit_stage1_right(data)

    result = RightCensoredEMTuner(
        data,
        stage1_estimator=stage1,
        do_over_smooth=True,
        oversmooth_factors=[0.7, 1.0],
        em_m_imputations=20,
        em_max_em_iter=8,
        em_tol=1e-3,
        em_norm_factor=1.5,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize()

    init_by_factor = {float(rec.factor): rec for rec in result.metadata["init_records"]}
    assert init_by_factor[1.0].estimator is stage1


def test_right_cv_oversmooth_returns_valid_result_and_metadata() -> None:
    data = _make_right_censored_data()
    stage1 = _fit_stage1_right(data)

    result = RightCensoredCVOversmoothEMTuner(
        data,
        stage1_estimator=stage1,
        cv_folds=2,
        oversmooth_factors=[0.5, 1.0],
        em_norm_factors=[1.0, 2.0],
        em_m_imputations=15,
        em_max_em_iter=5,
        em_tol=1e-2,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize(n_trials=4)

    assert result.metadata["mode"] == "cv_oversmooth"
    assert set(result.best_params.keys()) == {"oversmooth_factor", "em_norm_factor"}
    assert 0.5 <= result.best_params["oversmooth_factor"] <= 1.0
    assert 1.0 <= result.best_params["em_norm_factor"] <= 2.0

    assert len(result.metadata["cv_records"]) == 4
    assert result.metadata["n_trials"] == 4
    for rec in result.metadata["cv_records"]:
        assert len(rec.fold_scores) == 2
        assert isinstance(rec.mean_cv_ll, float)
        assert isinstance(rec.sd_cv_ll, float)

    assert result.metadata["best_record"] is not None
    assert result.estimator is not None


def test_right_cv_oversmooth_estimator_produces_valid_density() -> None:
    data = _make_right_censored_data()
    stage1 = _fit_stage1_right(data)

    result = RightCensoredCVOversmoothEMTuner(
        data,
        stage1_estimator=stage1,
        cv_folds=2,
        oversmooth_factors=[0.5, 1.0],
        em_norm_factors=[1.0, 2.0],
        em_m_imputations=15,
        em_max_em_iter=5,
        em_tol=1e-2,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize(n_trials=4)

    grid = np.linspace(0.001, 0.999, 500)
    density = np.asarray(result.estimator.get_density_at_points(grid), dtype=float)
    assert float(np.min(density)) >= -1e-12
    area = float(np.trapz(density, grid))
    assert 0.85 <= area <= 1.15


def test_cv_oversmooth_returns_valid_result_and_metadata() -> None:
    data = _make_interval_censored_data()
    stage1 = _fit_stage1_interval(data)

    result = IntervalCensoredCVOversmoothEMTuner(
        data,
        stage1_estimator=stage1,
        cv_folds=2,
        oversmooth_factors=[0.5, 1.0],
        em_norm_factors=[1.0, 2.0],
        em_m_imputations=15,
        em_max_em_iter=5,
        em_tol=1e-2,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize(n_trials=4)

    assert result.metadata["mode"] == "cv_oversmooth"
    assert set(result.best_params.keys()) == {"oversmooth_factor", "em_norm_factor"}
    assert 0.5 <= result.best_params["oversmooth_factor"] <= 1.0
    assert 1.0 <= result.best_params["em_norm_factor"] <= 2.0

    # One record per Optuna trial
    assert len(result.metadata["cv_records"]) == 4
    assert result.metadata["n_trials"] == 4
    for rec in result.metadata["cv_records"]:
        assert len(rec.fold_scores) == 2  # cv_folds=2
        assert isinstance(rec.mean_cv_ll, float)
        assert isinstance(rec.sd_cv_ll, float)

    assert result.metadata["best_record"] is not None
    assert result.estimator is not None


def test_cv_oversmooth_estimator_produces_valid_density() -> None:
    data = _make_interval_censored_data()
    stage1 = _fit_stage1_interval(data)

    result = IntervalCensoredCVOversmoothEMTuner(
        data,
        stage1_estimator=stage1,
        cv_folds=2,
        oversmooth_factors=[0.5, 1.0],
        em_norm_factors=[1.0, 2.0],
        em_m_imputations=15,
        em_max_em_iter=5,
        em_tol=1e-2,
        n_grid_points=100,
        random_state=42,
        silent=True,
    ).optimize(n_trials=4)

    grid = np.linspace(0.001, 0.999, 500)
    density = np.asarray(result.estimator.get_density_at_points(grid), dtype=float)
    assert float(np.min(density)) >= -1e-12
    area = float(np.trapz(density, grid))
    assert 0.85 <= area <= 1.15

