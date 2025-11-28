"""Tests for EMStage standalone functionality."""

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring import (
    KaplanMeier,
    compute_ipcw_weights,
    RightCensoredIPCWEstimator,
    RightCensoredEMEstimator,
    EMStage,
    EMStageResult,
    incomplete_loglik,
)


class TestEMStage:
    """Tests for EMStage class."""

    @pytest.fixture
    def initial_estimator(self, km_fitted):
        """Fitted initial IPCW estimator for EMStage tests."""
        km, data = km_fitted
        T_vals = data["T"].values
        Delta_vals = data["Delta"].values
        weights = compute_ipcw_weights(T_vals, Delta_vals, km.predict)

        uncensored_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        w_unc = weights[uncensored_mask]

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        est.fit(df_unc, sample_weights=w_unc)
        return est, km, data

    def test_em_stage_returns_result(self, initial_estimator):
        """EMStage.run() returns EMStageResult."""
        est, km, data = initial_estimator

        em_stage = EMStage(
            m_imputations=5,
            max_em_iter=2,
            norm_constraint=50.0,
        )

        result = em_stage.run(
            initial_estimator=est,
            data=data,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        assert isinstance(result, EMStageResult)

    def test_em_stage_result_has_estimator(self, initial_estimator):
        """EMStageResult should have final_estimator."""
        est, km, data = initial_estimator

        em_stage = EMStage(
            m_imputations=5,
            max_em_iter=2,
            norm_constraint=50.0,
        )

        result = em_stage.run(
            initial_estimator=est,
            data=data,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        assert result.final_estimator is not None
        assert hasattr(result.final_estimator, "get_density")

    def test_em_stage_result_has_theta_path(self, initial_estimator):
        """EMStageResult should have theta_path."""
        est, km, data = initial_estimator

        em_stage = EMStage(
            m_imputations=5,
            max_em_iter=3,
            norm_constraint=50.0,
        )

        result = em_stage.run(
            initial_estimator=est,
            data=data,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        assert len(result.theta_path) >= 1

    def test_em_stage_iterations_bounded(self, initial_estimator):
        """EM iterations should not exceed max_em_iter."""
        est, km, data = initial_estimator
        max_iter = 3

        em_stage = EMStage(
            m_imputations=5,
            max_em_iter=max_iter,
            norm_constraint=50.0,
        )

        result = em_stage.run(
            initial_estimator=est,
            data=data,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        assert result.em_iterations <= max_iter

    def test_em_stage_final_density_valid(self, initial_estimator):
        """Final estimator should produce valid density."""
        est, km, data = initial_estimator

        em_stage = EMStage(
            m_imputations=5,
            max_em_iter=2,
            norm_constraint=50.0,
        )

        result = em_stage.run(
            initial_estimator=est,
            data=data,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        grid, density = result.final_estimator.get_density()
        assert np.all(density >= 0)
        assert np.all(np.isfinite(density))

    def test_em_stage_augmented_data_available(self, initial_estimator):
        """final_augmented_data should be available."""
        est, km, data = initial_estimator

        em_stage = EMStage(
            m_imputations=5,
            max_em_iter=2,
            norm_constraint=50.0,
        )

        result = em_stage.run(
            initial_estimator=est,
            data=data,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        assert result.final_augmented_data is not None
        assert "W1" in result.final_augmented_data.columns
        assert "weight" in result.final_augmented_data.columns


class TestEMStageConsistency:
    """Tests for consistency between EMStage and RightCensoredEMEstimator."""

    def test_both_approaches_produce_valid_results(
        self, truncated_normal_data
    ):
        """Both direct and manual approaches should produce valid results."""
        data, _ = truncated_normal_data

        # Approach 1: Direct EMIPCWEstimator
        em_estimator = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=10,
            max_em_iter=3,
            rng_seed=99,
        )
        em_estimator.fit(data)

        grid1, density1 = em_estimator.get_density()
        ll1 = incomplete_loglik(em_estimator, data)

        # Approach 2: Manual IPCW + EMStage
        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
        T_vals = data["T"].values
        Delta_vals = data["Delta"].values
        weights = compute_ipcw_weights(T_vals, Delta_vals, km.predict)

        uncensored_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[uncensored_mask]})

        initial_est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        initial_est.fit(df_unc, sample_weights=weights[uncensored_mask])

        em_stage = EMStage(
            m_imputations=10,
            max_em_iter=3,
            norm_constraint=50.0,
            rng_seed=99,
        )

        result = em_stage.run(
            initial_estimator=initial_est,
            data=data,
            S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
        )

        grid2, density2 = result.final_estimator.get_density()
        ll2 = incomplete_loglik(result.final_estimator, data)

        # Both should produce valid (finite) log-likelihoods
        assert np.isfinite(ll1)
        assert np.isfinite(ll2)

        # Both should produce non-negative densities
        assert np.all(density1 >= 0)
        assert np.all(density2 >= 0)

