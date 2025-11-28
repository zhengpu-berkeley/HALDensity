"""Tests for right-censoring estimators."""

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring import (
    KaplanMeier,
    compute_ipcw_weights,
    WeightedCVXPYEstimator,
    EMIPCWEstimator,
    RightCensoredIPCWEstimator,
    RightCensoredEMEstimator,
    incomplete_loglik,
    kl_divergence,
)


class TestRightCensoredIPCWEstimator:
    """Tests for IPCW estimator (WeightedCVXPYEstimator)."""

    def test_fit_returns_self(self, km_fitted):
        """fit() returns self for chaining."""
        km, data = km_fitted
        weights = compute_ipcw_weights(
            data["T"].values, data["Delta"].values, km.predict
        )
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data["T"].values[uncensored_mask]})

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        result = est.fit(df_unc, sample_weights=weights[uncensored_mask])
        assert result is est

    def test_is_fitted_after_fit(self, km_fitted):
        """is_fitted should be True after fit()."""
        km, data = km_fitted
        weights = compute_ipcw_weights(
            data["T"].values, data["Delta"].values, km.predict
        )
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data["T"].values[uncensored_mask]})

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        est.fit(df_unc, sample_weights=weights[uncensored_mask])
        assert est.is_fitted

    def test_density_nonnegative(self, km_fitted):
        """Density should be non-negative everywhere."""
        km, data = km_fitted
        weights = compute_ipcw_weights(
            data["T"].values, data["Delta"].values, km.predict
        )
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data["T"].values[uncensored_mask]})

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        est.fit(df_unc, sample_weights=weights[uncensored_mask])

        grid, density = est.get_density()
        assert np.all(density >= 0)

    def test_density_integrates_to_one(self, km_fitted):
        """Density should integrate to approximately 1."""
        km, data = km_fitted
        weights = compute_ipcw_weights(
            data["T"].values, data["Delta"].values, km.predict
        )
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data["T"].values[uncensored_mask]})

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        est.fit(df_unc, sample_weights=weights[uncensored_mask])

        grid, density = est.get_density()
        delta = np.diff(np.linspace(0, 1, len(density) + 1))
        integral = np.sum(density * delta)
        assert abs(integral - 1.0) < 0.05  # Within 5%

    def test_get_density_at_points(self, km_fitted):
        """get_density_at_points should return valid densities."""
        km, data = km_fitted
        weights = compute_ipcw_weights(
            data["T"].values, data["Delta"].values, km.predict
        )
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data["T"].values[uncensored_mask]})

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        est.fit(df_unc, sample_weights=weights[uncensored_mask])

        points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        densities = est.get_density_at_points(points)
        assert densities.shape == points.shape
        assert np.all(densities >= 0)

    def test_get_results(self, km_fitted):
        """get_results() should return required fields."""
        km, data = km_fitted
        weights = compute_ipcw_weights(
            data["T"].values, data["Delta"].values, km.predict
        )
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data["T"].values[uncensored_mask]})

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        est.fit(df_unc, sample_weights=weights[uncensored_mask])

        results = est.get_results()
        assert "theta_hat" in results
        assert "estimated_density" in results
        assert "grid_points" in results
        assert "n_selected_knots" in results

    def test_alias_works(self):
        """WeightedCVXPYEstimator should be alias for RightCensoredIPCWEstimator."""
        assert WeightedCVXPYEstimator is RightCensoredIPCWEstimator


class TestRightCensoredEMEstimator:
    """Tests for EM estimator (EMIPCWEstimator)."""

    def test_fit_returns_self(self, truncated_normal_data):
        """fit() returns self for chaining."""
        data, _ = truncated_normal_data
        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )
        result = est.fit(data)
        assert result is est

    def test_is_fitted_after_fit(self, truncated_normal_data):
        """is_fitted should be True after fit()."""
        data, _ = truncated_normal_data
        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )
        est.fit(data)
        assert est.is_fitted

    def test_em_iterations_tracked(self, truncated_normal_data):
        """EM iterations should be tracked."""
        data, _ = truncated_normal_data
        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=3,
        )
        est.fit(data)
        assert est.em_iterations_ >= 1
        assert est.em_iterations_ <= 3

    def test_theta_path_stored(self, truncated_normal_data):
        """theta_path should be stored."""
        data, _ = truncated_normal_data
        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )
        est.fit(data)
        assert len(est.theta_path_) >= 1

    def test_density_nonnegative(self, truncated_normal_data):
        """Density should be non-negative everywhere."""
        data, _ = truncated_normal_data
        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )
        est.fit(data)

        grid, density = est.get_density()
        assert np.all(density >= 0)

    def test_density_integrates_to_one(self, truncated_normal_data):
        """Density should integrate to approximately 1."""
        data, _ = truncated_normal_data
        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )
        est.fit(data)

        grid, density = est.get_density()
        delta = np.diff(np.linspace(0, 1, len(density) + 1))
        integral = np.sum(density * delta)
        assert abs(integral - 1.0) < 0.05  # Within 5%

    def test_get_results_has_em_fields(self, truncated_normal_data):
        """get_results() should include EM-specific fields."""
        data, _ = truncated_normal_data
        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )
        est.fit(data)

        results = est.get_results()
        assert "em_iterations" in results
        assert "em_converged" in results
        assert "theta_path" in results

    def test_alias_works(self):
        """EMIPCWEstimator should be alias for RightCensoredEMEstimator."""
        assert EMIPCWEstimator is RightCensoredEMEstimator

    def test_raises_without_required_columns(self):
        """Should raise ValueError if T or Delta columns missing."""
        data = pd.DataFrame({"X": [0.1, 0.2, 0.3]})
        est = RightCensoredEMEstimator(norm_constraint=50.0)
        with pytest.raises(ValueError, match="T.*Delta"):
            est.fit(data)

