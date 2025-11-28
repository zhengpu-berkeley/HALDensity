"""Tests for utility functions: KaplanMeier, weights, metrics."""

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring import (
    KaplanMeier,
    compute_ipcw_weights,
    incomplete_loglik,
    kl_divergence,
)


class TestKaplanMeier:
    """Tests for KaplanMeier estimator."""

    def test_fit_returns_self(self, truncated_normal_data):
        """KaplanMeier.fit() returns self for chaining."""
        data, _ = truncated_normal_data
        km = KaplanMeier()
        result = km.fit(data, time_col="T", delta_col="Delta")
        assert result is km

    def test_predict_returns_array(self, km_fitted):
        """KaplanMeier.predict() returns array for array input."""
        km, _ = km_fitted
        times = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = km.predict(times)
        assert isinstance(result, np.ndarray)
        assert result.shape == times.shape

    def test_predict_returns_float_for_scalar(self, km_fitted):
        """KaplanMeier.predict() returns float for scalar input."""
        km, _ = km_fitted
        result = km.predict(0.5)
        assert isinstance(result, float)

    def test_survival_decreases(self, km_fitted):
        """Survival function should be non-increasing."""
        km, _ = km_fitted
        times = np.linspace(0.01, 0.99, 50)
        surv = km.predict(times)
        assert np.all(np.diff(surv) <= 1e-10)  # Non-increasing

    def test_survival_bounded(self, km_fitted):
        """Survival should be in [0, 1]."""
        km, _ = km_fitted
        times = np.linspace(0.0, 1.0, 100)
        surv = km.predict(times)
        assert np.all(surv >= 0.0)
        assert np.all(surv <= 1.0)

    def test_stepwise_survival_returns_copies(self, km_fitted):
        """stepwise_survival_() returns copies of internal arrays."""
        km, _ = km_fitted
        times1, surv1 = km.stepwise_survival_()
        times2, surv2 = km.stepwise_survival_()
        assert times1 is not times2
        assert surv1 is not surv2

    def test_raises_before_fit(self):
        """Should raise ValueError before fit() is called."""
        km = KaplanMeier()
        with pytest.raises(ValueError, match="fit"):
            km.predict(0.5)

    def test_raises_on_empty_data(self):
        """Should raise ValueError on empty data."""
        km = KaplanMeier()
        empty_data = pd.DataFrame({"T": [], "Delta": []})
        with pytest.raises(ValueError, match="Empty"):
            km.fit(empty_data)


class TestIPCWWeights:
    """Tests for compute_ipcw_weights function."""

    def test_weights_shape(self, ipcw_weights):
        """Weights should have same shape as input."""
        weights, data = ipcw_weights
        assert weights.shape == (len(data),)

    def test_censored_weights_zero(self, ipcw_weights):
        """Censored observations should have zero weight."""
        weights, data = ipcw_weights
        censored_mask = data["Delta"].values == 0
        assert np.all(weights[censored_mask] == 0.0)

    def test_uncensored_weights_positive(self, ipcw_weights):
        """Uncensored observations should have positive weight."""
        weights, data = ipcw_weights
        uncensored_mask = data["Delta"].values == 1
        assert np.all(weights[uncensored_mask] > 0.0)

    def test_weights_bounded(self, ipcw_weights):
        """Weights should be bounded (no extreme values)."""
        weights, _ = ipcw_weights
        max_weight = np.max(weights)
        assert max_weight < 1000  # Reasonable upper bound

    def test_raises_on_mismatched_lengths(self):
        """Should raise ValueError on mismatched array lengths."""
        T = np.array([0.1, 0.2, 0.3])
        Delta = np.array([1, 0])  # Wrong length
        with pytest.raises(ValueError, match="same length"):
            compute_ipcw_weights(T, Delta, lambda x: np.ones_like(x))


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_incomplete_loglik_finite(self, km_fitted):
        """incomplete_loglik should return finite value."""
        from haldensity.censoring import RightCensoredIPCWEstimator, compute_ipcw_weights

        km, data = km_fitted
        T_vals = data["T"].values
        Delta_vals = data["Delta"].values
        weights = compute_ipcw_weights(T_vals, Delta_vals, km.predict)

        uncensored_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[uncensored_mask]})
        w_unc = weights[uncensored_mask]

        est = RightCensoredIPCWEstimator(norm_constraint=50.0).fit(
            df_unc, sample_weights=w_unc
        )
        ll = incomplete_loglik(est, data, time_col="T", delta_col="Delta")
        assert np.isfinite(ll)

    def test_kl_divergence_zero_for_identical(self):
        """KL divergence should be near zero for identical distributions."""
        grid = np.linspace(0.1, 0.9, 100)
        true_pdf = lambda x: np.ones_like(x)  # Uniform
        est_density = np.ones_like(grid)

        kl = kl_divergence(true_pdf, grid, est_density)
        assert abs(kl) < 1e-10

    def test_kl_divergence_positive_for_different(self):
        """KL divergence should be positive for different distributions."""
        from scipy.stats import norm

        grid = np.linspace(-3, 3, 200)
        true_pdf = lambda x: norm.pdf(x, loc=0, scale=1)
        est_density = norm.pdf(grid, loc=1, scale=1)  # Shifted

        kl = kl_divergence(true_pdf, grid, est_density)
        assert kl > 0

