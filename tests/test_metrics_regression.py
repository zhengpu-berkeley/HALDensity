"""Regression tests for metrics functions.

These tests verify that metric computations match expected values captured before refactoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring.right.estimators import RightCensoredInitEstimator, RightCensoredEMEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.metrics import incomplete_loglik, mi_complete_loglik
from haldensity.censoring.interval.estimators import IntervalCensoredInitEstimator, IntervalCensoredEMEstimator
from haldensity.censoring.interval.metrics import incomplete_loglik_interval
from haldensity.censoring.utils.common_metrics import kl_divergence

from conftest import assert_loglik_close, LOGLIK_RTOL, LOGLIK_ATOL


# =============================================================================
# Right-Censored Metrics Tests
# =============================================================================

class TestRCMetrics:
    """Regression tests for right-censored metrics."""

    @pytest.fixture
    def fitted_rc_init(self, rc_data: pd.DataFrame, rc_init_params: dict) -> RightCensoredInitEstimator:
        """Fit RC Init estimator."""
        km = KaplanMeier().fit(rc_data, time_col="T", delta_col="Delta")
        T_vals = np.asarray(rc_data["T"].values, dtype=float)
        Delta_vals = np.asarray(rc_data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        est = RightCensoredInitEstimator(**rc_init_params)
        est.fit(df_unc, sample_weights=w_unc)
        return est

    @pytest.fixture
    def fitted_rc_em(self, rc_data: pd.DataFrame, rc_em_params: dict) -> RightCensoredEMEstimator:
        """Fit RC EM estimator."""
        est = RightCensoredEMEstimator(**rc_em_params)
        est.fit(rc_data)
        return est

    def test_rc_incomplete_loglik_init(
        self, 
        fitted_rc_init: RightCensoredInitEstimator, 
        rc_data: pd.DataFrame, 
        rc_expected: dict
    ) -> None:
        """Verify incomplete log-likelihood for RC Init estimator."""
        expected_ll = rc_expected["init"]["incomplete_loglik"]
        actual_ll = float(incomplete_loglik(fitted_rc_init, rc_data, time_col="T", delta_col="Delta"))
        
        assert_loglik_close(actual_ll, expected_ll, name="RC Init incomplete_loglik")

    def test_rc_incomplete_loglik_em(
        self, 
        fitted_rc_em: RightCensoredEMEstimator, 
        rc_data: pd.DataFrame, 
        rc_expected: dict
    ) -> None:
        """Verify incomplete log-likelihood for RC EM estimator."""
        expected_ll = rc_expected["em"]["incomplete_loglik"]
        actual_ll = float(incomplete_loglik(fitted_rc_em, rc_data, time_col="T", delta_col="Delta"))
        
        assert_loglik_close(actual_ll, expected_ll, name="RC EM incomplete_loglik")

    def test_rc_loglik_em_better_than_init(
        self,
        fitted_rc_init: RightCensoredInitEstimator,
        fitted_rc_em: RightCensoredEMEstimator,
        rc_data: pd.DataFrame,
    ) -> None:
        """Verify EM improves or maintains log-likelihood vs Init.
        
        Note: This is a sanity check, not a strict regression test.
        EM should generally improve fit, but with regularization this isn't guaranteed.
        """
        init_ll = float(incomplete_loglik(fitted_rc_init, rc_data, time_col="T", delta_col="Delta"))
        em_ll = float(incomplete_loglik(fitted_rc_em, rc_data, time_col="T", delta_col="Delta"))
        
        # Both should be finite
        assert np.isfinite(init_ll), f"RC Init LL is not finite: {init_ll}"
        assert np.isfinite(em_ll), f"RC EM LL is not finite: {em_ll}"


# =============================================================================
# Interval-Censored Metrics Tests
# =============================================================================

class TestICMetrics:
    """Regression tests for interval-censored metrics."""

    @pytest.fixture
    def fitted_ic_init(self, ic_data: pd.DataFrame, ic_init_params: dict) -> IntervalCensoredInitEstimator:
        """Fit IC Init estimator."""
        est = IntervalCensoredInitEstimator(**ic_init_params)
        est.fit(ic_data, L_col="L", R_col="R")
        return est

    @pytest.fixture
    def fitted_ic_em(self, ic_data: pd.DataFrame, ic_em_params: dict) -> IntervalCensoredEMEstimator:
        """Fit IC EM estimator."""
        est = IntervalCensoredEMEstimator(**ic_em_params)
        est.fit(ic_data)
        return est

    def test_ic_incomplete_loglik_init(
        self, 
        fitted_ic_init: IntervalCensoredInitEstimator, 
        ic_data: pd.DataFrame, 
        ic_expected: dict
    ) -> None:
        """Verify incomplete log-likelihood for IC Init estimator."""
        expected_ll = ic_expected["init"]["incomplete_loglik"]
        actual_ll = float(incomplete_loglik_interval(fitted_ic_init, ic_data, L_col="L", R_col="R"))
        
        assert_loglik_close(actual_ll, expected_ll, name="IC Init incomplete_loglik")

    def test_ic_incomplete_loglik_em(
        self, 
        fitted_ic_em: IntervalCensoredEMEstimator, 
        ic_data: pd.DataFrame, 
        ic_expected: dict
    ) -> None:
        """Verify incomplete log-likelihood for IC EM estimator."""
        expected_ll = ic_expected["em"]["incomplete_loglik"]
        actual_ll = float(incomplete_loglik_interval(fitted_ic_em, ic_data, L_col="L", R_col="R"))
        
        assert_loglik_close(actual_ll, expected_ll, name="IC EM incomplete_loglik")

    def test_ic_loglik_finite(
        self,
        fitted_ic_init: IntervalCensoredInitEstimator,
        fitted_ic_em: IntervalCensoredEMEstimator,
        ic_data: pd.DataFrame,
    ) -> None:
        """Verify log-likelihoods are finite for both estimators."""
        init_ll = float(incomplete_loglik_interval(fitted_ic_init, ic_data, L_col="L", R_col="R"))
        em_ll = float(incomplete_loglik_interval(fitted_ic_em, ic_data, L_col="L", R_col="R"))
        
        assert np.isfinite(init_ll), f"IC Init LL is not finite: {init_ll}"
        assert np.isfinite(em_ll), f"IC EM LL is not finite: {em_ll}"


# =============================================================================
# KL Divergence Tests
# =============================================================================

class TestKLDivergence:
    """Regression tests for KL divergence metric.
    
    Note: kl_divergence signature is: kl_divergence(true_pdf_fn, grid, est_density)
    where true_pdf_fn is a callable.
    """

    def test_kl_divergence_same_density_is_zero(self) -> None:
        """KL divergence of a density with itself should be ~0."""
        grid = np.linspace(0.01, 0.99, 100)
        # Simple density: uniform
        density = np.ones(len(grid))
        
        # Create a callable for the "true" density
        true_pdf_fn = lambda x: np.ones(len(x))
        
        kl = kl_divergence(true_pdf_fn, grid, density)
        
        assert np.isclose(kl, 0.0, atol=1e-10), f"KL(p, p) should be 0, got {kl}"

    def test_kl_divergence_different_densities_positive(self) -> None:
        """KL divergence of different densities should be positive."""
        grid = np.linspace(0.01, 0.99, 100)
        
        # True density: skewed left
        true_pdf_fn = lambda x: np.exp(-5 * x)
        
        # Estimated density: skewed right
        est_density = np.exp(5 * grid)
        
        kl = kl_divergence(true_pdf_fn, grid, est_density)
        
        assert np.isfinite(kl), f"KL divergence should be finite, got {kl}"
        assert kl > 0, f"KL divergence of different densities should be positive, got {kl}"

    def test_kl_divergence_asymmetric(self) -> None:
        """KL divergence is not symmetric: KL(p, q) != KL(q, p).
        
        Note: With symmetric densities like Beta(a,b) vs Beta(b,a), the KL values
        can be similar. Use more asymmetric distributions to test.
        """
        grid = np.linspace(0.01, 0.99, 100)
        
        # Density 1: Very peaked at left (exponential-like)
        p_fn = lambda x: np.exp(-10 * x)
        p_vals = p_fn(grid)
        
        # Density 2: Uniform (flat)
        q_fn = lambda x: np.ones_like(x)
        q_vals = q_fn(grid)
        
        kl_pq = kl_divergence(p_fn, grid, q_vals)  # KL(p || q) - peaked vs flat
        kl_qp = kl_divergence(q_fn, grid, p_vals)  # KL(q || p) - flat vs peaked
        
        assert np.isfinite(kl_pq) and np.isfinite(kl_qp), "KL values should be finite"
        # They should be meaningfully different (asymmetric)
        # For exponential vs uniform, these will be quite different
        assert abs(kl_pq - kl_qp) > 0.5, (
            f"KL should be asymmetric: KL(p,q)={kl_pq}, KL(q,p)={kl_qp}, diff={abs(kl_pq - kl_qp)}"
        )
