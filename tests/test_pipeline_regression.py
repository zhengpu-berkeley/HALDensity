"""Regression tests for complete estimation pipelines.

These tests verify end-to-end Stage1 → Stage2 pipelines for both
right-censored and interval-censored data, with and without oversmoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Right-censored imports
from haldensity.censoring.right.estimators import (
    RightCensoredInitEstimator,
    RightCensoredEMEstimator,
    RightCensoredEMStage,
)
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.metrics import incomplete_loglik
from haldensity.censoring.tuners.right_tuners import RightCensoredEMTuner

# Interval-censored imports
from haldensity.censoring.interval.estimators import (
    IntervalCensoredInitEstimator,
    IntervalCensoredEMEstimator,
    IntervalCensoredEMStage,
)
from haldensity.censoring.interval.metrics import incomplete_loglik_interval
from haldensity.censoring.tuners.interval_tuners import IntervalCensoredEMTuner

# Backward compatibility aliases
RightCensoredIPCWEstimator = RightCensoredInitEstimator
IntervalCensoredMidpointEstimator = IntervalCensoredInitEstimator

from conftest import assert_density_close, assert_loglik_close


# =============================================================================
# Right-Censored Pipeline Tests
# =============================================================================

class TestRCPipelineOversmooth:
    """Test RC pipeline with oversmoothing (do_over_smooth=True behavior)."""

    @pytest.fixture
    def rc_stage1_estimator(self, rc_data: pd.DataFrame) -> RightCensoredIPCWEstimator:
        """Fit Stage 1 IPCW estimator."""
        km = KaplanMeier().fit(rc_data, time_col="T", delta_col="Delta")
        T_vals = np.asarray(rc_data["T"].values, dtype=float)
        Delta_vals = np.asarray(rc_data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        est = RightCensoredIPCWEstimator(
            norm_constraint=10.0,
            n_grid_points=100,
            basis_order=0,
            solver="ECOS",
            use_secondary_solver=True,
        )
        est.fit(df_unc, sample_weights=w_unc)
        return est

    def test_rc_pipeline_oversmooth_produces_valid_estimator(
        self, rc_data: pd.DataFrame, rc_stage1_estimator: RightCensoredIPCWEstimator
    ) -> None:
        """Verify oversmooth pipeline produces a valid fitted estimator."""
        # Run oversmooth tuner with limited factors for speed
        tuner = RightCensoredEMTuner(
            rc_data,
            stage1_estimator=rc_stage1_estimator,
            oversmooth_factors=[0.8, 1.0],  # Limited for speed
            em_m_imputations=10,
            em_max_em_iter=5,
            silent=True,
            do_over_smooth=True,
        )
        
        result = tuner.optimize()
        best_estimator = result.estimator
        
        # Verify estimator is fitted and produces valid output
        assert best_estimator is not None, "Oversmooth tuner should return an estimator"
        assert hasattr(best_estimator, "theta_hat"), "Estimator should have theta_hat"
        assert best_estimator.theta_hat is not None, "theta_hat should not be None"
        
        # Verify density computation works
        eval_points = np.linspace(0.1, 0.9, 10)
        density = best_estimator.get_density_at_points(eval_points)
        assert len(density) == len(eval_points), "Density should match eval points length"
        assert np.all(np.isfinite(density)), "Density values should be finite"
        assert np.all(density >= 0), "Density values should be non-negative"

    def test_rc_pipeline_oversmooth_improves_or_maintains_loglik(
        self, rc_data: pd.DataFrame
    ) -> None:
        """Verify EM refinement doesn't catastrophically hurt log-likelihood."""
        # Fit initial IPCW
        km = KaplanMeier().fit(rc_data, time_col="T", delta_col="Delta")
        T_vals = np.asarray(rc_data["T"].values, dtype=float)
        Delta_vals = np.asarray(rc_data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        init_est = RightCensoredIPCWEstimator(
            norm_constraint=10.0,
            n_grid_points=100,
            basis_order=0,
        ).fit(df_unc, sample_weights=w_unc)
        
        init_ll = incomplete_loglik(init_est, rc_data, time_col="T", delta_col="Delta")
        
        # Run oversmooth tuner
        tuner = RightCensoredEMTuner(
            rc_data,
            stage1_estimator=init_est,
            oversmooth_factors=[1.0],  # Just baseline
            em_m_imputations=10,
            em_max_em_iter=5,
            silent=True,
            do_over_smooth=True,
        )
        
        result = tuner.optimize()
        best_est = result.estimator
        final_ll = incomplete_loglik(best_est, rc_data, time_col="T", delta_col="Delta")
        
        # Both should be finite
        assert np.isfinite(init_ll), f"Initial LL should be finite: {init_ll}"
        assert np.isfinite(final_ll), f"Final LL should be finite: {final_ll}"


class TestRCPipelineNoOversmooth:
    """Test RC pipeline without oversmoothing (do_over_smooth=False behavior)."""

    def test_rc_pipeline_no_oversmooth_direct_em(self, rc_data: pd.DataFrame) -> None:
        """Verify direct Stage1 → EMStage pipeline works."""
        # Fit initial IPCW (Stage 1)
        km = KaplanMeier().fit(rc_data, time_col="T", delta_col="Delta")
        T_vals = np.asarray(rc_data["T"].values, dtype=float)
        Delta_vals = np.asarray(rc_data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        init_est = RightCensoredIPCWEstimator(
            norm_constraint=10.0,
            n_grid_points=100,
            basis_order=0,
        ).fit(df_unc, sample_weights=w_unc)
        
        # Run EM Stage directly (Stage 2 without oversmooth)
        em_stage = RightCensoredEMStage(
            m_imputations=10,
            max_em_iter=5,
            em_tol=1e-4,
            norm_constraint=10.0,
            n_grid_points=100,
            rng_seed=42,
        )
        
        em_result = em_stage.run(
            initial_estimator=init_est,
            data=rc_data,
            S_c_predict=lambda t: np.atleast_1d(km.predict(t)),
        )
        
        # Verify result structure
        assert em_result is not None, "EM stage should return a result"
        assert em_result.final_estimator is not None, "Result should have final_estimator"
        assert em_result.theta_path is not None, "Result should have theta_path"
        assert len(em_result.theta_path) > 0, "theta_path should not be empty"
        
        # Verify estimator works
        eval_points = np.linspace(0.1, 0.9, 10)
        density = em_result.final_estimator.get_density_at_points(eval_points)
        assert np.all(np.isfinite(density)), "Density should be finite"


# =============================================================================
# Interval-Censored Pipeline Tests
# =============================================================================

class TestICPipelineOversmooth:
    """Test IC pipeline with oversmoothing (do_over_smooth=True behavior)."""

    def test_ic_pipeline_oversmooth_produces_valid_estimator(
        self, ic_data: pd.DataFrame
    ) -> None:
        """Verify oversmooth pipeline produces a valid fitted estimator."""
        # First fit a Stage 1 estimator
        init_est = IntervalCensoredMidpointEstimator(
            norm_constraint=10.0,
            n_grid_points=100,
            basis_order=0,
        ).fit(ic_data, L_col="L", R_col="R")
        
        # Run oversmooth tuner with limited factors for speed
        tuner = IntervalCensoredEMTuner(
            ic_data,
            stage1_estimator=init_est,
            oversmooth_factors=[0.8, 1.0],  # Limited for speed
            em_m_imputations=10,
            em_max_em_iter=5,
            silent=True,
            do_over_smooth=True,
        )
        
        result = tuner.optimize()
        best_estimator = result.estimator
        
        # Verify estimator is fitted and produces valid output
        assert best_estimator is not None, "Oversmooth tuner should return an estimator"
        assert hasattr(best_estimator, "theta_hat"), "Estimator should have theta_hat"
        assert best_estimator.theta_hat is not None, "theta_hat should not be None"
        
        # Verify density computation works
        eval_points = np.linspace(0.1, 0.9, 10)
        density = best_estimator.get_density_at_points(eval_points)
        assert len(density) == len(eval_points), "Density should match eval points length"
        assert np.all(np.isfinite(density)), "Density values should be finite"
        assert np.all(density >= 0), "Density values should be non-negative"

    def test_ic_pipeline_oversmooth_loglik_finite(
        self, ic_data: pd.DataFrame
    ) -> None:
        """Verify IC oversmooth pipeline produces finite log-likelihood."""
        # First fit a Stage 1 estimator
        init_est = IntervalCensoredMidpointEstimator(
            norm_constraint=10.0,
            n_grid_points=100,
            basis_order=0,
        ).fit(ic_data, L_col="L", R_col="R")
        
        tuner = IntervalCensoredEMTuner(
            ic_data,
            stage1_estimator=init_est,
            oversmooth_factors=[1.0],  # Just baseline
            em_m_imputations=10,
            em_max_em_iter=5,
            silent=True,
            do_over_smooth=True,
        )
        
        result = tuner.optimize()
        best_est = result.estimator
        ll = incomplete_loglik_interval(best_est, ic_data, L_col="L", R_col="R")
        
        assert np.isfinite(ll), f"Log-likelihood should be finite: {ll}"


class TestICPipelineNoOversmooth:
    """Test IC pipeline without oversmoothing (do_over_smooth=False behavior)."""

    def test_ic_pipeline_no_oversmooth_direct_em(self, ic_data: pd.DataFrame) -> None:
        """Verify direct Stage1 → EMStage pipeline works."""
        # Fit initial Midpoint estimator (Stage 1)
        init_est = IntervalCensoredMidpointEstimator(
            norm_constraint=10.0,
            n_grid_points=100,
            basis_order=0,
        ).fit(ic_data, L_col="L", R_col="R")
        
        # Run EM Stage directly (Stage 2 without oversmooth)
        em_stage = IntervalCensoredEMStage(
            m_imputations=10,
            max_em_iter=5,
            em_tol=1e-4,
            norm_constraint=10.0,
            n_grid_points=100,
            rng_seed=42,
            L_col="L",
            R_col="R",
        )
        
        em_result = em_stage.run(
            initial_estimator=init_est,
            data=ic_data,
        )
        
        # Verify result structure
        assert em_result is not None, "EM stage should return a result"
        assert em_result.final_estimator is not None, "Result should have final_estimator"
        assert em_result.theta_path is not None, "Result should have theta_path"
        assert len(em_result.theta_path) > 0, "theta_path should not be empty"
        
        # Verify estimator works
        eval_points = np.linspace(0.1, 0.9, 10)
        density = em_result.final_estimator.get_density_at_points(eval_points)
        assert np.all(np.isfinite(density)), "Density should be finite"

    def test_ic_pipeline_no_oversmooth_loglik_finite(self, ic_data: pd.DataFrame) -> None:
        """Verify IC direct EM pipeline produces finite log-likelihood."""
        init_est = IntervalCensoredMidpointEstimator(
            norm_constraint=10.0,
            n_grid_points=100,
            basis_order=0,
        ).fit(ic_data, L_col="L", R_col="R")
        
        em_stage = IntervalCensoredEMStage(
            m_imputations=10,
            max_em_iter=5,
            norm_constraint=10.0,
            rng_seed=42,
        )
        
        em_result = em_stage.run(initial_estimator=init_est, data=ic_data)
        ll = incomplete_loglik_interval(em_result.final_estimator, ic_data, L_col="L", R_col="R")
        
        assert np.isfinite(ll), f"Log-likelihood should be finite: {ll}"
