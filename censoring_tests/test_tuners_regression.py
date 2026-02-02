"""Regression tests for hyperparameter tuners.

These tests verify tuner output structure and basic functionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Right-censored tuners
from haldensity.censoring.tuners.right_tuners import (
    RightCensoredInitTuner,
    RightCensoredEMTuner,
)

# Interval-censored tuners
from haldensity.censoring.tuners.interval_tuners import (
    IntervalCensoredInitTuner,
    IntervalCensoredEMTuner,
)

from haldensity.censoring.tuners._base import TuningResult


# =============================================================================
# Right-Censored Tuner Tests
# =============================================================================

class TestRCInitTuner:
    """Regression tests for RightCensoredInitTuner."""

    def test_rc_init_tuner_output_structure(self, rc_data: pd.DataFrame) -> None:
        """Verify init tuner output contains expected keys."""
        tuner = RightCensoredInitTuner(
            data=rc_data,
            cv_folds=2,  # Minimal for speed
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)  # Minimal trials for speed
        
        # Verify output structure
        assert isinstance(result, TuningResult), "Result should be a TuningResult"
        assert result.estimator is not None, "Result should contain estimator"
        assert result.best_params is not None, "Result should contain best_params"
        assert result.metadata is not None, "Result should contain metadata"
        
        # Verify best_params has expected keys
        best_params = result.best_params
        assert "norm_constraint" in best_params, "best_params should have 'norm_constraint'"
        assert "basis_order" in best_params, "best_params should have 'basis_order'"

    def test_rc_init_tuner_params_valid(self, rc_data: pd.DataFrame) -> None:
        """Verify tuner produces valid parameter values."""
        tuner = RightCensoredInitTuner(
            data=rc_data,
            cv_folds=2,
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)
        best_params = result.best_params
        
        # Verify parameter values are valid
        assert best_params["norm_constraint"] > 0, "norm_constraint should be positive"
        assert best_params["basis_order"] >= 0, "basis_order should be non-negative"
        assert isinstance(best_params["basis_order"], int), "basis_order should be int"

    def test_rc_init_tuner_returns_valid_estimator(self, rc_data: pd.DataFrame) -> None:
        """Verify result contains a valid fitted estimator."""
        tuner = RightCensoredInitTuner(
            data=rc_data,
            cv_folds=2,
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)
        best_model = result.estimator
        
        # Verify model is fitted
        assert best_model is not None, "estimator should not be None"
        assert hasattr(best_model, "theta_hat"), "Model should have theta_hat"
        assert best_model.theta_hat is not None, "theta_hat should not be None"
        
        # Verify model can produce density
        eval_points = np.linspace(0.1, 0.9, 10)
        density = best_model.get_density_at_points(eval_points)
        assert np.all(np.isfinite(density)), "Density should be finite"


class TestRCEMTuner:
    """Regression tests for RightCensoredEMTuner."""

    def test_rc_em_tuner_output_structure(self, rc_data: pd.DataFrame) -> None:
        """Verify EM tuner returns expected records."""
        # First fit a Stage 1 estimator
        init_tuner = RightCensoredInitTuner(
            data=rc_data,
            cv_folds=2,
            silent=True,
        )
        init_result = init_tuner.optimize(n_trials=2)
        
        # Now run EM tuner
        tuner = RightCensoredEMTuner(
            rc_data,
            stage1_estimator=init_result.estimator,
            oversmooth_factors=[0.9, 1.0],  # Limited for speed
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        result = tuner.optimize()
        
        # Verify output structure
        assert isinstance(result, TuningResult), "Result should be a TuningResult"
        assert result.estimator is not None, "Result should contain estimator"
        assert result.metadata is not None, "Result should contain metadata"
        
        # Verify metadata has expected fields for oversmooth mode
        assert "init_records" in result.metadata, "metadata should have init_records"
        assert "em_records" in result.metadata, "metadata should have em_records"

    def test_rc_em_tuner_records_valid(self, rc_data: pd.DataFrame) -> None:
        """Verify EM tuner records have valid values."""
        # First fit a Stage 1 estimator
        init_tuner = RightCensoredInitTuner(
            data=rc_data,
            cv_folds=2,
            silent=True,
        )
        init_result = init_tuner.optimize(n_trials=2)
        
        tuner = RightCensoredEMTuner(
            rc_data,
            stage1_estimator=init_result.estimator,
            oversmooth_factors=[1.0],  # Single factor for simplicity
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        result = tuner.optimize()
        
        # Verify init records
        for record in result.metadata["init_records"]:
            assert record.factor > 0, "factor should be positive"
            assert record.norm_constraint > 0, "norm_constraint should be positive"
            assert record.n_knots >= 0, "n_knots should be non-negative"
            assert np.isfinite(record.log_likelihood), "log_likelihood should be finite"
        
        # Verify EM records
        for record in result.metadata["em_records"]:
            assert record.em_iterations >= 0, "em_iterations should be non-negative"
            assert np.isfinite(record.em_ll), "em_ll should be finite"
            assert record.em_estimator is not None, "em_estimator should not be None"


# =============================================================================
# Interval-Censored Tuner Tests
# =============================================================================

class TestICInitTuner:
    """Regression tests for IntervalCensoredInitTuner."""

    def test_ic_init_tuner_output_structure(self, ic_data: pd.DataFrame) -> None:
        """Verify init tuner output contains expected keys."""
        tuner = IntervalCensoredInitTuner(
            data=ic_data,
            cv_folds=2,  # Minimal for speed
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)  # Minimal trials for speed
        
        # Verify output structure
        assert isinstance(result, TuningResult), "Result should be a TuningResult"
        assert result.estimator is not None, "Result should contain estimator"
        assert result.best_params is not None, "Result should contain best_params"
        assert result.metadata is not None, "Result should contain metadata"
        
        # Verify best_params has expected keys
        best_params = result.best_params
        assert "norm_constraint" in best_params, "best_params should have 'norm_constraint'"
        assert "basis_order" in best_params, "best_params should have 'basis_order'"

    def test_ic_init_tuner_params_valid(self, ic_data: pd.DataFrame) -> None:
        """Verify tuner produces valid parameter values."""
        tuner = IntervalCensoredInitTuner(
            data=ic_data,
            cv_folds=2,
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)
        best_params = result.best_params
        
        # Verify parameter values are valid
        assert best_params["norm_constraint"] > 0, "norm_constraint should be positive"
        assert best_params["basis_order"] >= 0, "basis_order should be non-negative"
        assert isinstance(best_params["basis_order"], int), "basis_order should be int"

    def test_ic_init_tuner_returns_valid_estimator(self, ic_data: pd.DataFrame) -> None:
        """Verify result contains a valid fitted estimator."""
        tuner = IntervalCensoredInitTuner(
            data=ic_data,
            cv_folds=2,
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)
        best_model = result.estimator
        
        # Verify model is fitted
        assert best_model is not None, "estimator should not be None"
        assert hasattr(best_model, "theta_hat"), "Model should have theta_hat"
        assert best_model.theta_hat is not None, "theta_hat should not be None"
        
        # Verify model can produce density
        eval_points = np.linspace(0.1, 0.9, 10)
        density = best_model.get_density_at_points(eval_points)
        assert np.all(np.isfinite(density)), "Density should be finite"


class TestICEMTuner:
    """Regression tests for IntervalCensoredEMTuner."""

    def test_ic_em_tuner_output_structure(self, ic_data: pd.DataFrame) -> None:
        """Verify EM tuner returns expected records."""
        # First fit a Stage 1 estimator
        init_tuner = IntervalCensoredInitTuner(
            data=ic_data,
            cv_folds=2,
            silent=True,
        )
        init_result = init_tuner.optimize(n_trials=2)
        
        tuner = IntervalCensoredEMTuner(
            ic_data,
            stage1_estimator=init_result.estimator,
            oversmooth_factors=[0.9, 1.0],  # Limited for speed
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        result = tuner.optimize()
        
        # Verify output structure
        assert isinstance(result, TuningResult), "Result should be a TuningResult"
        assert result.estimator is not None, "Result should contain estimator"
        assert result.metadata is not None, "Result should contain metadata"
        
        # Verify metadata has expected fields for oversmooth mode
        assert "init_records" in result.metadata, "metadata should have init_records"
        assert "em_records" in result.metadata, "metadata should have em_records"

    def test_ic_em_tuner_records_valid(self, ic_data: pd.DataFrame) -> None:
        """Verify EM tuner records have valid values."""
        # First fit a Stage 1 estimator
        init_tuner = IntervalCensoredInitTuner(
            data=ic_data,
            cv_folds=2,
            silent=True,
        )
        init_result = init_tuner.optimize(n_trials=2)
        
        tuner = IntervalCensoredEMTuner(
            ic_data,
            stage1_estimator=init_result.estimator,
            oversmooth_factors=[1.0],  # Single factor for simplicity
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        result = tuner.optimize()
        
        # Verify init records
        for record in result.metadata["init_records"]:
            assert record.factor > 0, "factor should be positive"
            assert record.norm_constraint > 0, "norm_constraint should be positive"
            assert record.n_knots >= 0, "n_knots should be non-negative"
            assert np.isfinite(record.log_likelihood), "log_likelihood should be finite"
        
        # Verify EM records
        for record in result.metadata["em_records"]:
            assert record.em_iterations >= 0, "em_iterations should be non-negative"
            assert np.isfinite(record.em_ll), "em_ll should be finite"
            assert record.em_estimator is not None, "em_estimator should not be None"
