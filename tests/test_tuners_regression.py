"""Regression tests for hyperparameter tuners.

These tests verify tuner output structure and basic functionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Right-censored tuners
from haldensity.censoring.tuners.joint_tuner import RightCensoredOptunaHyperparameterTuner
from haldensity.censoring.tuners.em_stage_oversmooth_tuner import RightCensoredEMStageOverSmoothTuner

# Interval-censored tuners
from haldensity.censoring.tuners.interval_joint_tuner import IntervalCensoredOptunaHyperparameterTuner
from haldensity.censoring.tuners.interval_em_stage_oversmooth_tuner import IntervalCensoredEMStageOverSmoothTuner


# =============================================================================
# Right-Censored Tuner Tests
# =============================================================================

class TestRCJointTuner:
    """Regression tests for RightCensoredOptunaHyperparameterTuner."""

    def test_rc_joint_tuner_output_structure(self, rc_data: pd.DataFrame) -> None:
        """Verify joint tuner output contains expected keys."""
        tuner = RightCensoredOptunaHyperparameterTuner(
            estimator_name="RightCensoredIPCWEstimator",
            data=rc_data,
            cv_folds=2,  # Minimal for speed
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)  # Minimal trials for speed
        
        # Verify output structure
        assert isinstance(result, dict), "Result should be a dict"
        assert "best_params" in result, "Result should contain 'best_params'"
        assert "optuna_params" in result, "Result should contain 'optuna_params'"
        assert "best_metric_value" in result, "Result should contain 'best_metric_value'"
        
        # Verify best_params has expected keys
        best_params = result["best_params"]
        assert "norm_constraint" in best_params, "best_params should have 'norm_constraint'"
        assert "basis_order" in best_params, "best_params should have 'basis_order'"

    def test_rc_joint_tuner_params_valid(self, rc_data: pd.DataFrame) -> None:
        """Verify tuner produces valid parameter values."""
        tuner = RightCensoredOptunaHyperparameterTuner(
            estimator_name="RightCensoredIPCWEstimator",
            data=rc_data,
            cv_folds=2,
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)
        best_params = result["best_params"]
        
        # Verify parameter values are valid
        assert best_params["norm_constraint"] > 0, "norm_constraint should be positive"
        assert best_params["basis_order"] >= 0, "basis_order should be non-negative"
        assert isinstance(best_params["basis_order"], int), "basis_order should be int"

    def test_rc_joint_tuner_fit_best_model(self, rc_data: pd.DataFrame) -> None:
        """Verify fit_best_model returns a valid estimator."""
        tuner = RightCensoredOptunaHyperparameterTuner(
            estimator_name="RightCensoredIPCWEstimator",
            data=rc_data,
            cv_folds=2,
            silent=True,
        )
        
        tuner.optimize(n_trials=3)
        best_model = tuner.fit_best_model()
        
        # Verify model is fitted
        assert best_model is not None, "fit_best_model should return an estimator"
        assert hasattr(best_model, "theta_hat"), "Model should have theta_hat"
        assert best_model.theta_hat is not None, "theta_hat should not be None"
        
        # Verify model can produce density
        eval_points = np.linspace(0.1, 0.9, 10)
        density = best_model.get_density_at_points(eval_points)
        assert np.all(np.isfinite(density)), "Density should be finite"


class TestRCEMStageTuner:
    """Regression tests for RightCensoredEMStageOverSmoothTuner."""

    def test_rc_em_stage_tuner_output_structure(self, rc_data: pd.DataFrame) -> None:
        """Verify EM stage tuner returns expected records."""
        tuner = RightCensoredEMStageOverSmoothTuner(
            rc_data,
            ipcw_params={"norm_constraint": 10.0, "basis_order": 0},
            oversmooth_factors=[0.9, 1.0],  # Limited for speed
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        best_est = tuner.fit_best_estimator()
        
        # Verify internal records were populated
        assert tuner.ipcw_records is not None, "ipcw_records should be populated"
        assert len(tuner.ipcw_records) > 0, "ipcw_records should not be empty"
        assert tuner.em_records is not None, "em_records should be populated"
        assert len(tuner.em_records) > 0, "em_records should not be empty"
        assert tuner.best_em_record is not None, "best_em_record should be set"

    def test_rc_em_stage_tuner_records_valid(self, rc_data: pd.DataFrame) -> None:
        """Verify EM stage tuner records have valid values."""
        tuner = RightCensoredEMStageOverSmoothTuner(
            rc_data,
            ipcw_params={"norm_constraint": 10.0, "basis_order": 0},
            oversmooth_factors=[1.0],  # Single factor for simplicity
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        tuner.fit_best_estimator()
        
        # Verify IPCW records
        for record in tuner.ipcw_records:
            assert record.factor > 0, "factor should be positive"
            assert record.norm_constraint > 0, "norm_constraint should be positive"
            assert record.n_knots >= 0, "n_knots should be non-negative"
            assert np.isfinite(record.log_likelihood), "log_likelihood should be finite"
        
        # Verify EM records
        for record in tuner.em_records:
            assert record.em_iterations >= 0, "em_iterations should be non-negative"
            assert np.isfinite(record.em_ll), "em_ll should be finite"
            assert record.em_estimator is not None, "em_estimator should not be None"


# =============================================================================
# Interval-Censored Tuner Tests
# =============================================================================

class TestICJointTuner:
    """Regression tests for IntervalCensoredOptunaHyperparameterTuner."""

    def test_ic_joint_tuner_output_structure(self, ic_data: pd.DataFrame) -> None:
        """Verify joint tuner output contains expected keys."""
        tuner = IntervalCensoredOptunaHyperparameterTuner(
            estimator_name="IntervalCensoredMidpointEstimator",
            data=ic_data,
            cv_folds=2,  # Minimal for speed
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)  # Minimal trials for speed
        
        # Verify output structure
        assert isinstance(result, dict), "Result should be a dict"
        assert "best_params" in result, "Result should contain 'best_params'"
        assert "optuna_params" in result, "Result should contain 'optuna_params'"
        assert "best_metric_value" in result, "Result should contain 'best_metric_value'"
        
        # Verify best_params has expected keys
        best_params = result["best_params"]
        assert "norm_constraint" in best_params, "best_params should have 'norm_constraint'"
        assert "basis_order" in best_params, "best_params should have 'basis_order'"

    def test_ic_joint_tuner_params_valid(self, ic_data: pd.DataFrame) -> None:
        """Verify tuner produces valid parameter values."""
        tuner = IntervalCensoredOptunaHyperparameterTuner(
            estimator_name="IntervalCensoredMidpointEstimator",
            data=ic_data,
            cv_folds=2,
            silent=True,
        )
        
        result = tuner.optimize(n_trials=3)
        best_params = result["best_params"]
        
        # Verify parameter values are valid
        assert best_params["norm_constraint"] > 0, "norm_constraint should be positive"
        assert best_params["basis_order"] >= 0, "basis_order should be non-negative"
        assert isinstance(best_params["basis_order"], int), "basis_order should be int"

    def test_ic_joint_tuner_fit_best_model(self, ic_data: pd.DataFrame) -> None:
        """Verify fit_best_model returns a valid estimator."""
        tuner = IntervalCensoredOptunaHyperparameterTuner(
            estimator_name="IntervalCensoredMidpointEstimator",
            data=ic_data,
            cv_folds=2,
            silent=True,
        )
        
        tuner.optimize(n_trials=3)
        best_model = tuner.fit_best_model()
        
        # Verify model is fitted
        assert best_model is not None, "fit_best_model should return an estimator"
        assert hasattr(best_model, "theta_hat"), "Model should have theta_hat"
        assert best_model.theta_hat is not None, "theta_hat should not be None"
        
        # Verify model can produce density
        eval_points = np.linspace(0.1, 0.9, 10)
        density = best_model.get_density_at_points(eval_points)
        assert np.all(np.isfinite(density)), "Density should be finite"


class TestICEMStageTuner:
    """Regression tests for IntervalCensoredEMStageOverSmoothTuner."""

    def test_ic_em_stage_tuner_output_structure(self, ic_data: pd.DataFrame) -> None:
        """Verify EM stage tuner returns expected records."""
        tuner = IntervalCensoredEMStageOverSmoothTuner(
            ic_data,
            midpoint_params={"norm_constraint": 10.0, "basis_order": 0},
            oversmooth_factors=[0.9, 1.0],  # Limited for speed
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        best_est = tuner.fit_best_estimator()
        
        # Verify internal records were populated
        assert tuner.init_records is not None, "init_records should be populated"
        assert len(tuner.init_records) > 0, "init_records should not be empty"
        assert tuner.em_records is not None, "em_records should be populated"
        assert len(tuner.em_records) > 0, "em_records should not be empty"
        assert tuner.best_em_record is not None, "best_em_record should be set"

    def test_ic_em_stage_tuner_records_valid(self, ic_data: pd.DataFrame) -> None:
        """Verify EM stage tuner records have valid values."""
        tuner = IntervalCensoredEMStageOverSmoothTuner(
            ic_data,
            midpoint_params={"norm_constraint": 10.0, "basis_order": 0},
            oversmooth_factors=[1.0],  # Single factor for simplicity
            em_m_imputations=5,
            em_max_em_iter=3,
            silent=True,
        )
        
        tuner.fit_best_estimator()
        
        # Verify init records
        for record in tuner.init_records:
            assert record.factor > 0, "factor should be positive"
            assert record.norm_constraint > 0, "norm_constraint should be positive"
            assert record.n_knots >= 0, "n_knots should be non-negative"
            assert np.isfinite(record.log_likelihood), "log_likelihood should be finite"
        
        # Verify EM records
        for record in tuner.em_records:
            assert record.em_iterations >= 0, "em_iterations should be non-negative"
            assert np.isfinite(record.em_ll), "em_ll should be finite"
            assert record.em_estimator is not None, "em_estimator should not be None"
