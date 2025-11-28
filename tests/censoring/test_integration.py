"""Integration tests for censoring module workflows."""

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring import (
    KaplanMeier,
    compute_ipcw_weights,
    RightCensoredIPCWEstimator,
    RightCensoredEMEstimator,
    incomplete_loglik,
    kl_divergence,
    pipelines,
)
from haldensity.utils.density_computations import (
    generic_compute_survival_from_density,
    generic_compute_cdf_from_density,
)


class TestFullPipelines:
    """End-to-end workflow tests."""

    def test_ipcw_pipeline_produces_valid_density(self, truncated_normal_data):
        """IPCW pipeline should produce valid density."""
        data, true_pdf = truncated_normal_data

        # Fit KM
        km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")

        # Compute weights
        weights = compute_ipcw_weights(data["T"].values, data["Delta"].values, km.predict)

        # Fit IPCW estimator
        uncensored_mask = data["Delta"].values == 1
        df_unc = pd.DataFrame({"W1": data["T"].values[uncensored_mask]})

        est = RightCensoredIPCWEstimator(norm_constraint=50.0)
        est.fit(df_unc, sample_weights=weights[uncensored_mask])

        # Validate density
        grid, density = est.get_density()
        assert np.all(density >= 0)
        assert np.all(np.isfinite(density))

        # Check KL divergence is finite (just validate it computes without error)
        eval_grid = np.linspace(0.05, 0.95, 100)
        eval_density = est.get_density_at_points(eval_grid)
        kl = kl_divergence(true_pdf, eval_grid, eval_density)
        assert np.isfinite(kl)

    def test_em_pipeline_produces_valid_density(self, truncated_normal_data):
        """EM pipeline should produce valid density."""
        data, true_pdf = truncated_normal_data

        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=10,
            max_em_iter=5,
        )
        est.fit(data)

        # Validate density
        grid, density = est.get_density()
        assert np.all(density >= 0)
        assert np.all(np.isfinite(density))

        # Check KL divergence is finite (just validate it computes without error)
        eval_grid = np.linspace(0.05, 0.95, 100)
        eval_density = est.get_density_at_points(eval_grid)
        kl = kl_divergence(true_pdf, eval_grid, eval_density)
        assert np.isfinite(kl)

    def test_survival_cdf_monotonicity(self, truncated_normal_data):
        """Survival should decrease, CDF should increase."""
        data, _ = truncated_normal_data

        est = RightCensoredEMEstimator(
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )
        est.fit(data)

        grid, density = est.get_density()

        # Compute CDF and survival
        cdf, cdf_grid = generic_compute_cdf_from_density(grid, density)
        surv, surv_grid = generic_compute_survival_from_density(grid, density)

        # CDF should be non-decreasing
        assert np.all(np.diff(cdf) >= -1e-8)

        # Survival should be non-increasing
        assert np.all(np.diff(surv) <= 1e-8)


class TestPipelinesModule:
    """Tests for convenience pipeline functions."""

    def test_run_ipcw_hal_mle(self, truncated_normal_data):
        """run_ipcw_hal_mle should work correctly."""
        data, _ = truncated_normal_data

        results = pipelines.run_ipcw_hal_mle(
            data,
            norm_constraint=50.0,
            basis_order=0,
        )

        assert "theta_hat" in results
        assert "estimated_density" in results
        assert "grid_points" in results

    def test_run_ipcw_hal_mle_with_estimator(self, truncated_normal_data):
        """run_ipcw_hal_mle with return_estimator=True."""
        data, _ = truncated_normal_data

        results, est = pipelines.run_ipcw_hal_mle(
            data,
            norm_constraint=50.0,
            return_estimator=True,
        )

        assert est is not None
        assert est.is_fitted
        grid, density = est.get_density()
        assert np.all(density >= 0)

    def test_run_em_ipcw_hal_mle(self, truncated_normal_data):
        """run_em_ipcw_hal_mle should work correctly."""
        data, _ = truncated_normal_data

        results = pipelines.run_em_ipcw_hal_mle(
            data,
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
        )

        assert "theta_hat" in results
        assert "estimated_density" in results
        assert "em_iterations" in results
        assert "em_converged" in results

    def test_run_em_ipcw_hal_mle_with_estimator(self, truncated_normal_data):
        """run_em_ipcw_hal_mle with return_estimator=True."""
        data, _ = truncated_normal_data

        results, est = pipelines.run_em_ipcw_hal_mle(
            data,
            norm_constraint=50.0,
            m_imputations=5,
            max_em_iter=2,
            return_estimator=True,
        )

        assert est is not None
        assert est.is_fitted
        assert est.em_iterations_ >= 1


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_old_import_paths_work(self):
        """Old import patterns should still work."""
        from haldensity.censoring import (
            WeightedCVXPYEstimator,
            EMIPCWEstimator,
            EMStage,
            EMStageResult,
            KaplanMeier,
            compute_ipcw_weights,
            incomplete_loglik,
            kl_divergence,
        )

        # These should all be importable
        assert WeightedCVXPYEstimator is not None
        assert EMIPCWEstimator is not None
        assert EMStage is not None
        assert EMStageResult is not None

    def test_old_class_names_are_aliases(self):
        """Old class names should be aliases."""
        from haldensity.censoring import (
            WeightedCVXPYEstimator,
            EMIPCWEstimator,
            RightCensoredIPCWEstimator,
            RightCensoredEMEstimator,
        )

        assert WeightedCVXPYEstimator is RightCensoredIPCWEstimator
        assert EMIPCWEstimator is RightCensoredEMEstimator

