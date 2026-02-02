"""Regression tests for censored data estimators.

These tests verify that estimator outputs match expected values captured before refactoring.
This ensures the refactor preserves numerical behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring.right.estimators import RightCensoredInitEstimator, RightCensoredEMEstimator
from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.interval.estimators import IntervalCensoredInitEstimator, IntervalCensoredEMEstimator

from conftest import (
    assert_theta_close,
    assert_density_close,
    assert_loglik_close,
    THETA_RTOL,
    THETA_ATOL,
    DENSITY_RTOL,
    DENSITY_ATOL,
)


# =============================================================================
# Right-Censored Init (IPCW) Estimator Tests
# =============================================================================

class TestRCInitEstimator:
    """Regression tests for RightCensoredInitEstimator (future: RightCensoredInitEstimator)."""

    @pytest.fixture
    def fitted_rc_init(self, rc_data: pd.DataFrame, rc_init_params: dict) -> RightCensoredInitEstimator:
        """Fit RC Init estimator with fixture data and params."""
        # Compute IPCW weights
        km = KaplanMeier().fit(rc_data, time_col="T", delta_col="Delta")
        T_vals = np.asarray(rc_data["T"].values, dtype=float)
        Delta_vals = np.asarray(rc_data["Delta"].values, dtype=int)
        weights = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))
        
        # Fit on uncensored observations
        unc_mask = Delta_vals == 1
        df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
        w_unc = weights[unc_mask]
        
        est = RightCensoredInitEstimator(**rc_init_params)
        est.fit(df_unc, sample_weights=w_unc)
        return est

    def test_rc_init_estimator_theta_hat(
        self, fitted_rc_init: RightCensoredInitEstimator, rc_expected: dict
    ) -> None:
        """Verify theta_hat coefficients match expected values."""
        expected_theta = np.array(rc_expected["init"]["results"]["theta_hat"])
        actual_theta = fitted_rc_init.theta_hat
        
        assert actual_theta is not None, "theta_hat should not be None"
        assert_theta_close(actual_theta, expected_theta, name="RC Init theta_hat")

    def test_rc_init_estimator_density(
        self, fitted_rc_init: RightCensoredInitEstimator, rc_expected: dict, eval_points: np.ndarray
    ) -> None:
        """Verify density values at evaluation points match expected."""
        expected_density = np.array(rc_expected["init"]["density_at_eval_points"])
        actual_density = fitted_rc_init.get_density_at_points(eval_points)
        
        assert_density_close(actual_density, expected_density, name="RC Init density")

    def test_rc_init_estimator_n_knots(
        self, fitted_rc_init: RightCensoredInitEstimator, rc_expected: dict
    ) -> None:
        """Verify number of selected knots matches expected."""
        expected_n_knots = rc_expected["init"]["results"]["n_selected_knots"]
        results = fitted_rc_init.get_results()
        actual_n_knots = results["n_selected_knots"]
        
        assert actual_n_knots == expected_n_knots, (
            f"RC Init n_knots mismatch: actual={actual_n_knots}, expected={expected_n_knots}"
        )

    def test_rc_init_estimator_selected_knots(
        self, fitted_rc_init: RightCensoredInitEstimator, rc_expected: dict
    ) -> None:
        """Verify selected knot locations match expected."""
        expected_knots = np.array(rc_expected["init"]["results"]["grid_points_hal_selected"])
        actual_knots = fitted_rc_init.grid_points_hal_selected
        
        assert actual_knots is not None, "grid_points_hal_selected should not be None"
        assert len(actual_knots) == len(expected_knots), (
            f"RC Init knot count mismatch: actual={len(actual_knots)}, expected={len(expected_knots)}"
        )
        assert_theta_close(actual_knots, expected_knots, name="RC Init selected_knots")


# =============================================================================
# Right-Censored EM Estimator Tests
# =============================================================================

class TestRCEMEstimator:
    """Regression tests for RightCensoredEMEstimator."""

    @pytest.fixture
    def fitted_rc_em(self, rc_data: pd.DataFrame, rc_em_params: dict) -> RightCensoredEMEstimator:
        """Fit RC EM estimator with fixture data and params."""
        est = RightCensoredEMEstimator(**rc_em_params)
        est.fit(rc_data)
        return est

    def test_rc_em_estimator_theta_hat(
        self, fitted_rc_em: RightCensoredEMEstimator, rc_expected: dict
    ) -> None:
        """Verify final theta_hat coefficients match expected values."""
        expected_theta = np.array(rc_expected["em"]["results"]["theta_hat"])
        actual_theta = fitted_rc_em.theta_hat
        
        assert actual_theta is not None, "theta_hat should not be None"
        assert_theta_close(actual_theta, expected_theta, name="RC EM theta_hat")

    def test_rc_em_estimator_density(
        self, fitted_rc_em: RightCensoredEMEstimator, rc_expected: dict, eval_points: np.ndarray
    ) -> None:
        """Verify density values at evaluation points match expected."""
        expected_density = np.array(rc_expected["em"]["density_at_eval_points"])
        actual_density = fitted_rc_em.get_density_at_points(eval_points)
        
        assert_density_close(actual_density, expected_density, name="RC EM density")

    def test_rc_em_estimator_convergence(
        self, fitted_rc_em: RightCensoredEMEstimator, rc_expected: dict
    ) -> None:
        """Verify EM convergence status and iteration count."""
        expected_iters = rc_expected["em"]["em_iterations"]
        expected_converged = rc_expected["em"]["em_converged"]
        
        assert fitted_rc_em.em_iterations_ == expected_iters, (
            f"RC EM iterations mismatch: actual={fitted_rc_em.em_iterations_}, expected={expected_iters}"
        )
        assert fitted_rc_em.em_converged_ == expected_converged, (
            f"RC EM converged mismatch: actual={fitted_rc_em.em_converged_}, expected={expected_converged}"
        )

    def test_rc_em_estimator_theta_path_length(
        self, fitted_rc_em: RightCensoredEMEstimator, rc_expected: dict
    ) -> None:
        """Verify theta_path has expected length (iterations + 1 for initial)."""
        expected_path = rc_expected["em"]["results"]["theta_path"]
        actual_path = fitted_rc_em.theta_path_
        
        assert len(actual_path) == len(expected_path), (
            f"RC EM theta_path length mismatch: actual={len(actual_path)}, expected={len(expected_path)}"
        )

    def test_rc_em_estimator_n_knots(
        self, fitted_rc_em: RightCensoredEMEstimator, rc_expected: dict
    ) -> None:
        """Verify number of selected knots matches expected."""
        expected_n_knots = rc_expected["em"]["results"]["n_selected_knots"]
        results = fitted_rc_em.get_results()
        actual_n_knots = results["n_selected_knots"]
        
        assert actual_n_knots == expected_n_knots, (
            f"RC EM n_knots mismatch: actual={actual_n_knots}, expected={expected_n_knots}"
        )


# =============================================================================
# Interval-Censored Init (Midpoint) Estimator Tests
# =============================================================================

class TestICInitEstimator:
    """Regression tests for IntervalCensoredInitEstimator (future: IntervalCensoredInitEstimator)."""

    @pytest.fixture
    def fitted_ic_init(self, ic_data: pd.DataFrame, ic_init_params: dict) -> IntervalCensoredInitEstimator:
        """Fit IC Init estimator with fixture data and params."""
        est = IntervalCensoredInitEstimator(**ic_init_params)
        est.fit(ic_data, L_col="L", R_col="R")
        return est

    def test_ic_init_estimator_theta_hat(
        self, fitted_ic_init: IntervalCensoredInitEstimator, ic_expected: dict
    ) -> None:
        """Verify theta_hat coefficients match expected values."""
        expected_theta = np.array(ic_expected["init"]["results"]["theta_hat"])
        actual_theta = fitted_ic_init.theta_hat
        
        assert actual_theta is not None, "theta_hat should not be None"
        assert_theta_close(actual_theta, expected_theta, name="IC Init theta_hat")

    def test_ic_init_estimator_density(
        self, fitted_ic_init: IntervalCensoredInitEstimator, ic_expected: dict, eval_points: np.ndarray
    ) -> None:
        """Verify density values at evaluation points match expected."""
        expected_density = np.array(ic_expected["init"]["density_at_eval_points"])
        actual_density = fitted_ic_init.get_density_at_points(eval_points)
        
        assert_density_close(actual_density, expected_density, name="IC Init density")

    def test_ic_init_estimator_n_knots(
        self, fitted_ic_init: IntervalCensoredInitEstimator, ic_expected: dict
    ) -> None:
        """Verify number of selected knots matches expected."""
        expected_n_knots = ic_expected["init"]["results"]["n_selected_knots"]
        results = fitted_ic_init.get_results()
        actual_n_knots = results["n_selected_knots"]
        
        assert actual_n_knots == expected_n_knots, (
            f"IC Init n_knots mismatch: actual={actual_n_knots}, expected={expected_n_knots}"
        )


# =============================================================================
# Interval-Censored EM Estimator Tests
# =============================================================================

class TestICEMEstimator:
    """Regression tests for IntervalCensoredEMEstimator."""

    @pytest.fixture
    def fitted_ic_em(self, ic_data: pd.DataFrame, ic_em_params: dict) -> IntervalCensoredEMEstimator:
        """Fit IC EM estimator with fixture data and params."""
        est = IntervalCensoredEMEstimator(**ic_em_params)
        est.fit(ic_data)
        return est

    def test_ic_em_estimator_theta_hat(
        self, fitted_ic_em: IntervalCensoredEMEstimator, ic_expected: dict
    ) -> None:
        """Verify final theta_hat coefficients match expected values."""
        expected_theta = np.array(ic_expected["em"]["results"]["theta_hat"])
        actual_theta = fitted_ic_em.theta_hat
        
        assert actual_theta is not None, "theta_hat should not be None"
        assert_theta_close(actual_theta, expected_theta, name="IC EM theta_hat")

    def test_ic_em_estimator_density(
        self, fitted_ic_em: IntervalCensoredEMEstimator, ic_expected: dict, eval_points: np.ndarray
    ) -> None:
        """Verify density values at evaluation points match expected."""
        expected_density = np.array(ic_expected["em"]["density_at_eval_points"])
        actual_density = fitted_ic_em.get_density_at_points(eval_points)
        
        assert_density_close(actual_density, expected_density, name="IC EM density")

    def test_ic_em_estimator_convergence(
        self, fitted_ic_em: IntervalCensoredEMEstimator, ic_expected: dict
    ) -> None:
        """Verify EM convergence status and iteration count."""
        expected_iters = ic_expected["em"]["em_iterations"]
        expected_converged = ic_expected["em"]["em_converged"]
        
        assert fitted_ic_em.em_iterations_ == expected_iters, (
            f"IC EM iterations mismatch: actual={fitted_ic_em.em_iterations_}, expected={expected_iters}"
        )
        assert fitted_ic_em.em_converged_ == expected_converged, (
            f"IC EM converged mismatch: actual={fitted_ic_em.em_converged_}, expected={expected_converged}"
        )

    def test_ic_em_estimator_n_knots(
        self, fitted_ic_em: IntervalCensoredEMEstimator, ic_expected: dict
    ) -> None:
        """Verify number of selected knots matches expected."""
        expected_n_knots = ic_expected["em"]["results"]["n_selected_knots"]
        results = fitted_ic_em.get_results()
        actual_n_knots = results["n_selected_knots"]
        
        assert actual_n_knots == expected_n_knots, (
            f"IC EM n_knots mismatch: actual={actual_n_knots}, expected={expected_n_knots}"
        )
