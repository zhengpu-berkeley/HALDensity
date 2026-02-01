"""Pytest configuration and shared fixtures for regression tests.

This module provides:
- Fixtures for loading test data from JSON files
- Helper functions for numerical comparison
- Synthetic data generators with fixed seeds
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Numerical comparison thresholds
THETA_RTOL = 1e-6
THETA_ATOL = 1e-10
DENSITY_RTOL = 1e-4
DENSITY_ATOL = 1e-8
LOGLIK_RTOL = 1e-6
LOGLIK_ATOL = 1e-8


# =============================================================================
# Helper functions for numerical comparison
# =============================================================================

def assert_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = DENSITY_RTOL,
    atol: float = DENSITY_ATOL,
    name: str = "array",
) -> None:
    """Assert two arrays are close with helpful error message."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: Shape mismatch. Actual: {actual.shape}, Expected: {expected.shape}"
        )
    
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.abs(actual - expected)
        max_diff_idx = np.argmax(diff)
        max_diff = diff.flat[max_diff_idx]
        raise AssertionError(
            f"{name}: Arrays not close. Max diff: {max_diff:.2e} at index {max_diff_idx}. "
            f"Actual: {actual.flat[max_diff_idx]:.6f}, Expected: {expected.flat[max_diff_idx]:.6f}"
        )


def assert_scalar_close(
    actual: float,
    expected: float,
    rtol: float = LOGLIK_RTOL,
    atol: float = LOGLIK_ATOL,
    name: str = "scalar",
) -> None:
    """Assert two scalars are close with helpful error message."""
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        diff = abs(actual - expected)
        raise AssertionError(
            f"{name}: Values not close. Actual: {actual:.8f}, Expected: {expected:.8f}, "
            f"Diff: {diff:.2e}"
        )


def assert_theta_close(actual: np.ndarray, expected: np.ndarray, name: str = "theta") -> None:
    """Assert theta coefficients are close (stricter tolerance)."""
    assert_array_close(actual, expected, rtol=THETA_RTOL, atol=THETA_ATOL, name=name)


def assert_density_close(actual: np.ndarray, expected: np.ndarray, name: str = "density") -> None:
    """Assert density values are close."""
    assert_array_close(actual, expected, rtol=DENSITY_RTOL, atol=DENSITY_ATOL, name=name)


def assert_loglik_close(actual: float, expected: float, name: str = "loglik") -> None:
    """Assert log-likelihood values are close."""
    assert_scalar_close(actual, expected, rtol=LOGLIK_RTOL, atol=LOGLIK_ATOL, name=name)


# =============================================================================
# Data loading functions
# =============================================================================

def load_json(filename: str) -> dict:
    """Load a JSON file from the fixtures directory."""
    filepath = FIXTURES_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Fixture file not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


# =============================================================================
# Synthetic data generators (for unit tests that don't need fixtures)
# =============================================================================

def generate_right_censored_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic right-censored data.
    
    True event times ~ Beta(2, 5) scaled to [0, 1]
    Censoring times ~ Uniform(0, 1)
    """
    rng = np.random.default_rng(seed)
    
    t_event = rng.beta(2, 5, size=n)
    t_cens = rng.uniform(0.0, 1.0, size=n)
    
    t_obs = np.minimum(t_event, t_cens)
    delta = (t_event <= t_cens).astype(int)
    
    # Ensure mix of censored/uncensored
    if delta.sum() == 0 or delta.sum() == n:
        delta[:n // 10] = 1
        delta[-n // 10:] = 0
    
    return pd.DataFrame({
        "T": t_obs.astype(float),
        "Delta": delta.astype(int),
    })


def generate_interval_censored_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic interval-censored data.
    
    True event times ~ Beta(2, 5) scaled to [0, 1]
    Interval width ~ Uniform(0.05, 0.2)
    """
    rng = np.random.default_rng(seed)
    
    t_event = rng.beta(2, 5, size=n)
    widths = rng.uniform(0.05, 0.2, size=n)
    
    left_shift = rng.uniform(0, 1, size=n) * widths
    L = np.clip(t_event - left_shift, 0.0, 1.0)
    R = np.clip(L + widths, 0.0, 1.0)
    
    R = np.maximum(R, L + 0.01)
    R = np.minimum(R, 1.0)
    
    return pd.DataFrame({
        "L": L.astype(float),
        "R": R.astype(float),
    })


# =============================================================================
# Pytest fixtures
# =============================================================================

@pytest.fixture(scope="session")
def rc_data() -> pd.DataFrame:
    """Load right-censored test data from fixture."""
    data = load_json("synthetic_right_censored.json")
    return pd.DataFrame({
        "T": data["T"],
        "Delta": data["Delta"],
    })


@pytest.fixture(scope="session")
def ic_data() -> pd.DataFrame:
    """Load interval-censored test data from fixture."""
    data = load_json("synthetic_interval_censored.json")
    return pd.DataFrame({
        "L": data["L"],
        "R": data["R"],
    })


@pytest.fixture(scope="session")
def rc_expected() -> dict:
    """Load expected results for right-censored estimators."""
    return load_json("expected_results_rc.json")


@pytest.fixture(scope="session")
def ic_expected() -> dict:
    """Load expected results for interval-censored estimators."""
    return load_json("expected_results_ic.json")


@pytest.fixture(scope="session")
def eval_points() -> np.ndarray:
    """Standard evaluation points for density comparison."""
    return np.linspace(0.01, 0.99, 50)


# =============================================================================
# Estimator parameter fixtures
# =============================================================================

@pytest.fixture(scope="session")
def rc_init_params(rc_expected: dict) -> dict:
    """Get parameters for RC Init estimator."""
    return rc_expected["params"]["init"]


@pytest.fixture(scope="session")
def rc_em_params(rc_expected: dict) -> dict:
    """Get parameters for RC EM estimator."""
    return rc_expected["params"]["em"]


@pytest.fixture(scope="session")
def ic_init_params(ic_expected: dict) -> dict:
    """Get parameters for IC Init estimator."""
    return ic_expected["params"]["init"]


@pytest.fixture(scope="session")
def ic_em_params(ic_expected: dict) -> dict:
    """Get parameters for IC EM estimator."""
    return ic_expected["params"]["em"]
