"""Shared fixtures for censoring tests."""

from typing import Callable, Tuple
import numpy as np
import pandas as pd
import pytest
from scipy.stats import truncnorm


@pytest.fixture
def truncated_normal_params() -> dict:
    """Parameters for truncated normal distribution."""
    return {
        "mean": 0.5,
        "std": 0.1,
        "lower": 0.0,
        "upper": 1.0,
    }


@pytest.fixture
def truncated_normal_data(
    truncated_normal_params: dict,
) -> Tuple[pd.DataFrame, Callable[[np.ndarray], np.ndarray]]:
    """Generate right-censored data from truncated normal.

    Returns
    -------
    tuple
        (data, true_pdf) where data has columns 'T' and 'Delta'.
    """
    rng = np.random.default_rng(42)
    n_samples = 300

    mean = truncated_normal_params["mean"]
    std = truncated_normal_params["std"]
    lower = truncated_normal_params["lower"]
    upper = truncated_normal_params["upper"]

    # Truncated normal event times
    a, b = (lower - mean) / std, (upper - mean) / std
    X_true = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples, random_state=rng)

    # Uniform censoring times
    C = rng.uniform(lower, upper, size=n_samples)

    # Observed data
    T = np.minimum(X_true, C)
    Delta = (X_true <= C).astype(int)

    data = pd.DataFrame({"T": T, "Delta": Delta})
    true_pdf = lambda x: truncnorm.pdf(x, a, b, loc=mean, scale=std)

    return data, true_pdf


@pytest.fixture
def larger_truncated_normal_data(
    truncated_normal_params: dict,
) -> Tuple[pd.DataFrame, Callable[[np.ndarray], np.ndarray]]:
    """Generate larger right-censored dataset for more reliable tests."""
    rng = np.random.default_rng(123)
    n_samples = 500

    mean = truncated_normal_params["mean"]
    std = truncated_normal_params["std"]
    lower = truncated_normal_params["lower"]
    upper = truncated_normal_params["upper"]

    a, b = (lower - mean) / std, (upper - mean) / std
    X_true = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples, random_state=rng)
    C = rng.uniform(lower, upper, size=n_samples)

    T = np.minimum(X_true, C)
    Delta = (X_true <= C).astype(int)

    data = pd.DataFrame({"T": T, "Delta": Delta})
    true_pdf = lambda x: truncnorm.pdf(x, a, b, loc=mean, scale=std)

    return data, true_pdf


@pytest.fixture
def km_fitted(truncated_normal_data):
    """Pre-fitted KaplanMeier estimator."""
    from haldensity.censoring import KaplanMeier

    data, _ = truncated_normal_data
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    return km, data


@pytest.fixture
def ipcw_weights(km_fitted):
    """Computed IPCW weights."""
    from haldensity.censoring import compute_ipcw_weights

    km, data = km_fitted
    T_vals = np.asarray(data["T"].values, dtype=float)
    Delta_vals = np.asarray(data["Delta"].values, dtype=int)

    weights = compute_ipcw_weights(
        T=T_vals,
        Delta=Delta_vals,
        S_c_predict=lambda x: np.atleast_1d(km.predict(x)),
    )
    return weights, data

