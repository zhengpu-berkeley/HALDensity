from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from ..multivariate_basis import coerce_multivariate_data


class BivariateTruncatedNormal:
    """Bivariate normal distribution truncated to a rectangular box."""

    def __init__(
        self,
        mean: tuple[float, float] = (0.5, 0.5),
        covariance: np.ndarray | list[list[float]] = ((0.02, 0.0), (0.0, 0.02)),
        lower: tuple[float, float] = (0.0, 0.0),
        upper: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

        if self.mean.shape != (2,):
            raise ValueError("mean must have length 2")
        if self.covariance.shape != (2, 2):
            raise ValueError("covariance must have shape (2, 2)")
        if self.lower.shape != (2,) or self.upper.shape != (2,):
            raise ValueError("lower and upper must have length 2")
        if np.any(self.lower >= self.upper):
            raise ValueError("Each lower bound must be strictly less than upper bound")

        self.base_distribution = multivariate_normal(
            mean=self.mean,
            cov=self.covariance,
        )
        self.normalizer = self._rectangle_probability(self.lower, self.upper)
        if self.normalizer <= 0.0:
            raise ValueError("Truncation region has zero probability under the base distribution")

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (float(self.lower[0]), float(self.upper[0])),
            (float(self.lower[1]), float(self.upper[1])),
        )

    def _rectangle_probability(self, lower: np.ndarray, upper: np.ndarray) -> float:
        upper_upper = self.base_distribution.cdf(upper)
        lower_upper = self.base_distribution.cdf(
            np.array([lower[0], upper[1]], dtype=float)
        )
        upper_lower = self.base_distribution.cdf(
            np.array([upper[0], lower[1]], dtype=float)
        )
        lower_lower = self.base_distribution.cdf(lower)
        return float(upper_upper - lower_upper - upper_lower + lower_lower)

    def _inside_support(self, points: np.ndarray) -> np.ndarray:
        return np.all((points >= self.lower) & (points <= self.upper), axis=1)

    def generate_samples(self, n_samples: int, seed: int | None = None) -> pd.DataFrame:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        rng = np.random.default_rng(seed)
        accepted_batches: list[np.ndarray] = []
        accepted_total = 0

        while accepted_total < n_samples:
            batch_size = max(1000, 4 * (n_samples - accepted_total))
            draws = rng.multivariate_normal(self.mean, self.covariance, size=batch_size)
            accepted = draws[self._inside_support(draws)]
            if accepted.size == 0:
                continue
            accepted_batches.append(accepted)
            accepted_total += accepted.shape[0]

        sample = np.vstack(accepted_batches)[:n_samples]
        return pd.DataFrame(sample, columns=["x1", "x2"])

    def compute_density(self, points: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = coerce_multivariate_data(points, column_names=["x1", "x2"]).to_numpy(
            dtype=float
        )
        density = np.zeros(arr.shape[0], dtype=float)
        inside = self._inside_support(arr)
        if np.any(inside):
            density[inside] = self.base_distribution.pdf(arr[inside]) / self.normalizer
        return density

    def density_on_mesh(self, mesh_x1: np.ndarray, mesh_x2: np.ndarray) -> np.ndarray:
        grid_points = np.column_stack([mesh_x1.ravel(), mesh_x2.ravel()])
        density = self.compute_density(grid_points)
        return density.reshape(mesh_x1.shape)

    def get_stats(self) -> dict[str, float]:
        return {
            "mean_x1": float(self.mean[0]),
            "mean_x2": float(self.mean[1]),
            "variance_x1": float(self.covariance[0, 0]),
            "variance_x2": float(self.covariance[1, 1]),
            "covariance_x1_x2": float(self.covariance[0, 1]),
            "lower_x1": float(self.lower[0]),
            "lower_x2": float(self.lower[1]),
            "upper_x1": float(self.upper[0]),
            "upper_x2": float(self.upper[1]),
            "truncation_probability": float(self.normalizer),
        }

    def __repr__(self) -> str:
        return (
            "BivariateTruncatedNormal("
            f"mean={tuple(self.mean)}, covariance={self.covariance.tolist()}, "
            f"lower={tuple(self.lower)}, upper={tuple(self.upper)})"
        )

