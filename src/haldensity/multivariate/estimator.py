from __future__ import annotations

import time
from typing import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import qmc

from ..utils.cvxpy_solver import solve_cvxpy_problem
from ..utils.multivariate_basis import (
    coerce_multivariate_data,
    create_multivariate_basis,
    summarize_basis_catalog,
    summarize_design_matrix,
)


def _validate_support(
    support: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    normalized_support = tuple((float(lower), float(upper)) for lower, upper in support)
    if len(normalized_support) == 0:
        raise ValueError("support must contain at least one dimension")
    if any(lower >= upper for lower, upper in normalized_support):
        raise ValueError("Each lower bound must be strictly less than upper bound")
    return normalized_support


def build_midpoint_quadrature(
    support: Sequence[tuple[float, float]],
    points_per_dim: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Build a product midpoint quadrature rule on a rectangular support."""
    if points_per_dim < 2:
        raise ValueError("points_per_dim must be at least 2")

    normalized_support = _validate_support(support)
    midpoint_axes: list[np.ndarray] = []
    widths: list[float] = []

    for lower, upper in normalized_support:
        edges = np.linspace(lower, upper, points_per_dim + 1)
        midpoint_axes.append(0.5 * (edges[1:] + edges[:-1]))
        widths.append((upper - lower) / points_per_dim)

    meshes = np.meshgrid(*midpoint_axes, indexing="ij")
    points = np.column_stack([mesh.ravel() for mesh in meshes])
    weights = np.full(points.shape[0], np.prod(widths), dtype=float)
    return points, weights, tuple(midpoint_axes)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def default_sobol_budget(dimension: int) -> int:
    """Return a default Sobol budget of 128 * 2^dimension."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return 1 << (dimension + 7)


K0_AUTO_SOBOL_MULTIPLIER = 8


def build_sobol_normalizer_bank(
    support: Sequence[tuple[float, float]],
    budget: int,
    scramble: bool = True,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a Sobol QMC bank over a rectangular support."""
    if budget <= 0:
        raise ValueError("budget must be positive")

    normalized_support = _validate_support(support)
    lower = np.array([bound[0] for bound in normalized_support], dtype=float)
    upper = np.array([bound[1] for bound in normalized_support], dtype=float)
    volume = float(np.prod(upper - lower))

    sobol_engine = qmc.Sobol(
        d=len(normalized_support),
        scramble=scramble,
        seed=seed,
    )
    if _is_power_of_two(budget):
        unit_points = sobol_engine.random_base2(m=int(np.log2(budget)))
    else:
        unit_points = sobol_engine.random(n=budget)

    points = qmc.scale(unit_points, lower, upper)
    weights = np.full(points.shape[0], volume / points.shape[0], dtype=float)
    return points, weights


def _exact_k0_axis_structure(
    support: Sequence[tuple[float, float]],
    knot_data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    normalized_support = _validate_support(support)
    knot_df = coerce_multivariate_data(knot_data)
    if knot_df.shape[1] != len(normalized_support):
        raise ValueError(
            "support dimension must match the number of columns in knot_data"
        )

    axis_midpoints: list[np.ndarray] = []
    axis_widths: list[np.ndarray] = []
    for axis_idx, (lower, upper) in enumerate(normalized_support):
        knots = np.unique(knot_df.iloc[:, axis_idx].to_numpy(dtype=float))
        interior_knots = knots[(knots > lower) & (knots < upper)]
        edges = np.concatenate(([lower], interior_knots, [upper]))
        widths = np.diff(edges)
        if np.any(widths <= 0.0):
            raise ValueError("Exact k=0 cell structure has non-positive widths")

        axis_midpoints.append(0.5 * (edges[1:] + edges[:-1]))
        axis_widths.append(widths)

    return tuple(axis_midpoints), tuple(axis_widths)


def exact_k0_cell_budget(
    support: Sequence[tuple[float, float]],
    knot_data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> int:
    """Return the number of exact rectangular cells for k=0."""
    axis_midpoints, _ = _exact_k0_axis_structure(support, knot_data)
    return int(np.prod([len(axis) for axis in axis_midpoints], dtype=np.int64))


def build_exact_k0_normalizer_bank(
    support: Sequence[tuple[float, float]],
    knot_data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Build the exact knot-induced rectangular cell bank for k=0."""
    axis_midpoints, axis_widths = _exact_k0_axis_structure(support, knot_data)

    midpoint_meshes = np.meshgrid(*axis_midpoints, indexing="ij")
    width_meshes = np.meshgrid(*axis_widths, indexing="ij")
    points = np.column_stack([mesh.ravel() for mesh in midpoint_meshes])
    weights = np.prod(np.stack(width_meshes, axis=0), axis=0).ravel().astype(float)
    return points, weights, axis_midpoints


class MultvarHAL:
    """Finite initialized multivariate HAL density estimator."""

    def __init__(
        self,
        k: int,
        norm_constraint: float,
        support: Sequence[tuple[float, float]],
        quadrature_points_per_dim: int = 28,
        solver: str = "MOSEK",
        selection_tol: float = 1e-4,
        solver_kwargs: dict[str, float] | None = None,
        use_secondary_solver: bool = True,
        solver_waterfall: Sequence[str] = ("MOSEK", "CLARABEL", "ECOS", "SCS"),
        normalizer: str = "auto",
        sobol_budget: int | None = None,
        sobol_seed: int = 0,
        sobol_scramble: bool = True,
    ) -> None:
        if int(k) != k or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if norm_constraint <= 0:
            raise ValueError("norm_constraint must be positive")
        if normalizer.lower() not in {"auto", "exact", "midpoint", "sobol"}:
            raise ValueError(
                "normalizer must be one of 'auto', 'exact', 'midpoint', or 'sobol'"
            )
        if sobol_budget is not None and sobol_budget <= 0:
            raise ValueError("sobol_budget must be positive when provided")

        self.k = int(k)
        self.norm_constraint = float(norm_constraint)
        self.support = _validate_support(support)
        self.quadrature_points_per_dim = int(quadrature_points_per_dim)
        self.solver = solver.upper()
        self.selection_tol = float(selection_tol)
        self.solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
        self.use_secondary_solver = bool(use_secondary_solver)
        self.solver_waterfall = tuple(solver_name.upper() for solver_name in solver_waterfall)
        self.normalizer = normalizer.lower()
        self.sobol_budget = None if sobol_budget is None else int(sobol_budget)
        self.sobol_seed = int(sobol_seed)
        self.sobol_scramble = bool(sobol_scramble)
        self.is_fitted_ = False

    def _solve_kwargs_for_solver(self, solver_name: str) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "solver": solver_name,
            "verbose": False,
        }
        if solver_name == "SCS":
            kwargs.update({"eps": 5e-7, "max_iters": 100000})
        return kwargs

    def _solve_problem(self, problem: cp.Problem) -> None:
        def _build_solve_kwargs(solver_name: str) -> dict[str, object]:
            solve_kwargs = self._solve_kwargs_for_solver(solver_name)
            solve_kwargs.pop("solver", None)
            if solver_name == self.solver:
                solve_kwargs.update(self.solver_kwargs)
            return solve_kwargs

        solve_result = solve_cvxpy_problem(
            problem=problem,
            primary_solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            solver_waterfall=self.solver_waterfall,
            build_solve_kwargs=_build_solve_kwargs,
        )
        self.solver_used_ = solve_result.solver_used

    def _resolve_sobol_budget(self, dimension: int) -> int:
        if self.sobol_budget is not None:
            return self.sobol_budget
        return default_sobol_budget(dimension)

    def _resolve_k0_auto_sobol_budget(self, dimension: int) -> int:
        return K0_AUTO_SOBOL_MULTIPLIER * self._resolve_sobol_budget(dimension)

    def _select_normalizer_method(self) -> tuple[str, str]:
        dimension = self.training_data_.shape[1]
        self.exact_candidate_budget_ = None
        if self.k == 0:
            self.exact_candidate_budget_ = exact_k0_cell_budget(
                self.support,
                self.training_data_,
            )

        sobol_budget = self._resolve_sobol_budget(dimension)

        if self.normalizer == "auto":
            if self.k == 0:
                if dimension == 1:
                    return "exact", "auto: k=0 and d=1 uses exact cell normalizer"
                k0_auto_sobol_budget = self._resolve_k0_auto_sobol_budget(dimension)
                if (
                    self.exact_candidate_budget_ is not None
                    and self.exact_candidate_budget_ <= k0_auto_sobol_budget
                ):
                    return (
                        "exact",
                        "auto: exact k=0 budget "
                        f"{self.exact_candidate_budget_} <= 8x sobol budget {k0_auto_sobol_budget}",
                    )
                return (
                    "sobol",
                    "auto: exact k=0 budget "
                    f"{self.exact_candidate_budget_} > 8x sobol budget {k0_auto_sobol_budget}",
                )

            if dimension == 1:
                return "midpoint", "auto: k>=1 and d=1 uses midpoint normalizer"
            return (
                "sobol",
                f"auto: k>=1 and d>1 uses sobol budget {sobol_budget}",
            )

        if self.normalizer == "exact" and self.k != 0:
            raise ValueError("normalizer='exact' is only available for k=0")
        if self.normalizer == "midpoint":
            return "midpoint", "explicit midpoint normalizer requested"
        if self.normalizer == "sobol":
            return (
                "sobol",
                f"explicit sobol normalizer requested with budget {sobol_budget}",
            )
        return "exact", "explicit exact normalizer requested"

    def _build_normalizer_bank(self) -> None:
        method, reason = self._select_normalizer_method()
        self.normalizer_method_used_ = method
        self.normalizer_auto_reason_ = reason
        self.sobol_budget_used_ = None
        self.midpoint_points_per_dim_used_ = None

        if method == "midpoint":
            points, weights, axes = build_midpoint_quadrature(
                self.support,
                self.quadrature_points_per_dim,
            )
            self.midpoint_points_per_dim_used_ = self.quadrature_points_per_dim
        elif method == "sobol":
            if self.normalizer == "auto" and self.k == 0:
                resolved_budget = self._resolve_k0_auto_sobol_budget(self.training_data_.shape[1])
            else:
                resolved_budget = self._resolve_sobol_budget(self.training_data_.shape[1])
            points, weights = build_sobol_normalizer_bank(
                self.support,
                budget=resolved_budget,
                scramble=self.sobol_scramble,
                seed=self.sobol_seed,
            )
            axes = None
            self.sobol_budget_used_ = int(points.shape[0])
        else:
            points, weights, axes = build_exact_k0_normalizer_bank(
                self.support,
                self.training_data_,
            )

        self.normalizer_points_ = np.asarray(points, dtype=float)
        self.normalizer_weights_ = np.asarray(weights, dtype=float)
        self.normalizer_axes_ = axes
        self.normalizer_size_ = int(self.normalizer_points_.shape[0])
        self.normalizer_basis_, _, _ = create_multivariate_basis(
            self.normalizer_points_,
            k=self.k,
            include_intercept=True,
            knot_data=self.training_data_,
        )

        # Backward-compatible aliases for earlier notebook code.
        self.quadrature_points_ = self.normalizer_points_
        self.quadrature_weights_ = self.normalizer_weights_
        self.quadrature_axes_ = self.normalizer_axes_
        self.quadrature_basis_ = self.normalizer_basis_
        self.quadrature_grid_size_ = self.normalizer_size_

    def fit(
        self,
        data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    ) -> "MultvarHAL":
        fit_start = time.perf_counter()
        self.training_data_ = coerce_multivariate_data(data)
        if self.training_data_.shape[1] != len(self.support):
            raise ValueError(
                "support dimension must match the number of columns in data"
            )
        self.train_basis_, self.basis_names_, self.basis_meta_ = create_multivariate_basis(
            self.training_data_,
            k=self.k,
            include_intercept=True,
        )
        self._build_normalizer_bank()

        theta = cp.Variable(self.train_basis_.shape[1])
        log_density_train = self.train_basis_ @ theta
        log_density_quadrature = self.normalizer_basis_ @ theta
        log_weights = np.log(self.normalizer_weights_)

        objective = cp.Minimize(
            -cp.sum(log_density_train)
            + self.training_data_.shape[0]
            * cp.log_sum_exp(log_weights + log_density_quadrature)
        )
        constraints = [cp.norm1(theta[1:]) <= self.norm_constraint]
        problem = cp.Problem(objective, constraints)
        self._solve_problem(problem)

        if theta.value is None:
            raise RuntimeError(f"Solver failed with status {problem.status}")

        self.problem_status_ = str(problem.status)
        self.objective_value_ = float(problem.value)
        self.theta_raw_ = np.asarray(theta.value, dtype=float).reshape(-1)
        self.theta_ = self.theta_raw_.copy()
        self.theta_[1:][np.abs(self.theta_[1:]) < self.selection_tol] = 0.0
        self.norm_used_ = float(np.sum(np.abs(self.theta_[1:])))
        self.design_diagnostics_ = summarize_design_matrix(self.train_basis_)
        self.selected_basis_ = self._build_selected_basis_table()
        self.is_fitted_ = True
        self.selected_basis_summary_ = self.selected_basis_summary()
        self.train_loglik_ = float(
            np.sum(np.log(self.get_density_at_points(self.training_data_) + 1e-12))
        )
        self.train_midpoint_integral_ = self.normalization_integral()
        self.fit_time_seconds_ = time.perf_counter() - fit_start
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("Estimator must be fitted before calling this method")

    def _build_selected_basis_table(self) -> pd.DataFrame:
        selected_mask = np.abs(self.theta_) > self.selection_tol
        selected = self.basis_meta_.loc[selected_mask].copy()
        selected["coefficient"] = self.theta_[selected_mask]
        selected["abs_coefficient"] = np.abs(selected["coefficient"])
        selected = selected.sort_values(
            "abs_coefficient",
            ascending=False,
        ).reset_index(drop=True)
        return selected.loc[
            :,
            [
                "basis_index",
                "coefficient",
                "abs_coefficient",
                "family",
                "term_type",
                "section",
                "section_size",
                "complement",
                "knot_row",
                "exponents",
                "name",
            ],
        ]

    def selected_basis_summary(self) -> pd.DataFrame:
        self._require_fitted()
        return summarize_basis_catalog(self.selected_basis_, coefficient_col="coefficient")

    def _log_normalizer(self) -> float:
        self._require_fitted()
        return float(
            logsumexp(self.normalizer_basis_ @ self.theta_ + np.log(self.normalizer_weights_))
        )

    def get_density_at_points(
        self,
        points: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        self._require_fitted()
        eval_df = coerce_multivariate_data(points, column_names=self.training_data_.columns)
        arr = eval_df.to_numpy(dtype=float)
        density = np.zeros(arr.shape[0], dtype=float)
        lower = np.array([bound[0] for bound in self.support], dtype=float)
        upper = np.array([bound[1] for bound in self.support], dtype=float)
        inside = np.all((arr >= lower) & (arr <= upper), axis=1)

        if np.any(inside):
            basis_eval, _, _ = create_multivariate_basis(
                arr[inside],
                k=self.k,
                include_intercept=True,
                knot_data=self.training_data_,
            )
            log_density = basis_eval @ self.theta_ - self._log_normalizer()
            density[inside] = np.exp(np.clip(log_density, -700.0, 700.0))
        return density

    def get_density_on_mesh(
        self,
        points_per_dim: int = 80,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()
        if len(self.support) != 2:
            raise ValueError("get_density_on_mesh is only available for two-dimensional supports")

        axis_x1 = np.linspace(self.support[0][0], self.support[0][1], points_per_dim)
        axis_x2 = np.linspace(self.support[1][0], self.support[1][1], points_per_dim)
        mesh_x1, mesh_x2 = np.meshgrid(axis_x1, axis_x2, indexing="xy")
        grid_points = np.column_stack([mesh_x1.ravel(), mesh_x2.ravel()])
        density = self.get_density_at_points(grid_points).reshape(mesh_x1.shape)
        return mesh_x1, mesh_x2, density

    def normalization_integral(self, points_per_dim: int | None = None) -> float:
        self._require_fitted()
        if points_per_dim is None:
            density = np.exp(
                np.clip(self.normalizer_basis_ @ self.theta_ - self._log_normalizer(), -700.0, 700.0)
            )
            return float(np.sum(self.normalizer_weights_ * density))

        points, weights, _ = build_midpoint_quadrature(self.support, points_per_dim)
        density = self.get_density_at_points(points)
        return float(np.sum(weights * density))

    def summary(self) -> dict[str, float | int | str]:
        self._require_fitted()
        nonintercept_selected = int(np.sum(np.abs(self.theta_[1:]) > self.selection_tol))
        return {
            "k": self.k,
            "solver_requested": self.solver,
            "solver_used": self.solver_used_,
            "solver_status": self.problem_status_,
            "normalizer_requested": self.normalizer,
            "normalizer_method_used": self.normalizer_method_used_,
            "normalizer_size": self.normalizer_size_,
            "normalizer_auto_reason": self.normalizer_auto_reason_,
            "exact_candidate_budget": self.exact_candidate_budget_,
            "sobol_budget_used": self.sobol_budget_used_,
            "midpoint_points_per_dim_used": self.midpoint_points_per_dim_used_,
            "basis_width": int(self.train_basis_.shape[1]),
            "quadrature_grid_size": self.quadrature_grid_size_,
            "fit_time_seconds": self.fit_time_seconds_,
            "selected_basis_count": int(self.selected_basis_.shape[0]),
            "selected_nonintercept": nonintercept_selected,
            "norm_constraint_M": self.norm_constraint,
            "norm_used": self.norm_used_,
            "objective_value": self.objective_value_,
            "train_loglik": self.train_loglik_,
            "near_zero_variance_fraction": self.design_diagnostics_[
                "near_zero_variance_fraction"
            ],
            "max_abs_correlation_subset": self.design_diagnostics_[
                "max_abs_correlation_subset"
            ],
        }

    def get_results(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "summary": self.summary(),
            "theta_hat": self.theta_.copy(),
            "theta_raw": self.theta_raw_.copy(),
            "basis_names": list(self.basis_names_),
            "basis_metadata": self.basis_meta_.copy(),
            "selected_basis": self.selected_basis_.copy(),
            "selected_basis_summary": self.selected_basis_summary_.copy(),
            "support": self.support,
            "normalizer_method": self.normalizer_method_used_,
            "normalizer_size": self.normalizer_size_,
            "normalizer_axes": None
            if self.normalizer_axes_ is None
            else tuple(axis.copy() for axis in self.normalizer_axes_),
            "quadrature_axes": None
            if self.quadrature_axes_ is None
            else tuple(axis.copy() for axis in self.quadrature_axes_),
        }

