from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import Sequence

import numpy as np
import pandas as pd


def coerce_multivariate_data(
    data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    column_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Coerce input data into a numeric multivariate DataFrame."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if column_names is not None:
            requested_columns = [str(column) for column in column_names]
            current_columns = [str(column) for column in df.columns]
            if len(requested_columns) != df.shape[1]:
                raise ValueError("column_names must match the number of columns in data")
            if set(requested_columns) == set(current_columns):
                df = df.loc[:, requested_columns]
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("data must be a 2D array or DataFrame")
        if column_names is not None and len(column_names) != arr.shape[1]:
            raise ValueError("column_names must match the number of columns in data")
        columns = (
            list(column_names)
            if column_names is not None
            else [f"x{i + 1}" for i in range(arr.shape[1])]
        )
        df = pd.DataFrame(arr, columns=columns)

    if df.shape[1] == 0:
        raise ValueError("data must contain at least one column")
    if df.isnull().any().any():
        raise ValueError("data must not contain missing values")

    df = df.reset_index(drop=True).astype(float)
    if column_names is not None:
        if len(column_names) != df.shape[1]:
            raise ValueError("column_names must match the number of columns in data")
        df.columns = list(column_names)
    else:
        df.columns = [str(column) for column in df.columns]
    return df


def _format_power_term(var_name: str, power: int) -> str:
    if power == 1:
        return var_name
    return f"{var_name}^{power}"


def _format_polynomial_name(
    exponents: tuple[int, ...],
    variable_names: Sequence[str],
) -> str:
    factors = [
        _format_power_term(var_name, power)
        for var_name, power in zip(variable_names, exponents)
        if power > 0
    ]
    return "Intercept" if not factors else " * ".join(factors)


def _format_truncated_term(var_name: str, knot: float, k: int) -> str:
    if k == 0:
        return f"I({var_name} >= {knot:.3f})"
    if k == 1:
        return f"({var_name} - {knot:.3f})_+"
    return f"({var_name} - {knot:.3f})_+^{k}"


def _truncated_component(values: np.ndarray, knot: float, k: int) -> np.ndarray:
    if k == 0:
        return (values >= knot).astype(np.float32)
    return (np.maximum(values - knot, 0.0) ** k).astype(np.float32)


def _term_type_label(family: str, section_size: int, dimension: int) -> str:
    if family == "parametric":
        return "poly"
    if section_size == dimension:
        return "full"
    return f"section-{section_size}"


def theoretical_basis_count(
    n: int,
    d: int,
    k: int,
    include_intercept: bool = True,
) -> int:
    """Return the theoretical multivariate HAL basis size."""
    total = (k + 1) ** d + n * ((k + 2) ** d - (k + 1) ** d)
    if not include_intercept:
        total -= 1
    return total


def expected_section_counts(n: int, d: int, k: int) -> pd.Series:
    """Return expected counts by section size."""
    return pd.Series(
        {
            section_size: comb(d, section_size) * n * (k + 1) ** (d - section_size)
            for section_size in range(1, d + 1)
        },
        name="expected_count",
    )


def summarize_basis_catalog(
    metadata: pd.DataFrame,
    coefficient_col: str | None = None,
) -> pd.DataFrame:
    """Summarize basis terms by type and section size."""
    if metadata.empty:
        columns = ["term_type", "section_size", "count"]
        if coefficient_col is not None:
            columns.append("abs_coefficient_sum")
        return pd.DataFrame(columns=columns)

    summary = (
        metadata.groupby(["term_type", "section_size"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["section_size", "term_type"], kind="stable")
        .reset_index(drop=True)
    )

    if coefficient_col is not None and coefficient_col in metadata.columns:
        abs_summary = (
            metadata.assign(abs_coefficient=np.abs(metadata[coefficient_col]))
            .groupby(["term_type", "section_size"], dropna=False)["abs_coefficient"]
            .sum()
            .rename("abs_coefficient_sum")
            .reset_index()
        )
        summary = summary.merge(
            abs_summary,
            on=["term_type", "section_size"],
            how="left",
        )

    return summary


def extract_term_type_metrics(
    summary: pd.DataFrame,
    prefix: str = "",
    term_types: tuple[str, ...] = ("poly", "section-1", "full"),
) -> dict[str, float | int]:
    """Flatten term-type counts and coefficient mass into a metric dictionary."""
    metrics: dict[str, float | int] = {}
    prefix_text = f"{prefix}_" if prefix else ""

    for term_type in term_types:
        term_key = term_type.replace("-", "_")
        row = summary.loc[summary["term_type"].eq(term_type)]
        metrics[f"{prefix_text}{term_key}_count"] = (
            int(row["count"].iloc[0]) if not row.empty else 0
        )
        if "abs_coefficient_sum" in summary.columns:
            metrics[f"{prefix_text}{term_key}_abs_mass"] = (
                float(row["abs_coefficient_sum"].iloc[0]) if not row.empty else 0.0
            )
    return metrics


def summarize_design_matrix(
    basis_matrix: np.ndarray,
    correlation_subset_size: int = 20,
    near_zero_tol: float = 1e-10,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return lightweight conditioning diagnostics for a basis matrix."""
    matrix = np.asarray(basis_matrix, dtype=float)
    column_norms = np.linalg.norm(matrix, axis=0)
    column_variances = np.var(matrix, axis=0)

    summary: dict[str, float | int] = {
        "n_rows": int(matrix.shape[0]),
        "n_cols": int(matrix.shape[1]),
        "min_col_norm": float(np.min(column_norms)),
        "median_col_norm": float(np.median(column_norms)),
        "max_col_norm": float(np.max(column_norms)),
        "near_zero_variance_fraction": float(
            np.mean(column_variances <= near_zero_tol)
        ),
    }

    if matrix.shape[1] > 1:
        rng = np.random.default_rng(seed)
        subset_size = min(correlation_subset_size, matrix.shape[1])
        subset_idx = rng.choice(matrix.shape[1], size=subset_size, replace=False)
        subset = matrix[:, subset_idx]
        subset_std = np.std(subset, axis=0)
        nonconstant = subset_std > np.sqrt(near_zero_tol)
        reduced_subset = subset[:, nonconstant]

        if reduced_subset.shape[1] > 1:
            correlation_matrix = np.corrcoef(reduced_subset, rowvar=False)
            upper = np.abs(
                correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            )
            finite_upper = upper[np.isfinite(upper)]
            summary["correlation_subset_size"] = int(reduced_subset.shape[1])
            summary["max_abs_correlation_subset"] = (
                float(np.max(finite_upper)) if finite_upper.size else 0.0
            )
            summary["median_abs_correlation_subset"] = (
                float(np.median(finite_upper)) if finite_upper.size else 0.0
            )
        else:
            summary["correlation_subset_size"] = int(reduced_subset.shape[1])
            summary["max_abs_correlation_subset"] = 0.0
            summary["median_abs_correlation_subset"] = 0.0
    else:
        summary["correlation_subset_size"] = 1
        summary["max_abs_correlation_subset"] = 0.0
        summary["median_abs_correlation_subset"] = 0.0

    if min(matrix.shape) > 0 and matrix.shape[0] <= 200 and matrix.shape[1] <= 200:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        summary["max_singular_value"] = float(np.max(singular_values))
        summary["min_singular_value"] = float(np.min(singular_values))
    return summary


def create_multivariate_basis(
    data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    k: int,
    include_intercept: bool = True,
    knot_data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Build the finite initialized multivariate HAL basis."""
    if int(k) != k or k < 0:
        raise ValueError("k must be a nonnegative integer")
    k = int(k)

    eval_df = coerce_multivariate_data(data)
    knot_df = (
        eval_df.copy()
        if knot_data is None
        else coerce_multivariate_data(knot_data, column_names=eval_df.columns)
    )
    if knot_df.shape[1] != eval_df.shape[1]:
        raise ValueError("data and knot_data must have the same number of columns")

    knot_df.columns = list(eval_df.columns)
    x_eval = eval_df.to_numpy(dtype=float)
    x_knots = knot_df.to_numpy(dtype=float)
    n_eval, d = x_eval.shape
    variable_names = list(eval_df.columns)

    basis_columns: list[np.ndarray] = []
    basis_names: list[str] = []
    metadata_rows: list[dict[str, object]] = []

    def append_basis(
        column: np.ndarray,
        name: str,
        family: str,
        section: tuple[str, ...],
        complement: tuple[str, ...],
        exponents: tuple[int, ...],
        knot_row: int | None,
        knot_values: np.ndarray | None,
    ) -> None:
        section_size = len(section)
        basis_index = len(basis_names)
        basis_names.append(name)
        basis_columns.append(np.asarray(column, dtype=np.float32))
        metadata_rows.append(
            {
                "basis_index": basis_index,
                "family": family,
                "term_type": _term_type_label(family, section_size, d),
                "section": tuple(section),
                "section_size": section_size,
                "complement": tuple(complement),
                "exponents": tuple(int(power) for power in exponents),
                "knot_row": knot_row,
                "knot_values": None
                if knot_values is None
                else tuple(
                    float(knot_values[variable_names.index(var_name)])
                    for var_name in section
                ),
                "is_full_section": family == "sectional" and section_size == d,
                "name": name,
            }
        )

    for exponents in product(range(k + 1), repeat=d):
        if not include_intercept and all(power == 0 for power in exponents):
            continue

        column = np.ones(n_eval, dtype=np.float32)
        factors: list[str] = []
        for idx, power in enumerate(exponents):
            if power == 0:
                continue
            column *= (x_eval[:, idx] ** power).astype(np.float32)
            factors.append(_format_power_term(variable_names[idx], power))

        name = _format_polynomial_name(
            tuple(int(power) for power in exponents),
            variable_names,
        )
        append_basis(
            column=column,
            name=name,
            family="parametric",
            section=(),
            complement=tuple(variable_names),
            exponents=tuple(int(power) for power in exponents),
            knot_row=None,
            knot_values=None,
        )

    for section_size in range(1, d + 1):
        for section in combinations(range(d), section_size):
            complement = tuple(idx for idx in range(d) if idx not in section)
            exponent_grid = list(product(range(k + 1), repeat=len(complement)))

            for knot_row, knot_values in enumerate(x_knots):
                for exponents in exponent_grid:
                    column = np.ones(n_eval, dtype=np.float32)
                    factors: list[str] = []

                    for comp_idx, power in zip(complement, exponents):
                        if power == 0:
                            continue
                        column *= (x_eval[:, comp_idx] ** power).astype(np.float32)
                        factors.append(
                            _format_power_term(variable_names[comp_idx], power)
                        )

                    for sec_idx in section:
                        column *= _truncated_component(
                            x_eval[:, sec_idx],
                            knot_values[sec_idx],
                            k,
                        )
                        factors.append(
                            _format_truncated_term(
                                variable_names[sec_idx],
                                knot_values[sec_idx],
                                k,
                            )
                        )

                    append_basis(
                        column=column,
                        name=" * ".join(factors),
                        family="sectional",
                        section=tuple(variable_names[idx] for idx in section),
                        complement=tuple(variable_names[idx] for idx in complement),
                        exponents=tuple(int(power) for power in exponents),
                        knot_row=knot_row,
                        knot_values=knot_values,
                    )

    basis_matrix = np.column_stack(basis_columns).astype(np.float32)
    metadata = pd.DataFrame(metadata_rows)
    return basis_matrix, basis_names, metadata


def get_basis_index(
    metadata: pd.DataFrame,
    *,
    family: str,
    section: tuple[str, ...],
    exponents: tuple[int, ...],
    knot_row: int | None = None,
) -> int:
    """Find the unique index of a basis term in the metadata catalog."""
    mask = metadata["family"].eq(family)
    mask &= metadata["section"].map(lambda value: tuple(value) == tuple(section))
    mask &= metadata["exponents"].map(
        lambda value: tuple(value) == tuple(exponents)
    )

    if knot_row is None:
        mask &= metadata["knot_row"].isna()
    else:
        mask &= metadata["knot_row"].eq(knot_row)

    matches = metadata.index[mask]
    if len(matches) != 1:
        raise ValueError(f"Expected one basis match, found {len(matches)}")
    return int(matches[0])

