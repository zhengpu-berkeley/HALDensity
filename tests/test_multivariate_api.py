import numpy as np
import pytest

from haldensity.multivariate import (
    BivariateTruncatedNormal,
    MultvarHAL,
    create_multivariate_basis,
    default_sobol_budget,
    expected_section_counts,
    exact_k0_cell_budget,
    summarize_basis_catalog,
    theoretical_basis_count,
)


def test_create_multivariate_basis_matches_theoretical_count():
    data = np.array(
        [
            [0.10, 0.20],
            [0.35, 0.65],
            [0.80, 0.40],
        ]
    )

    basis, names, metadata = create_multivariate_basis(data, k=1)

    assert basis.shape == (3, theoretical_basis_count(n=3, d=2, k=1))
    assert names[0] == "Intercept"

    summary = summarize_basis_catalog(metadata).set_index("term_type")["count"]
    expected_by_section = expected_section_counts(n=3, d=2, k=1)

    assert int(summary["poly"]) == 4
    assert int(summary["section-1"]) == int(expected_by_section.loc[1])
    assert int(summary["full"]) == int(expected_by_section.loc[2])


def test_multvarhal_smoke_fit_has_near_normalized_density():
    dgp = BivariateTruncatedNormal(
        mean=(0.45, 0.60),
        covariance=[[0.015, 0.006], [0.006, 0.020]],
    )
    data = dgp.generate_samples(24, seed=11)

    model = MultvarHAL(
        k=1,
        norm_constraint=25.0,
        support=dgp.bounds,
        quadrature_points_per_dim=10,
        use_secondary_solver=True,
    ).fit(data)

    summary = model.summary()
    fine_integral = model.normalization_integral(points_per_dim=18)

    assert summary["solver_requested"] == "MOSEK"
    assert summary["solver_used"] in {"MOSEK", "ECOS", "CLARABEL", "SCS"}
    assert summary["solver_status"] in {"optimal", "optimal_inaccurate"}
    assert summary["normalizer_requested"] == "auto"
    assert summary["normalizer_method_used"] == "sobol"
    assert summary["sobol_budget_used"] == default_sobol_budget(2)
    assert summary["selected_basis_count"] >= 1
    assert np.isclose(fine_integral, 1.0, atol=0.05)


def test_multvarhal_auto_k0_prefers_exact_when_exact_budget_is_smaller():
    data = np.array(
        [
            [0.10, 0.20],
            [0.35, 0.65],
            [0.80, 0.40],
        ]
    )
    support = ((0.0, 1.0), (0.0, 1.0))

    model = MultvarHAL(
        k=0,
        norm_constraint=8.0,
        support=support,
        use_secondary_solver=True,
    ).fit(data)

    summary = model.summary()
    assert summary["normalizer_method_used"] == "exact"
    assert summary["exact_candidate_budget"] == exact_k0_cell_budget(support, data)
    assert summary["normalizer_size"] == summary["exact_candidate_budget"]
    assert summary["sobol_budget_used"] is None


def test_multvarhal_auto_k0_prefers_sobol_when_exact_budget_is_larger():
    data = np.array(
        [
            [0.10, 0.15],
            [0.20, 0.25],
            [0.30, 0.35],
            [0.40, 0.45],
            [0.50, 0.55],
            [0.60, 0.65],
            [0.70, 0.75],
            [0.80, 0.85],
            [0.90, 0.95],
        ]
    )

    model = MultvarHAL(
        k=0,
        norm_constraint=8.0,
        support=((0.0, 1.0), (0.0, 1.0)),
        sobol_budget=4,
        use_secondary_solver=True,
    ).fit(data)

    summary = model.summary()
    assert summary["exact_candidate_budget"] > 8 * 4
    assert summary["normalizer_method_used"] == "sobol"
    assert summary["sobol_budget_used"] == 8 * 4


def test_multvarhal_auto_k_ge_1_d1_prefers_midpoint():
    data = np.array([[0.10], [0.35], [0.55], [0.80], [0.90]])

    model = MultvarHAL(
        k=1,
        norm_constraint=12.0,
        support=((0.0, 1.0),),
        quadrature_points_per_dim=12,
        use_secondary_solver=True,
    ).fit(data)

    summary = model.summary()
    assert summary["normalizer_method_used"] == "midpoint"
    assert summary["midpoint_points_per_dim_used"] == 12


def test_multvarhal_exact_is_only_available_for_k0():
    with pytest.raises(ValueError, match="only available for k=0"):
        MultvarHAL(
            k=1,
            norm_constraint=8.0,
            support=((0.0, 1.0), (0.0, 1.0)),
            normalizer="exact",
            use_secondary_solver=True,
        ).fit(np.array([[0.2, 0.3], [0.6, 0.8]]))

