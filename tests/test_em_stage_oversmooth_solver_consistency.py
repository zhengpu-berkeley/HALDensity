import numpy as np
import pandas as pd

from haldensity.censoring.right.km import KaplanMeier
from haldensity.censoring.right.weights import compute_ipcw_weights
from haldensity.censoring.right.ipcw_estimator import RightCensoredIPCWEstimator
from haldensity.censoring.right.metrics import incomplete_loglik
from haldensity.censoring.tuners.em_stage_oversmooth_tuner import (
    RightCensoredEMStageOverSmoothTuner,
)


def _make_synthetic_right_censoring_data(n: int = 200, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_event = rng.uniform(0.0, 1.0, size=n)
    t_cens = rng.uniform(0.0, 1.0, size=n)
    t_obs = np.minimum(t_event, t_cens)
    delta = (t_event <= t_cens).astype(int)

    # Ensure both censored and uncensored exist (test needs IPCW weights + EM E-step).
    if delta.sum() == 0 or delta.sum() == n:
        # Force a mix by flipping a few labels deterministically.
        delta[: max(1, n // 10)] = 1
        delta[-max(1, n // 10) :] = 0

    return pd.DataFrame({"T": t_obs.astype(float), "Delta": delta.astype(int)})


def _fit_stage1_ipcw_refit(
    data: pd.DataFrame,
    *,
    norm_constraint: float,
    basis_order: int,
    n_grid_points: int,
    solver: str,
    use_secondary_solver: bool,
) -> RightCensoredIPCWEstimator:
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    T_vals = np.asarray(data["T"].values, dtype=float)
    Delta_vals = np.asarray(data["Delta"].values, dtype=int)
    w = compute_ipcw_weights(T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t)))

    unc_mask = Delta_vals == 1
    df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
    w_unc = w[unc_mask]

    est = RightCensoredIPCWEstimator(
        tol=1e-4,
        norm_constraint=float(norm_constraint),
        n_grid_points=int(n_grid_points),
        basis_order=int(basis_order),
        solver=str(solver),
        use_secondary_solver=bool(use_secondary_solver),
    )
    return est.fit(df_unc, sample_weights=w_unc)


def test_stage2_factor1_ipcw_matches_stage1_ipcw_and_em_solver_matches_ipcw_solver() -> None:
    """Regression test:

    - Stage-2 refits an IPCW initializer at factor=1.0. That refit should match an
      externally fitted IPCW estimator that uses the same (basis_order, lambda*) and
      solver configuration.

    - EM refinement should use the same solver as IPCW (by construction in the tuner).

    This test is intentionally lightweight and uses ECOS (always available in deps).
    """

    data = _make_synthetic_right_censoring_data(n=200, seed=123)

    solver = "ECOS"
    use_secondary_solver = True
    n_grid_points = 50
    basis_order = 0
    lambda_star = 6.0

    stage1_ipcw = _fit_stage1_ipcw_refit(
        data,
        norm_constraint=lambda_star,
        basis_order=basis_order,
        n_grid_points=n_grid_points,
        solver=solver,
        use_secondary_solver=use_secondary_solver,
    )

    stage2 = RightCensoredEMStageOverSmoothTuner(
        data,
        ipcw_params={"norm_constraint": lambda_star, "basis_order": basis_order},
        n_grid_points=n_grid_points,
        # Only baseline; tuner always includes factor=1.0.
        oversmooth_factors=[1.0],
        ipcw_solver=solver,
        ipcw_use_secondary_solver=use_secondary_solver,
        # Keep EM light; we only need it to run once to assert solver propagation.
        em_m_imputations=5,
        em_max_em_iter=1,
        em_tol=1e9,
        silent=True,
    )

    best_em = stage2.fit_best_estimator()

    assert stage2.ipcw_records is not None
    baseline_records = [r for r in stage2.ipcw_records if float(r.factor) == 1.0]
    assert len(baseline_records) == 1
    baseline = baseline_records[0]

    # Solver propagation: IPCW init solver and EM M-step solver should match.
    assert baseline.estimator.solver == solver
    assert best_em.solver == solver

    # Factor=1.0 IPCW init should match externally fitted IPCW refit.
    res1 = stage1_ipcw.get_results()
    res2 = baseline.estimator.get_results()

    knots1 = np.asarray(res1.get("grid_points_hal_selected", []), dtype=float)
    knots2 = np.asarray(res2.get("grid_points_hal_selected", []), dtype=float)

    assert knots1.shape == knots2.shape
    assert np.allclose(knots1, knots2, rtol=1e-6, atol=1e-10)

    ll1 = float(incomplete_loglik(stage1_ipcw, data, time_col="T", delta_col="Delta"))
    ll2 = float(incomplete_loglik(baseline.estimator, data, time_col="T", delta_col="Delta"))
    assert np.isfinite(ll1) and np.isfinite(ll2)
    assert np.isclose(ll1, ll2, rtol=1e-6, atol=1e-8)

    grid = np.linspace(0.0, 1.0, 101)
    pdf1 = np.asarray(stage1_ipcw.get_density_at_points(grid), dtype=float)
    pdf2 = np.asarray(baseline.estimator.get_density_at_points(grid), dtype=float)
    assert pdf1.shape == pdf2.shape
    assert np.allclose(pdf1, pdf2, rtol=1e-6, atol=1e-10)
