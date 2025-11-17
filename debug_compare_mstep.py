import sys
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, '.')
sys.path.insert(0, 'src')

from legacy_em_ipcw_hal import IPCW_HAL_init, E_step, M_step
from haldensity.censoring import EMIPCWEstimator
from haldensity.censoring.sampling import e_step_multiple_imputation
from haldensity.censoring.metrics import incomplete_loglik
from haldensity.censoring.km import KaplanMeier
from haldensity.censoring.weights import compute_ipcw_weights
from haldensity.censoring.legacy_m_step import solve_legacy_m_step, LegacyMStepResult

def generate_data(n=200, seed=12776):
    rng = np.random.default_rng(seed)
    a, b = (0 - 0.5)/0.1, (1 - 0.5)/0.1
    T = truncnorm.rvs(a, b, loc=0.5, scale=0.1, size=n, random_state=rng)
    C = rng.uniform(0, 1, size=n)
    T_tilde = np.minimum(T, C)
    delta = (T <= C).astype(int)
    legacy_df = pd.DataFrame({'T_tilde': T_tilde, 'delta': delta})
    new_df = pd.DataFrame({'T': T_tilde, 'Delta': delta})
    return legacy_df, new_df

legacy_df, new_df = generate_data()

print('Running legacy IPCW init...')
legacy_init = IPCW_HAL_init(legacy_df, norm_constraint=70.0)
print('Legacy init LL:', legacy_init['estimated_density'].sum())
legacy_e = E_step(legacy_df, legacy_init, num_samples=50)
legacy_m = M_step(legacy_e, legacy_init, legacy_init['theta_value'], norm_constraint=350.0)
print('Legacy M-step keys:', legacy_m.keys())
legacy_theta = legacy_m['theta_value']
legacy_knots = legacy_m['grid_points_hal_selected']
legacy_eval_grid = legacy_m['grid_eval']
legacy_estimator = LegacyMStepResult(
    legacy_theta,
    legacy_knots,
    legacy_eval_grid,
    density_values=legacy_m["estimated_density"],
    grid_midpoints=legacy_m["grid_midpoints"],
    delta_j=legacy_m["delta_j"],
)
print('Legacy theta len:', len(legacy_theta))

print('\nRunning new IPCW init...')
em = EMIPCWEstimator(norm_constraint=350.0, init_norm_constraint=70.0, m_step_norm_constraint=350.0,
                     m_imputations=50, max_em_iter=1, e_step_n_grid=1000, verbose=False,
                     init_solver='SCS', m_step_solver='ECOS', rng_seed=12776,
                     use_sc_adjustment=False)
init_est = em._init_ipcw(new_df)
km = em.km_
sc = km.predict
pooled = e_step_multiple_imputation(data=new_df, theta_hat=init_est.theta_hat.copy(),
                                    basis_grid_points=init_est._grid_points_hal.copy(),
                                    basis_order=0, S_c_predict=sc, m_imputations=50,
                                    n_grid=1000, use_sc_adjustment=True, rng=np.random.default_rng(12776))
legacy_pooled = legacy_e[['W1', 'weights']].copy()
legacy_pooled.rename(columns={'weights': 'weight'}, inplace=True)
new_solver_res = solve_legacy_m_step(
    pooled_df=legacy_pooled,
    knots=legacy_knots,
    norm_constraint=350.0,
    warm_start_theta=legacy_theta,
    tol=1e-4,
    solver_sequence=['ECOS', 'SCS'],
    n_eval_grid=200,
)
print('New theta len:', len(new_solver_res.theta_hat))

# Compare LLs
legacy_ll = incomplete_loglik(legacy_estimator, new_df, time_col='T', delta_col='Delta')
new_ll = incomplete_loglik(new_solver_res, new_df, time_col='T', delta_col='Delta')
print('Legacy LL:', legacy_ll)
print('New LL   :', new_ll)
