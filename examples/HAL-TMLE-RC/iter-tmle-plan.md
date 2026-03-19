# Iterative Pointwise TMLE Plan

This note records the agreed design for an iterative pointwise TMLE for right-censored survival.

The immediate goal is to prototype the workflow in `examples/HAL-TMLE-RC/iter-hal-tmle.ipynb`.
If the notebook behavior looks stable, we can then promote the same logic into
`src/haldensity/targeting/right_censored_survival/learner.py`.

## Goal

For each fixed target time `t0`, start from the Stage 1 IPCW HAL fit and apply repeated local TMLE updates until the empirical EIF mean is small enough.

The statistical target does not change:

- `Psi_t0(F) = S_F(t0)`

The key refinement is computational rather than conceptual:

- after the first tilt, the current fit is no longer "just re-evaluate the Stage 1 estimator";
- the outer loop must carry the updated pointwise fit forward on a fixed augmented grid.

## Core Design Change

For each fixed `t0`, define one augmented grid

- `G(t0) = G_stage1 union {t0}`

and keep it fixed across all local TMLE updates for that target.

On that fixed grid, carry a mutable pointwise state object. The state should contain:

- `t0`
- `target_grid_edges`
- `target_grid_midpoints`
- `target_delta_j`
- `log_density_grid`
- `density_grid`
- `survival_grid`
- `edge_survival`
- `psi_current`
- `raw_direction`
- `centered_direction`
- `eif_values`
- `eif_mean`
- `eif_sigma`
- `score_at_zero`
- `shared_censoring_cache`
- `target_grid_augmented_with_t0`

This state is initialized once from the Stage 1 estimator and then updated iteratively.

## Outer Stopping Rule

The outer loop should stop based on the empirical EIF mean of the current fit, not only the inner score equation.

At iteration `k`, compute:

- `D_i^{*,(k)}(t0)` = current observed-data EIF value
- `mean_eif_k = n^{-1} sum_i D_i^{*,(k)}(t0)`
- `sigma_k` = sample standard deviation of the current EIF values

Use the practical stopping rule

- `tol_k = max(min_score_tol, sigma_k / (sqrt(n) log n))`

and stop when

- `|mean_eif_k| <= tol_k`

Recommended defaults:

- `max_iter = 25`
- `min_abs_eps = 1e-8`
- `min_score_tol = 1e-8`

## Inner Local Update

At the current state:

1. Recompute the full-data gradient using the current survival:
   - `D_F^Full(t; t0) = I(t > t0) - S^{(k)}(t0)`
2. Recompute the right-censoring fluctuation direction `D_{F^(k), G_n}(t; t0)`.
3. Center that direction under the current density.
4. Compute the current EIF values and the current EIF mean.
5. Solve the one-dimensional local targeting problem for `epsilon_k`.
6. Update the current fit through the exponential tilt:
   - `log f^{(k+1)}(t) = log f^{(k)}(t) + epsilon_k D_c^{(k)}(t; t0) - log C^{(k)}(epsilon_k; t0)`
7. Recompute density, log-density, survival, and edge survival on the same fixed grid.

Important:

- recompute the least favorable direction at every iteration;
- do not freeze the iteration-0 direction and optimize repeatedly on that frozen path.

## Solver Contract

The inner one-dimensional solver should return one of three outcomes:

1. `score_root`
   - a bracketed score root is found and solved
2. `bounded_optimum`
   - no score root is found, but bounded optimization returns an objective maximizer
3. `solver_failure`
   - neither route gives a trustworthy update

Outer-loop acceptance policy:

- if `score_root`, accept the update;
- if `bounded_optimum`, accept only when:
  - the observed-data objective improves enough, and
  - `|score_at_solution|` is meaningfully smaller than `|score_at_zero|`;
- otherwise stop with `stop_reason = "solver_failure"`.

This avoids taking weak fallback steps indefinitely.

## Function Split for `learner.py`

The current one-step `_target_survival_t0(...)` should be refactored into three helpers.

### 1. `_initialize_pointwise_state(...)`

Purpose:

- build the augmented grid for one `t0`;
- evaluate the Stage 1 fit on that grid;
- compute initial density, log-density, survival, and `psi_init`;
- attach the shared censoring cache.

This is the only place where the pointwise state should depend on the Stage 1 estimator.

### 2. `_one_local_tmle_update(...)`

Purpose:

- take a current pointwise state;
- recompute the local direction;
- solve for one `epsilon_k`;
- update the density and survival on the same grid;
- return the updated state and one iteration record.

This helper should not implement the outer stopping loop.

### 3. `_iterate_tmle_t0(...)`

Purpose:

- evaluate the current EIF mean;
- check the outer stopping rule;
- call `_one_local_tmle_update(...)` when another step is needed;
- stop cleanly on `score_tolerance`, `max_iter`, `epsilon_tiny`, or `solver_failure`.

## Suggested Public API Changes

Keep the current public behavior as default and extend it with:

- `iterative=False`
- `max_iter=25`
- `min_abs_eps=1e-8`
- `min_score_tol=1e-8`
- `adaptive_gate=True`
- `one_step_eif_gate=1e-8`

Behavior:

- `iterative=False`: preserve the current one-step pointwise TMLE
- `iterative=True`: run the outer loop and return iteration history per target
- `adaptive_gate=True`: run one-step first and skip iterative updates when
  `abs(eif_mean_one_step) <= one_step_eif_gate`

## Adaptive Decision Rule (Notebook-First)

Given the empirical pattern that central target points are usually well-behaved while boundary points are harder, add a per-`t0` decision stage:

1. Run one-step TMLE.
2. Compute `abs(eif_mean_one_step)`.
3. If `abs(eif_mean_one_step) <= one_step_eif_gate` (default `1e-8`), do not iterate.
4. Otherwise, run iterative TMLE with the standard outer stopping rule.

This is a pragmatic compute-saving policy: it avoids unnecessary iterative updates when one-step targeting is already numerically solved at machine precision.

## History to Record Per Target

For each `t0`, record:

- `n_iterations`
- `stop_reason`
- `epsilon_path`
- `eif_mean_path`
- `sigma_path`
- `psi_path`

and, when detailed arrays are requested, store per-iteration records with:

- `iteration`
- `psi`
- `epsilon`
- `score_at_zero`
- `score_at_solution`
- `objective_at_solution`
- `eif_mean`
- `sigma`
- `solve_method`
- `converged_inner`
- `stop_tolerance`

Convention:

- define `n_iterations` as the number of accepted local epsilon updates
- if the initial state already satisfies the stopping rule, then `n_iterations = 0`

## What Stays Fixed vs Recomputed

Fixed for one target `t0`:

- observed data
- shared censoring cache
- target time `t0`
- augmented pointwise grid

Recomputed at each iteration:

- current density on the pointwise grid
- current log-density on the pointwise grid
- current survival on the pointwise grid
- current raw direction
- current centered direction
- current EIF values
- current EIF mean and sigma
- current score at zero
- local objective and score functions

## Notebook-First Implementation Path

Before changing the public API, validate the design in the notebook demo:

- keep the iterative helper layer local to `examples/HAL-TMLE-RC/iter-hal-tmle.ipynb`
- compare one-step and iterative targeting on one hard upper-tail target
- then compare them over a small target-time grid
- inspect which target points need more than one update

This lets us confirm the state-based design before we commit to a package API.

## Minimum Tests After API Promotion

Once the notebook behavior is satisfactory and we move the logic into `learner.py`, add at least these tests:

1. `iterative=False` reproduces the current one-step behavior.
2. If the initial EIF mean already satisfies the stopping rule, no update is taken.
3. If one local update is enough, iterative and one-step results agree.
4. `solver_failure` stops cleanly and still returns finite output.
5. A hard tail target reduces the residual EIF mean more in iterative mode than in one-step mode.

## Expected Pattern

The expected empirical pattern is:

- central target times often stop after zero or one accepted update;
- upper-tail target times can need multiple updates;
- weaker initializers should need more iterative correction;
- iterative TMLE mainly matters when the one-step solve leaves a visible tail residual.
Yes — that is the right next refinement.

If the pointwise one-step targeting solve is not driving the score close enough to zero in the tail, then the natural fix is an iterative local least favorable TMLE:
• keep the current pointwise targeting machinery,
• after one targeting update, treat the updated fit as the new baseline,
• recompute the full-data direction and observed-data score at that updated fit,
• solve for a new \epsilon,
• repeat until the empirical EIF mean is small enough.

That is exactly the right way to make the targeting stronger without changing the statistical target.

Here is a concrete implementation plan.

⸻

Iterative pointwise TMLE plan

We work at one fixed target time t_0.

Let the current event-time fit at iteration k be F_n^{(k)}, with density f_n^{(k)}, survival S_n^{(k)}, and shared censoring fit G_n.

We start from the Stage 1 initializer:
F_n^{(0)} = F_n^{\mathrm{init}}.

The target parameter is
\Psi\_{t_0}(F)=S_F(t_0).

⸻

1. Stopping criterion

At iteration k, compute the observed-data EIF values
\hat D\_{i}^{\*,(k)}(t_0)

A*{F_n^{(k)}}\!\left(D*{F_n^{(k)},G_n}(\cdot;t_0)\right)(O_i),
or equivalently with the centered direction.

Let
\bar D*n^{(k)}(t_0)=\frac{1}{n}\sum*{i=1}^n \hat D_i^{\*,(k)}(t_0),
and
\hat \sigma_k^2(t_0)

\frac{1}{n}\sum\_{i=1}^n
\left(\hat D_i^{\*,(k)}(t_0)-\bar D_n^{(k)}(t_0)\right)^2.

Stop when
\left|\bar D_n^{(k)}(t_0)\right|
\le
\frac{\hat \sigma_k(t_0)}{\sqrt n \log n}.

This is a sensible practical criterion and matches what you described.

I would also impose:
• max_iter, say 25 or 50,
• min_abs_eps, say 10^{-8} or 10^{-10}, to stop if the update size becomes numerically negligible.

⸻

2. One iteration: recompute the local least favorable direction

At iteration k, compute the full-data gradient
D\_{F_n^{(k)}}^{\mathrm{Full}}(t;t_0)=I(t>t_0)-S_n^{(k)}(t_0).

Then compute the new full-data fluctuation direction
D\_{F_n^{(k)},G_n}(t;t_0)

I^{-1}{F_n^{(k)},G_n}D{F_n^{(k)}}^{\mathrm{Full}}(t;t_0).

Then center it under the current fit:
D^{c,(k)}(t;t_0)

D\_{F_n^{(k)},G_n}(t;t_0)

E*{F_n^{(k)}}\!\left[D*{F_n^{(k)},G_n}(T;t_0)\right].

Important point: recompute the direction at every iteration.
Do not reuse the iteration-0 direction for all steps. This is what makes the procedure iterative TMLE rather than repeated optimization on a frozen path.

⸻

3. One iteration: build the local fluctuation submodel

Define the local exponential tilt from the current fit:
f\_{n,\epsilon}^{(k)}(t;t_0)

\frac{\exp\{\eta_n^{(k)}(t)+\epsilon D^{c,(k)}(t;t_0)\}}
{C_n^{(k)}(\epsilon;t_0)},
with
C_n^{(k)}(\epsilon;t_0)

\int_0^\tau
\exp\{\eta_n^{(k)}(u)+\epsilon D^{c,(k)}(u;t_0)\}\,du.

This is a local least favorable submodel around the current fit.

⸻

4. One iteration: solve the observed-data score equation

Define the observed-data log-likelihood along this local path:
\ell\_{t_0}^{(k)}(\epsilon)

\sum*{i=1}^n
\left[
\Delta_i \log f*{n,\epsilon}^{(k)}(\tilde T_i;t_0)

- (1-\Delta*i)\log S*{n,\epsilon}^{(k)}(\tilde T_i;t_0)
  \right].

Let the score be
U*{t_0}^{(k)}(\epsilon)=\frac{d}{d\epsilon}\ell*{t_0}^{(k)}(\epsilon).

Solve for
\hat\epsilon_k(t_0)

\arg\max*\epsilon \ell*{t_0}^{(k)}(\epsilon),
preferably by score root-finding first, bounded fallback second, exactly as in the current code.

⸻

5. One iteration: update the current fit

Set
\eta_n^{(k+1)}(t)

\eta_n^{(k)}(t)

- \hat\epsilon_k(t_0)\,D^{c,(k)}(t;t_0)

\log C_n^{(k)}(\hat\epsilon_k(t_0);t_0).

Equivalently,
f*n^{(k+1)}(t;t_0)=f*{n,\hat\epsilon_k}^{(k)}(t;t_0).

Then recompute:
• density on the pointwise augmented grid,
• survival on that grid,
• score at \epsilon=0 for the next iteration,
• EIF values,
• stopping criterion.

⸻

6. What should be cached and what should be recomputed

For each fixed t_0:

Shared across all iterations
• observed data,
• shared KM object / censoring cache,
• pointwise augmented grid for that t_0,
• target time t_0.

Recomputed every iteration
• current density on the pointwise grid,
• current survival on the pointwise grid,
• current D\_{F_n^{(k)},G_n},
• centered direction,
• score-at-zero,
• objective/score functions,
• EIF values,
• variance estimate,
• stopping criterion.

The pointwise augmented grid should not change across iterations for a fixed t_0.
Build it once for that target and reuse it.

⸻

7. What to record for each target time

You said you want to know how many iterations it takes for each point. I would explicitly store:

For each t*0:
• n_iterations
• converged
• stop_reason
• "score_tolerance"
• "max_iter"
• "epsilon_tiny"
• "solver_failure"
• score_path
\left\{\bar D_n^{(k)}(t_0)\right\}*{k=0}^{K}
• sigma*path
\left\{\hat\sigma_k(t_0)\right\}*{k=0}^{K}
• epsilon*path
\left\{\hat\epsilon_k(t_0)\right\}*{k=0}^{K-1}
• psi*path
\left\{S_n^{(k)}(t_0)\right\}*{k=0}^{K}

That gives you exactly what you need to study:
• which target points are hard,
• how many updates are needed in the tail,
• whether the score decreases monotonically,
• whether the updates become tiny before the score criterion is met.

For summary tables, I would report per t_0:
• final n_iterations,
• final score_at_solution,
• final eif_mean,
• final standard_error,
• final psi_star.

⸻

8. API / function structure

I would add an iterative version of the current pointwise targeting routine rather than rewriting everything.

Something like:

Core pointwise iterative routine

target_survival_t0_iterative(...)

Inputs:
• current Stage 1 fit
• observed data
• target time t_0
• shared censoring cache
• max_iter
• score_tol_rule="sigma_over_sqrt_n_log_n"
• optimization settings

Outputs:
• final pointwise targeted fit
• iteration history

Curve wrapper

target_survival_curve_iterative(...)

Loops over t_1,\dots,t_K and returns one result per point.

This should reuse your existing scalar targeting code as the inner update.

⸻

9. Suggested pseudocode

Here is the exact logical flow.

For each t*0: 1. Build the augmented pointwise grid. 2. Initialize F_n^{(0)} from the Stage 1 estimator on that grid. 3. For k=0,1,\dots:
• compute D*{F*n^{(k)}}^{\mathrm{Full}}(\cdot;t_0),
• compute D*{F_n^{(k)},G_n}(\cdot;t_0),
• center it,
• compute EIF values \hat D_i^{\*,(k)}(t_0),
• compute
\bar D_n^{(k)}(t_0),\qquad \hat\sigma_k(t_0),
• if
|\bar D_n^{(k)}(t_0)|\le \hat\sigma_k(t_0)/(\sqrt n \log n),
stop,
• otherwise solve the local 1D targeting problem for \hat\epsilon_k(t_0),
• update the current density and survival,
• continue. 4. Return final fit and iteration history.

⸻

10. What to compare after implementation

Once this is implemented, I would compare:

By target time
• number of iterations needed,
• whether tail points need more iterations,
• final score residual,
• final SE,
• plug-in vs one-update vs iterative TMLE.

Across methods

For each initializer m=1,2,3,4, compare:
• average iterations per target point,
• fraction of targets converged by one update,
• fraction needing >1 iteration,
• final RMSE / bias after iterative TMLE.

This will answer whether the tail issue is:
• purely a one-step targeting issue,
• or partly an initializer issue too.

⸻

11. Expected pattern

My expectation is:
• central target points will often converge in 1 update,
• tail points will need more iterations,
• weaker initializers will need more iterations,
• iterative TMLE will mostly matter in the upper tail.

That would be a very interpretable result.

⸻

12. Practical recommendation

I would implement this as:
• keep the current one-update pointwise TMLE as the default building block,
• add an iterative=True option,
• store a per-t_0 iteration history,
• add a summary table with n_iterations and stop_reason,
• and compare one-update vs iterative TMLE in your experiment notebook.

If you want, I can next turn this into a more code-oriented implementation spec with function signatures and return fields.
