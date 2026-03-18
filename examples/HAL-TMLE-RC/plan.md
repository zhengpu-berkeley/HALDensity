Yes — here is the updated **full implementation plan** I would recommend now, incorporating all of our corrections:

- use the **existing IPCW-HAL-MLE** as the initial estimator,
- assume **(T \perp C)** and **no covariates**,
- do **pointwise targeting** for each fixed (t_0),
- use a **single targeting update** by default,
- build the tilt with the **full-data fluctuation direction** (D\_{F_n,G_n}),
- maximize the **observed-data likelihood along that tilt**,
- use (C_n) for the normalizing constant.

I’ll write it as a practical roadmap you could hand to someone implementing it.

---

## 1. Statistical setup

Observed data:
[
O_i=(\tilde T_i,\Delta_i), \qquad \tilde T_i=\min(T_i,C_i), \qquad \Delta_i=I(T_i\le C_i),
]
with
[
T \perp C,
]
and no covariates.

For a fixed target time (t*0), the parameter is
[
\Psi*{t_0}(F)=S_F(t_0)=P_F(T>t_0).
]

The key objects are:

1. **Full-data canonical gradient**
   [
   D_F^{\mathrm{Full}}(T;t_0)=I(T>t_0)-S_F(t_0).
   ]

2. **Full-data fluctuation direction**
   [
   D_{F,G}(t;t_0)=I_{F,G}^{-1}D_F^{\mathrm{Full}}(t;t_0).
   ]

3. **Observed-data EIC**
   [
   D^**{P*{F,G}}(O;t_0)=A_F\big(D_{F,G}(\cdot;t_0)\big)(O).
   ]

This is the exact operator logic in your draft: solve the inverse-information equation in the full-data space, then map it to the observed-data EIF with the score operator.

---

## 2. Overall design choice

For the **first implementation**, do the following:

- estimate (F_n) once using the **existing IPCW-HAL-MLE**,
- estimate (G_n) once using **reverse Kaplan–Meier**,
- for each target (t*0), compute its own (D*{F_n,G_n}(\cdot;t_0)),
- do a **scalar** TMLE update in (\epsilon),
- return the pointwise updated estimate
  [
  \hat S_n^*(t_0).
  ]

If you want estimates at 20 time points (t*1,\dots,t*{20}), just repeat the same pointwise procedure 20 times.

That is the cleanest first version because your closed-form (D\_{F,G}) derivation is already pointwise in (t_0).

---

## 3. Step 1: estimate the censoring distribution (G_n)

Since censoring is independent and unconditional, estimate the censoring survivor with reverse Kaplan–Meier:

- treat censoring as the event,
- use (1-\Delta_i) as the censoring-event indicator.

Store:

[
\bar G_n(t)=P_{G_n}(C\ge t),
]
the censoring jump times
[
u_1<\cdots<u_J,
]
and jump masses
[
\Delta G_n(u_j)=G_n(u_j)-G_n(u_j^-).
]

### Numerical safeguard

Define a truncated survivor
[
\bar G_n^\dagger(t)=\max{\bar G_n(t^-),\gamma_G},
]
with something like (\gamma_G=0.01) or (0.02).

Use (\bar G_n^\dagger) everywhere in the code where division by (\bar G_n) appears.

---

## 4. Step 2: fit the initial event-time estimator (F_n) using the existing IPCW-HAL-MLE

Let the HAL density model be parameterized by (f), with density

[
p_f(t)=\frac{\exp{f(t)}}{\int_0^\tau \exp{f(s)},ds}.
]

The package already has the IPCW-HAL-MLE, so use that directly as the initial fit. Conceptually it solves

[
f_n
===

\arg\max*f
\frac{1}{n}\sum*{i=1}^n
\frac{\Delta_i}{\bar G_n^\dagger(\tilde T_i)}
\log p_f(\tilde T_i),
]
subject to the HAL regularization/constraint already built into the package.

From this obtain:

[
F_n,\qquad p_n(t):=p_{f_n}(t),\qquad S_n(t):=\int_t^\tau p_n(u),du.
]

This is your initial plug-in estimator.

---

## 5. Step 3: for a fixed target time (t_0), compute the full-data gradient

For the chosen (t_0),

[
D_{F_n}^{\mathrm{Full}}(t;t_0)=I(t>t_0)-S_n(t_0).
]

This is the full-data EIF for survival at (t_0). It is the input to the inverse-information operator.

---

## 6. Step 4: compute the fluctuation direction (D\_{F_n,G_n}(\cdot;t_0))

This is the central mathematical step.

Define

[
h_n(t;t_0):=D_{F_n,G_n}(t;t_0)=I_{F_n,G_n}^{-1}D_{F_n}^{\mathrm{Full}}(t;t_0).
]

From your draft, this has a closed form involving (\bar G_n), (S_n), and a Stieltjes integral against (dG_n). Since (G_n) is KM, that integral becomes a sum over censoring jump times. Concretely, if you use the simplified representation, code it as a step function in (t):

[
h_n(t;t_0)
==========

## \frac{I(t>t_0)}{\bar G_n^\dagger(t_0)}

## S_n(t_0)

\sum\_{u_j\le t\wedge t_0}
\frac{S_n(t_0)}{S_n(u_j)}
\frac{\Delta G_n(u_j)}{(\bar G_n^\dagger(u_j))^2},
]

or whatever algebraically equivalent closed form you decide to keep from the draft. The important part is that **the (dG)-integral becomes a finite sum** over censoring jump times.

### Numerical centering

After computing (h_n), numerically center it under (F_n):

[
h_n^c(t;t_0)
============

h_n(t;t_0)-\int_0^\tau h_n(u;t_0)p_n(u),du.
]

Use (h_n^c) in the fluctuation model. This helps numerical stability and makes the exponential-family score cleaner.

---

## 7. Step 5: define the pointwise fluctuation submodel

For this fixed (t_0), define the one-dimensional exponential tilt

[
p\_{n,\epsilon}^{(t_0)}(t)
=========================

\frac{\exp{f_n(t)+\epsilon h_n^c(t;t_0)}}{C_n(\epsilon;t_0)},
]
where the normalizing constant is

[
C_n(\epsilon;t_0)
=================

\int_0^\tau \exp{f_n(s)+\epsilon h_n^c(s;t_0)},ds.
]

The induced survival function is

[
S\_{n,\epsilon}^{(t_0)}(y)
=========================

# \int*y^\tau p*{n,\epsilon}^{(t_0)}(u),du

\frac{\int_y^\tau \exp{f_n(u)+\epsilon h_n^c(u;t_0)},du}{C_n(\epsilon;t_0)}.
]

This is the full-data local least favorable submodel used for targeting.

---

## 8. Step 6: construct the observed-data log-likelihood along the submodel

Now hold (G_n) fixed and evaluate the observed-data likelihood induced by the tilted event-time law.

For the fixed (t_0), define

[
\ell\_{t_0}(\epsilon)
====================

\sum*{i=1}^n
\left[
\Delta_i \log p*{n,\epsilon}^{(t_0)}(\tilde T_i)

- (1-\Delta*i)\log S*{n,\epsilon}^{(t_0)}(\tilde T_i)
  \right].
  ]

This is the correct targeting objective.

A computationally convenient form is obtained by defining

[
A_i(\epsilon;t_0)
=================

\int\_{\tilde T_i}^\tau \exp{f_n(u)+\epsilon h_n^c(u;t_0)},du.
]

Then

[
S_{n,\epsilon}^{(t_0)}(\tilde T_i)=\frac{A_i(\epsilon;t_0)}{C_n(\epsilon;t_0)},
]

so

[
\ell\_{t_0}(\epsilon)
====================

\sum\_{i=1}^n
\left[
\Delta_i\big(f_n(\tilde T_i)+\epsilon h_n^c(\tilde T_i;t_0)\big)

- (1-\Delta_i)\log A_i(\epsilon;t_0)
  \right]
  ***

n\log C_n(\epsilon;t_0).
]

That is the version I would code.

---

## 9. Step 7: derive the score for the scalar optimization

Define

[
B_i(\epsilon;t_0)
=================

\int\_{\tilde T_i}^\tau
h_n^c(u;t_0)\exp{f_n(u)+\epsilon h_n^c(u;t_0)},du,
]

and

[
C\_{n,h}(\epsilon;t_0)
=====================

\int_0^\tau
h_n^c(u;t_0)\exp{f_n(u)+\epsilon h_n^c(u;t_0)},du.
]

Then the derivative of the log-likelihood is

[
U\_{t_0}(\epsilon)
=================

# \frac{d}{d\epsilon}\ell\_{t_0}(\epsilon)

\sum\_{i=1}^n
\left[
\Delta_i h_n^c(\tilde T_i;t_0)

- (1-\Delta_i)\frac{B_i(\epsilon;t_0)}{A_i(\epsilon;t_0)}
  \right]
  ***

n\frac{C\_{n,h}(\epsilon;t_0)}{C_n(\epsilon;t_0)}.
]

This is the scalar score equation you want to solve.

At (\epsilon=0), this score corresponds to the empirical observed-data score induced by the full-data fluctuation direction, i.e. the empirical mean of the observed-data EIF generated by (h_n^c).

---

## 10. Step 8: solve for (\hat\epsilon(t_0))

Because this is now a **scalar** optimization problem, do **not** use gradient descent.

Use a 1D solver.

### Recommended method

1. Evaluate (U\_{t_0}(0)).
2. Search for a bracket ([a,b]) such that
   [
   U_{t_0}(a),U_{t_0}(b)\le 0.
   ]
3. If such a bracket is found, solve
   [
   U_{t_0}(\epsilon)=0
   ]
   with a bracketed method like Brent.
4. If no sign change is found, maximize (\ell\_{t_0}(\epsilon)) on a bounded interval ([a,b]) with bounded Brent or another scalar line-search routine.

Define

[
\hat\epsilon(t_0)=\arg\max_\epsilon \ell_{t_0}(\epsilon).
]

This is the **single targeting update** for the time point (t_0). It is not a one-step estimator.

---

## 11. Step 9: update the density and survival

Once (\hat\epsilon(t_0)) is found, define the updated density

[
p_n^\*(t;t_0)
============

# p\_{n,\hat\epsilon(t_0)}^{(t_0)}(t)

\frac{\exp{f_n(t)+\hat\epsilon(t_0)h_n^c(t;t_0)}}{C_n(\hat\epsilon(t_0);t_0)}.
]

Then define the updated survival

[
S_n^*(y;t_0)=\int_y^\tau p_n^*(u;t_0),du.
]

The pointwise TMLE is

[
\hat\Psi_n^*(t_0)=S_n^*(t_0;t_0).
]

---

## 12. Step 10: diagnostic check

After the update, compute the observed-data EIF values:

[
\hat D_i^\*(t_0)
===============

A*{F_n^\*}\big(D*{F_n,G_n}(\cdot;t_0)\big)(O_i),
]

or recompute them in the updated model if you prefer to be fully consistent.

Check whether

[
\left|
\frac{1}{n}\sum_{i=1}^n \hat D_i^*(t_0)
\right|
]

is small.

That is the main targeting diagnostic.

Since your supervisor thinks a single targeting update is probably enough, I would make that the default. A second targeting pass can be added later as a safeguard, but not as the main implementation.

---

## 13. Multiple time points: repeat pointwise

If you want estimates at
[
t_1,\dots,t_K,
]
do **not** jointly optimize over a (K)-vector yet.

Instead, repeat the full scalar procedure separately for each (t_j):

1. compute
   [
   D_{F_n}^{\mathrm{Full}}(\cdot;t_j),
   ]
2. compute
   [
   h_n(\cdot;t_j)=D_{F_n,G_n}(\cdot;t_j),
   ]
3. define
   [
   p_{n,\epsilon}^{(t_j)},
   ]
4. maximize
   [
   \ell_{t_j}(\epsilon),
   ]
5. obtain
   [
   \hat\epsilon(t_j),
   ]
6. report
   [
   \hat S_n^*(t_j).
   ]

This is much easier to debug and matches your derivation naturally.

Yes, this means the resulting set of point estimates ({\hat S_n^\*(t_j)}) may not be perfectly monotone as a function of (t_j), because each time point gets its own updated distribution. But for a first method paper and first package implementation, this is the sensible route.

---

## 14. Computational simplification: reduce all integrals to sums on a grid

This is the main implementation trick.

If the package represents the univariate HAL log-density on intervals

[
0=a_0<a_1<\cdots<a_M=\tau
]

and both (f_n) and (h_n^c(\cdot;t_0)) are step functions on that grid, then

[
C_n(\epsilon;t_0)
=================

\sum\_{m=1}^M
\exp{f_m+\epsilon h_m(t_0)}\Delta a_m,
]

[
A_i(\epsilon;t_0)
=================

\sum*{m:,a*{m-1}\ge \tilde T_i \text{ or interval overlaps } [\tilde T_i,\tau]}
\exp{f_m+\epsilon h_m(t_0)}\Delta a_m,
]

and likewise for (B*i) and (C*{n,h}).

So everything reduces to finite sums and cumulative sums. No numerical quadrature is necessary.

This is probably the most important engineering point for integrating the method into the package.

---

## 15. Suggested function structure

I would build it as a thin wrapper around the existing IPCW-HAL-MLE.

### Core functions

`fit_censoring_km(y, delta)`
Returns (\bar G_n), censoring jump times, and jump masses.

`fit_ipcw_hal_density(y, delta, gbar, hal_args)`
Calls the existing package estimator and returns (f_n), (p_n), (S_n), and grid objects.

`compute_DFG_t0(fit, gbar, dG, t0)`
Computes (D\_{F_n,G_n}(\cdot;t_0)).

`tmle_objective_t0(eps, fit, h_t0, y, delta)`
Returns (\ell\_{t_0}(\epsilon)).

`tmle_score_t0(eps, fit, h_t0, y, delta)`
Returns (U\_{t_0}(\epsilon)).

`solve_epsilon_t0(...)`
Uses bracketed root-finding or bounded scalar optimization.

`target_survival_t0(...)`
Runs the full pointwise procedure and returns
[
\hat S_n(t_0),\quad \hat S_n^*(t_0),\quad \hat\epsilon(t_0),\quad \text{diagnostics}.
]

Then a higher-level function can loop over a user-supplied grid (t_1,\dots,t_K).

---

## 16. What I would implement first, exactly

If I were coding this next, I would do it in this order:

1. reverse-KM for censoring,
2. wrapper to the existing IPCW-HAL-MLE,
3. pointwise computation of (D\_{F_n,G_n}(\cdot;t_0)),
4. pointwise scalar log-likelihood (\ell\_{t_0}(\epsilon)),
5. pointwise scalar score (U\_{t_0}(\epsilon)),
6. bracketed solver for (\hat\epsilon(t_0)),
7. updated survival estimate (\hat S_n^\*(t_0)),
8. loop over multiple (t_j)’s.

That is the most plausible, stable, and theory-aligned first implementation.

---

## Final summary

The updated plan is:

- **Initial fit:** existing IPCW-HAL-MLE for (F_n)
- **Censoring fit:** reverse KM for (G_n)
- **Target type:** pointwise (S_F(t_0))
- **Full-data direction:** (D\_{F*n,G_n}=I^{-1}*{F*n,G_n}D*{F_n}^{\mathrm{Full}})
- **Tilt model:**
  [
  p_{n,\epsilon}^{(t_0)}(t)=\frac{\exp{f_n(t)+\epsilon h_n^c(t;t_0)}}{C_n(\epsilon;t_0)}
  ]
- **Targeting criterion:** observed-data log-likelihood
  [
  \ell\_{t_0}(\epsilon)
  ====================

  \sum*i \Big[\Delta_i \log p*{n,\epsilon}^{(t*0)}(\tilde T_i)+(1-\Delta_i)\log S*{n,\epsilon}^{(t_0)}(\tilde T_i)\Big]
  ]

- **Solver:** scalar root-finding / scalar optimization, not gradient descent
- **Update type:** single targeting update
- **Multiple times:** repeat pointwise, one (t_j) at a time

If you want, the next step is for me to convert this exact plan into a polished paper subsection or into implementation pseudocode.
