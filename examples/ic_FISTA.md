Yes — that skeleton is right (up to small details like open/closed interval endpoints and whether you penalize the intercept). Concretely:

## Objective

For interval-censored (O*i=(L_i,R_i]),
[
\min*\beta;; g(\beta)+\lambda|\beta|_1,
\qquad
g(\beta)= -\sum_{i=1}^n \log P*i(\beta),
\quad
P_i(\beta)=\int*{L*i}^{R_i} p*\beta(x),dx.
]

If you use a HAL **log-density** model
[
f_\beta(x)=\beta^\top \phi(x),\qquad
p_\beta(x)=\frac{\exp(f_\beta(x))}{\int \exp(f_\beta(u)),du},
]
then (g) is smooth and (\lambda|\beta|\_1) is the nonsmooth part for ISTA/FISTA.

## Gradient of the smooth part

Your expression is correct:
[
\nabla g(\beta)
=\sum*{i=1}^n\Big( \mathbb E*\beta[\phi(T)] - \mathbb E\_\beta[\phi(T)\mid T\in(L_i,R_i]]\Big).
]

Equivalently, writing (\mu(\beta)=\mathbb E*\beta[\phi(T)]),
[
\nabla g(\beta)= n,\mu(\beta) - \sum*{i=1}^n \mu*i(\beta),
\quad
\mu_i(\beta)=\mathbb E*\beta[\phi(T)\mid T\in(L_i,R_i]].
]

## ISTA / proximal-gradient step

Given step size (t*k),
[
\beta^{k+1}=\operatorname{soft}*{\lambda t_k}\Big(\beta^k - t_k \nabla g(\beta^k)\Big),
]
(where you typically **don’t** soft-threshold the intercept).

Use backtracking line search for (t_k) unless you have a trustworthy Lipschitz bound.

## FISTA acceleration

Same prox step but at an extrapolated point (y*k):
[
\beta^{k+1}=\operatorname{prox}*{\lambda t_k}\big(y_k - t_k \nabla g(y_k)\big),
]
[
\theta\*{k+1}=\frac{1+\sqrt{1+4\theta_k^2}}{2},\qquad
y*{k+1}=\beta^{k+1}+\frac{\theta*k-1}{\theta*{k+1}}(\beta^{k+1}-\beta^k).
]

## Practical computation note (what makes this work)

To evaluate (\mu(\beta)) and each (\mu_i(\beta)), you’ll almost surely do **numerical integration on a grid**:

- compute normalized weights proportional to (\exp(f\_\beta(t_j))\Delta_j),
- (\mu(\beta)) is the weighted average of (\phi(t_j)) over the whole grid,
- (\mu_i(\beta)) is the same weighted average restricted to grid points in ((L_i,R_i]), divided by (P_i(\beta)).

That’s the computational core.

If you want, I can write the exact “grid formula” for (\nabla g(\beta)) that avoids looping over (i) too expensively (using cumulative sums / interval indexing), but your conceptual skeleton is exactly the right one.
