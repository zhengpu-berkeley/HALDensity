# IPCW-HAL-MLE summary

For right-censored data, the observed data are

\[
O_i = (\tilde T_i, \Delta_i), \qquad \tilde T_i = \min(T_i, C_i), \qquad \Delta_i = I(T_i \le C_i).
\]

In this repo, the IPCW initializer is implemented in three steps:

1. Estimate the censoring survival curve with a reverse Kaplan-Meier fit.
   The code fits `KaplanMeier` on `(\tilde T, \Delta)` but internally treats censoring as the event, i.e. uses `1 - \Delta`, so it estimates
   \[
   \hat S_C(t) = P(C > t).
   \]

2. Construct inverse-probability-of-censoring weights.
   For each observation, the weight is
   \[
   w_i = \frac{\Delta_i}{\max\{\hat S_C(\tilde T_i), \texttt{clip}\}},
   \]
   with default `clip = 10^{-6}`.

   This means:
   - uncensored observations (`\Delta_i = 1`) get weight `1 / \hat S_C(\tilde T_i)`;
   - censored observations (`\Delta_i = 0`) get weight `0`.

3. Fit the HAL density using only uncensored times, weighted by the IPCW weights.
   In code, the estimator subsets to the uncensored rows and calls HAL with those event times plus `sample_weights = w_i`.

Conceptually, the fitted initializer solves the weighted density estimation problem

\[
\hat f_{\mathrm{IPCW}}
\;=\;
\arg\max_{f \in \mathcal{F}_{\mathrm{HAL}}}
\sum_{i=1}^n
\frac{\Delta_i}{\max\{\hat S_C(\tilde T_i), \texttt{clip}\}}
\log f(\tilde T_i).
\]

Because `\Delta_i = 0` for censored observations, this is equivalent to summing only over uncensored observations:

\[
\hat f_{\mathrm{IPCW}}
\;=\;
\arg\max_{f \in \mathcal{F}_{\mathrm{HAL}}}
\sum_{i:\Delta_i = 1}
\frac{1}{\max\{\hat S_C(\tilde T_i), \texttt{clip}\}}
\log f(\tilde T_i).
\]

The HAL model is normalized as a density, so if the log-density basis expansion is written as `g(t)`, then

\[
f(t)
=
\frac{\exp\{g(t)\}}{\int_0^\tau \exp\{g(u)\}\,du}.
\]

So the IPCW-HAL-MLE is a weighted HAL maximum likelihood estimator where censoring is handled entirely through the weights.

## Validation score used in tuning

The repo also includes an IPCW validation metric, `ipcw_loglik`, used for cross-validation/tuning. On a validation fold it evaluates

\[
\mathrm{IPCW\_loglik}(f)
=
\sum_{i=1}^n
\frac{\Delta_i}{\max\{\hat S_C(\tilde T_i), \texttt{clip}\}}
\log f(\tilde T_i),
\]

where `\hat S_C` is estimated from the training fold.

So the implementation has a clean split:

- `KaplanMeier`: estimates `\hat S_C`;
- `compute_ipcw_weights`: computes `w_i = \Delta_i / \hat S_C(\tilde T_i)`;
- `RightCensoredInitEstimator`: fits the weighted HAL density on uncensored event times;
- `ipcw_loglik`: reuses the same IPCW-weighted log-likelihood as a validation criterion.
