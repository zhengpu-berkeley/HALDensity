# EM Algorithm: Incomplete-Data Log-Likelihood Observations

## Key Insight: EM with Multiple Imputation Does NOT Guarantee Monotonic LL Increase

When using the EM algorithm with multiple imputation for right-censored density estimation,
the **incomplete-data log-likelihood is NOT guaranteed to increase monotonically**.

This is counterintuitive because classical EM theory guarantees that L(θ^(t+1)) ≥ L(θ^(t)).
However, this guarantee requires:

1. **Exact E-step**: Computing the exact expectation E[log L_complete | data, θ^(t)]
2. **Unconstrained M-step**: Finding the exact argmax of Q(θ | θ^(t))

Our implementation violates both assumptions:

### 1. Monte Carlo E-step (Multiple Imputation)
We approximate the E-step by sampling M imputations from f(T* | T* > C, θ^(t)).
This introduces variance and approximation error.

### 2. Regularized M-step (L1 Norm Constraint)
The M-step uses an L1 norm constraint on theta, which:
- Prevents overfitting to the augmented data
- But also prevents finding the true maximum of Q(θ | θ^(t))

---

## The Two Log-Likelihoods

### Complete-Data Log-Likelihood (what M-step optimizes)
```
L_complete(θ) = Σᵢ log f(Tᵢ*; θ)
```
Where Tᵢ* is the true event time (observed for uncensored, imputed for censored).

### Incomplete-Data Log-Likelihood (what we measure)
```
L_incomplete(θ) = Σᵢ [δᵢ · log f(Tᵢ; θ) + (1-δᵢ) · log S(Cᵢ; θ)]
```
Where:
- δᵢ = 1 for uncensored (observed event)
- δᵢ = 0 for censored (only know T* > C)
- S(C) = P(T > C) = 1 - F(C) is the survival function

### Why They Can Diverge

The M-step fits a density that best explains the **imputed complete data**.
But the incomplete-data LL includes log S(C) for censored observations, which is:
- Not directly optimized by the M-step
- Can decrease even as the complete-data LL improves

---

## Empirical Observations

### Experiment: IPCW init → EM iterations

| Iteration | LL(uncensored) | LL(censored) | LL(total) |
|-----------|----------------|--------------|-----------|
| Init      | 431.49         | -99.50       | 331.99    |
| EM-1      | 428.94 (-2.5)  | -98.16 (+1.3)| 330.79    |
| EM-2      | 426.38 (-2.6)  | -97.50 (+0.7)| 328.88    |
| ...       | decreasing     | improving    | net loss  |

**Pattern**: EM trades off between components:
- Uncensored LL decreases (density shifts away from observed events)
- Censored LL increases (survival at censoring times improves)
- Net effect is often a decrease in total LL

---

## Practical Implications

### ❌ Do NOT Assert: `ipcw_ll <= em_ll`
This assertion is invalid for regularized EM with multiple imputation.

### ✅ DO: Use Appropriate Regularization
Without regularization (`m_step_norm_constraint` very large), theta values explode:
- IPCW: theta L1 ≈ 9
- After 1 EM step with constraint=1000: theta L1 ≈ 1256 (explosion!)
- This leads to degenerate densities and catastrophic LL collapse

### ✅ DO: Use Reasonable Stopping Criteria
- Stop based on parameter stability (theta change < tol)
- Or complete-data LL convergence
- NOT based on expecting incomplete-data LL to increase

### ✅ DO: Keep M-step Constraint Similar to Init
Using `m_step_norm_constraint ≈ norm_constraint` (e.g., both = 8.0) works well.

---

## Theoretical Background

The EM algorithm's monotonicity proof relies on:

```
L(θ) = Q(θ | θ^(t)) - H(θ | θ^(t))
```

Where H involves the entropy of the conditional distribution of missing data.
The proof shows:
1. Q(θ^(t+1) | θ^(t)) ≥ Q(θ^(t) | θ^(t)) by definition of M-step
2. H(θ^(t) | θ^(t)) ≥ H(θ^(t+1) | θ^(t)) by KL divergence properties

Therefore L(θ^(t+1)) ≥ L(θ^(t)).

**But with Monte Carlo E-step and constrained M-step, neither step is exact!**

---

## References

- Dempster, Laird, Rubin (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm"
- Wei & Tanner (1990). "A Monte Carlo Implementation of the EM Algorithm"
- Nielsen (2000). "The Stochastic EM Algorithm: Estimation and Asymptotic Results"

---

*Last updated: Based on debugging session for `censor_data_comp_tests.py`*

