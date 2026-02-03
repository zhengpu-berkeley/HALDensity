Cool — then the checklist needs to shift from “assume EM + knot invariance + density validity are fine” to **making them auditable and falsifiable**.

Below is the **updated checklist**, explicitly targeting:

1. **EM diagnostics** (is EM doing EM?)
2. **Knot invariance** (is EM truly parametric on the input knots?)
3. **Density validity** (is the returned density a proper density, and is normalization internally consistent?)

I’ll keep Stage 1 conservative CV as-is (you already believe it), and I’ll also include the “no-oversmooth is a member of oversmooth family” test, but now it will hinge on the knot/EM checks.

---

## Updated checklist (focus: EM + knots + density validity)

### 1) EM diagnostics: confirm EM is progressing and M-step is solving what you think

**1.1 Print an EM trace (per iteration)**
You want each EM iteration to print at least:

* iter index
* objective you track (preferably `incomplete_loglik` on observed data, or your MC `Q`)
* step size / parameter change (e.g. `||theta^{new}-theta^{old}||_∞`)
* solver status from M-step (optimal/infeasible/etc.)

**Pass conditions**

* M-step solver status is consistently “optimal” (or your accepted status).
* The EM objective is **nondecreasing up to MC noise** (allow small dips if MC E-step).
* Parameter updates shrink and stopping criterion triggers for the right reason.

**Drop-in diagnostic cell**

```python
def print_em_trace(em_result, name="EM"):
    meta = getattr(em_result, "metadata", {}) or {}
    recs = meta.get("em_records", None)
    print(f"\n[{name}] metadata keys:", list(meta.keys())[:30])
    if recs is None:
        print(f"[{name}] No em_records found in metadata.")
        return
    df = pd.DataFrame(recs)
    print(f"[{name}] em_records columns:", df.columns.tolist())
    print(df.tail(8))
    # monotonicity check for any likely objective column
    for col in ["incomplete_ll", "loglik", "objective", "Q"]:
        if col in df.columns:
            d = df[col].diff()
            print(f"[{name}] {col}: #decreases={(d < -1e-10).sum()}, last={df[col].iloc[-1]}")
            break

print_em_trace(em_result_oversmooth, "EM oversmooth(best)")
print_em_trace(em_result_no_oversmooth, "EM no-oversmooth")
```

**If this fails**, the next thing to print is the M-step solver status *per iteration* (if you store it). If you don’t store it, add it to `em_records` (high value).

---

### 2) Knot invariance: confirm “parametric EM” truly reuses the stage-1 knots

There are **two** knot invariance checks you need:

#### 2.1 Output knots equal to the *input stage1 knots* (for each run)

For both:

* no-oversmooth EM: compare EM knots vs `init_result.estimator` knots
* oversmooth EM: compare EM knots vs the oversmoothed-stage1 knots for the chosen factor

**Pass condition**

* exact equality of knot arrays (same length and same values), or max abs diff ~ 0 if floats.

**Drop-in diagnostic**

```python
def get_knots(est):
    r = est.get_results()
    # adapt if your key differs
    K = r.get("knots", r.get("selected_knots", None))
    return None if K is None else np.asarray(K)

def compare_knots(K1, K2, name1="A", name2="B"):
    if K1 is None or K2 is None:
        print(f"Knot compare {name1} vs {name2}: missing knots (None).")
        return
    print(f"Knot compare {name1} vs {name2}: len {len(K1)} vs {len(K2)}")
    if len(K1) != len(K2):
        print("  FAIL: different lengths")
        return
    print("  exact_equal:", np.array_equal(K1, K2))
    print("  max_abs_diff:", np.max(np.abs(K1 - K2)))

K_init = get_knots(init_result.estimator)
K_em_no = get_knots(em_result_no_oversmooth.estimator)
compare_knots(K_init, K_em_no, "Stage1(init)", "EM(no-oversmooth)")
```

#### 2.2 Oversmooth path must preserve knots factor-by-factor (stronger)

This is the one people often miss:

For each oversmooth factor `a`:

* oversmoothed stage1 knots = `K_a`
* EM output knots for that `a` must equal `K_a`

**Pass condition**

* holds for *every factor*, not just the best one.

If your tuner doesn’t expose per-factor stage1 estimators, you should temporarily modify it to return a debug list like:
`[(factor, stage1_estimator_for_factor, em_estimator_for_factor)]`.

**Minimal print you want**

* factor, n_knots_stage1, n_knots_em, exact_equal, max_abs_diff

---

### 3) Density validity: confirm the returned density is a proper density and normalization is consistent

You need **three** checks:

#### 3.1 Nonnegativity

**Pass:** min density ≥ 0 (allow tiny negative numerical noise like -1e-12)

#### 3.2 Integrates to 1 (numerically)

Use a *fine* grid and trapezoid rule on (0,1), not including endpoints.

**Pass:** integral in [0.98, 1.02] (tighter if you expect precision)

#### 3.3 Internal normalization constant matches external integration

This catches the specific bug class: “the estimator’s internal normalizer is computed on knots/segments, but get_density_at_points is interpolating differently.”

**Drop-in diagnostic**

```python
def density_sanity(est, name="est"):
    grid = np.linspace(0.001, 0.999, 5000)
    f = est.get_density_at_points(grid)
    f = np.asarray(f)

    print(f"\n[{name}] density sanity")
    print("  min/max:", float(np.min(f)), float(np.max(f)))
    print("  any_neg:", bool(np.any(f < -1e-12)))

    area = np.trapz(f, grid)
    print("  integral(trapz):", float(area))

density_sanity(init_result.estimator, "Stage1(init)")
density_sanity(em_result_oversmooth.estimator, "EM oversmooth(best)")
density_sanity(em_result_no_oversmooth.estimator, "EM no-oversmooth")
```

**If integral is off**, you need to determine which of these is wrong:

* the fitted parameterization is fine but `get_density_at_points` is wrong,
* or the estimator is not enforcing normalization the way you think.

A strong follow-up debug print is:

* compute the normalizing constant using *the same piecewise structure* implied by knots and theta, and compare it to 1.
  (If you tell me whether your basis is 0-order piecewise constant on knot bins vs piecewise linear in log-density, I can give the exact “model-consistent integral” check.)

---

## 4) Membership check (now tied to the above): no-oversmooth must equal oversmooth with factor=1 **when everything else is frozen**

Since you’re now worried EM/knot/density validity are not fine, **membership only makes sense after those pass**. But here is the exact membership test:

**Pass condition (strong membership):**

* Run oversmooth tuner with `oversmooth_factors=[1.0]` (only).
* Freeze EM knobs to match the no-oversmooth run (especially any CV-tuned `m_step_norm_multiplier`).
* Then:

  * stage1 knots match
  * EM knots match
  * theta close
  * densities close

**Why this matters in your current notebook**
In your notebook, these two are *not the same algorithm* by design:

* `do_over_smooth=True` → grid search over `oversmooth_factors`
* `do_over_smooth=False` → CV for `m_step_norm_multiplier`

So “factor=1” is *not* automatically identical unless you **disable/align** the other tuning path.

---

## The updated checklist in one screen

1. **EM trace exists + solver optimal + objective stable (up to MC noise)**
2. **Knot invariance (run-level):** `K_EM == K_input_stage1`
3. **Knot invariance (factor-level):** for each factor `a`, `K_EM(a) == K_stage1(a)`
4. **Density validity:** nonnegative + integrates ~1 (fine grid)
5. **Normalization consistency:** internal normalizer aligns with numerical integral
6. **Membership:** oversmooth([1.0]) reproduces no-oversmooth when EM knobs are frozen

---

If you paste the output of:

* `init_result.estimator.get_results().keys()`
* `em_result_oversmooth.estimator.get_results().keys()`
* `em_result_oversmooth.metadata.keys()` (or whatever container holds records)

…I can tailor the exact key names so the diagnostics run *without guessing* (right now I had to use common placeholders like `"knots"`, `"em_records"`).
