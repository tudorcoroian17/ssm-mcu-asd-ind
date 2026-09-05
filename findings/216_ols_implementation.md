# Implementation: how `runs/knob_cost.py` fits the OLS model

**Status:** Reference. Documents the library calls that produce the knob-cost figures: which
functions, what they take, what they return, and the two non-obvious failure modes that have bitten
this code. The statistical justification for the method is the companion file
`findings/215_ols_derivation.md`; read that for why these quantities mean what they mean. This
file is what each line does.

**Context:** `runs/knob_cost.py` fits one OLS model per (pooling, distance head) cell over the 72
case-1 configurations, then reads adjacent-step contrasts from it. The statistics library is
`statsmodels`; the formula parser underneath it is Patsy. Everything here concerns those two.

---

## 1. Libraries

| library | role | import in the script |
| :--- | :--- | :--- |
| `statsmodels` | fits the OLS model, returns coefficients, covariances, contrasts | `import statsmodels.formula.api as smf` |
| Patsy | parses the formula string into the design matrix $X$ (§1 of `215`) | used implicitly by `smf.ols` |
| `pandas` | holds the 72-row table the model is fit on | `import pandas as pd` |
| `numpy` | flattens the arrays `t_test` returns | `import numpy as np` |

Patsy is not imported directly. `statsmodels.formula.api` calls it internally to turn the
formula string into the design matrix, so its behavior — especially categorical coding, §3 — is
part of this pipeline even though it never appears in the code.

Versions matter for two things: the exact term-name spelling (§3.3) and the return shape of
`t_test` (§5.2). Both are noted where they bite. Confirm your installed versions with
`import statsmodels; print(statsmodels.__version__)`.

References: statsmodels formula API,
<https://www.statsmodels.org/stable/example_formulas.html>; Patsy,
<https://patsy.readthedocs.io/en/latest/>.

---

## 2. Fitting the model

The fit is one call:

```python
import statsmodels.formula.api as smf

model = smf.ols(
    'auc ~ C(selective) + C(d_state) + C(n_layers) + C(expand) + C(discretization)',
    data=sub,
).fit()
```

`sub` is the DataFrame for one (pooling, head) cell: 72 rows, one per configuration, deduplicated
so each configuration appears once (`findings/215` §1 requires one $y_i$ per configuration).

### 2.1 The formula string

The string `'auc ~ C(selective) + C(d_state) + ...'` is a **Wilkinson formula**. It is not Python;
it is a small domain-specific language Patsy parses. Reading it:

- `auc` left of `~` is the response, $\mathbf{y}$.
- Each term right of `~` is a predictor. `+` means "include this term," not addition — it adds a
  factor to the model, it does not sum values.
- `C(...)` marks a variable as **categorical**. Without it, Patsy would treat `d_state` as a
  numeric column and fit a single slope (one coefficient for a linear trend in 8, 16, 32). With
  it, Patsy fits a separate effect per level. Categorical is what makes the model additive over
  discrete knob levels rather than assuming a linear response to knob magnitude — which is the
  model `findings/215` derives.
- The intercept is added automatically. The formula above is implicitly `auc ~ 1 + C(selective) +
  ...`; the leading `1` (the intercept) is default.

`.fit()` runs the estimation, returning a results object (a `RegressionResults`). `model` in the
snippet is that results object, not the unfitted specification.

References: formula syntax, statsmodels,
<https://www.statsmodels.org/stable/example_formulas.html>; Wilkinson notation origin and the
`C()` operator, Patsy, <https://patsy.readthedocs.io/en/latest/formulas.html>.

### 2.2 Column dtypes before fitting

`runs/knob_cost.py` casts the numeric knob columns to `int` before the fit:

```python
sub['d_state'] = sub['d_state'].astype(int)
sub['expand'] = sub['expand'].astype(int)
sub['n_layers'] = sub['n_layers'].astype(int)
```

This is not cosmetic. The level labels in the fitted term names come from the **string form of the
column values** (§3.3). If `d_state` arrives as a float, the terms are named `C(d_state)[T.16.0]`;
as an int, `C(d_state)[T.16]`. The contrast strings in §5 are written to match the int spelling,
so the cast is what makes them resolve. Casting also guarantees a stable, predictable reference
level (§3.2).

---

## 3. Categorical coding and the reference level

This section is the one most likely to silently produce wrong or empty output, so it is spelled
out fully.

### 3.1 Treatment (dummy) coding

`C(d_state)` uses **treatment coding** by default (also called dummy coding). For a $k$-level
factor, treatment coding creates $k-1$ indicator columns, each comparing one level against a
single **reference level**. The reference level gets no column; its effect is folded into the
intercept. So `C(d_state)` with levels {8, 16, 32} produces two columns, `C(d_state)[T.16]` and
`C(d_state)[T.32]`, each measuring its level's effect **relative to the reference**. This is the
coding `findings/215` §6 assumes when it explains why two same-factor coefficients share the
reference's noise and are therefore correlated.

Reference: Patsy categorical coding, including treatment coding,
<https://patsy.readthedocs.io/en/latest/categorical-coding.html>.

### 3.2 Which level is the reference

By default, Patsy picks the reference level as the **first level in sorted order**. For integer
columns that is the smallest value, so `d_state` references 8 and `n_layers` references 1 — the
cheapest levels. This is deliberate and convenient: the two `d_state` coefficients then read as
"16 versus 8" and "32 versus 8," which is the direction the knob-cost table wants, and it makes
the smallest, cheapest model the baseline everything else is measured against.

For the two-level factors the reference is the alphabetically or numerically first level:
`selective` references `False`, `expand` references 1, `discretization` references `euler`
(alphabetically before `zoh`). Each two-level factor then has exactly one coefficient.

If a different reference were ever wanted, Patsy allows `C(d_state, Treatment(reference=32))`, but
the script does not use this — the default sorted-first behavior gives the wanted baselines
directly.

### 3.3 The term-name failure mode

The fitted coefficient names are built as `C(<factor>)[T.<level>]`, where `<level>` is the
**string** form of the level value. This is why §2.2's int cast matters. The failure it prevents:

- `d_state` left as float → terms named `C(d_state)[T.16.0]`.
- Contrast strings and lookups in the script written for `C(d_state)[T.16]`.
- The lookup finds no matching term. Depending on the code path this either raises or, worse,
  silently skips the row.

This exact mismatch produced an empty or short coefficient table earlier in development. The guard
is to verify the term names against what the fit actually produced before trusting any output:

```python
print(list(model.params.index))
```

Expected, with the int cast in place:

```python
['Intercept', 'C(selective)[T.True]', 'C(d_state)[T.16]', 'C(d_state)[T.32]',
'C(n_layers)[T.2]', 'C(n_layers)[T.4]', 'C(expand)[T.2]', 'C(discretization)[T.zoh]']
```


If any level shows a trailing `.0`, the cast did not take and every contrast in §5 that references
that factor will fail to resolve.

**Claude comment:** this is the single most fragile point in the pipeline. It is fragile precisely
because it fails quietly — a missing term is skipped, not flagged, so the figure renders with a row
silently absent rather than erroring. The `print(list(model.params.index))` check is cheap and
should be run whenever the input schema might have changed.

---

## 4. Reading coefficients, intervals, and p-values directly

For the two-level factors and the against-reference steps (`16 -> 8`, `2 -> 1`), the wanted
quantity is a single coefficient, read straight off the results object:

```python
model.params['C(expand)[T.2]']          # point estimate, a float
model.conf_int(alpha=0.05).loc['C(expand)[T.2]']   # [lower, upper], a 2-vector
model.pvalues['C(expand)[T.2]']         # two-sided p-value for H0: coef = 0
```

- `model.params` is a pandas Series indexed by term name; each entry is a $\hat\beta_j$.
- `model.conf_int(alpha=0.05)` returns a DataFrame with columns 0 and 1 (lower, upper) for the
  $100(1-\alpha)\%$ interval — 95% at $\alpha=0.05$. This is the interval derived in `findings/215`
  §7.1.
- `model.pvalues` is the two-sided $p$-value for the null that the coefficient is zero. Read it as
  an effect-size flag, not a significance verdict, for the reasons in `findings/215` §7.3.

The script applies a **sign flip** to each of these so that positive means "the cheaper or simpler
option scores higher." The flip negates the estimate and swaps and negates the interval endpoints,
because negating an interval reverses which endpoint is the lower bound:

```python
lo, hi = sorted([sign * ci_lower, sign * ci_upper])
```

The `sorted(...)` is what keeps `lo <= hi` after a negation. Omitting it would produce inverted
error bars whenever `sign` is $-1$, which for this script is every row.

Reference: `RegressionResults` attributes (`params`, `pvalues`, `conf_int`),
<https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html>.

---

## 5. Adjacent-step contrasts with `t_test`

The `32 -> 16` and `4 -> 2` steps are contrasts, not coefficients (`findings/215` §6). They are
evaluated with `model.t_test`, which is the call that carries the coefficient covariance correctly.

### 5.1 The call

```python
tt = model.t_test('C(d_state)[T.32] - C(d_state)[T.16]')
```

The argument is a **contrast string** written in terms of the fitted coefficient names. Patsy
parses it into the row vector $L$ of `findings/215` §6.1:
`C(d_state)[T.32] - C(d_state)[T.16]` becomes $L$ with $+1$ in the `[T.32]` position, $-1$ in the
`[T.16]` position, and zero elsewhere. `t_test` then computes $L\hat{\boldsymbol{\beta}}$ and its
standard error $\sqrt{\hat\sigma^2 L (X^\top X)^{-1} L^\top}$ from the **full** covariance matrix,
so the correlation between the two coefficients (`findings/215` §6.2) is included. This is the
whole reason for using `t_test` instead of `model.params['...'] - model.params['...']`, which would
give the right point estimate with no correct standard error.

The contrast string must use the exact term names from §3.3. A trailing `.0` or a stray space
makes Patsy fail to resolve the term, and the script's `try/except` around the call then drops the
row silently — the same quiet-failure mode as §3.3, one level down. Verify with:

```python
print(model.t_test('C(d_state)[T.32] - C(d_state)[T.16]').effect)   # must print a number
```

### 5.2 What `t_test` returns

The returned object (a `ContrastResults`) exposes the contrast's estimate, standard error,
$t$-statistic, $p$-value, and confidence interval:

```python
tt.effect            # point estimate of the contrast, L @ beta_hat
tt.sd                # standard error of the contrast
tt.tvalue            # t-statistic
tt.pvalue            # two-sided p-value
tt.conf_int(alpha=0.05)   # 95% interval, as derived in findings/215 §7.1
```

**Return-shape caveat.** Across statsmodels versions these come back as scalars, 0-d arrays, or
1-element arrays depending on how the contrast was specified. `runs/knob_cost.py` flattens
defensively so one row does not break on a shape it did not expect:

```python
est   = float(np.ravel(tt.effect)[0])
lo_ci, hi_ci = np.ravel(tt.conf_int(alpha=0.05))[:2]
pval  = float(np.ravel(tt.pvalue)[0])
```

`np.ravel` turns any of scalar/0-d/1-d into a flat 1-d array, and `[0]` or `[:2]` then reads the
value regardless of the original shape. Without this, a version that returns a 2-d
`conf_int` would raise on the unpacking.

The same sign-flip and `sorted` as §4 apply to `est`, `lo_ci`, `hi_ci`, so positive still means
"cheaper option better."

Reference: `t_test` and the `ContrastResults` it returns,
<https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.t_test.html>.

### 5.3 Why not `t_test` for the single-coefficient steps too

The against-reference steps (`16 -> 8`, `2 -> 1`, and the two-level factors) are single
coefficients, so `t_test('C(expand)[T.2]')` and reading `model.params`/`conf_int` give the
identical number — for a single coefficient there is no covariance term to carry. The script can
and does use `t_test` uniformly for both cases, which keeps one code path; the derivation in
`findings/215` §6 shows the two agree when $L$ selects a single coefficient.

---

## 6. The per-cell loop

The full pipeline, assembled:

1. For each (pooling, head) cell, filter the scores table to that cell and deduplicate to one row
   per configuration (§2).
2. Cast the numeric knob columns to `int` (§2.2).
3. Fit `smf.ols(formula, data=sub).fit()` (§2).
4. For each row in the knob-cost table, evaluate its contrast with `model.t_test(...)` (§5),
   apply the sign flip (§4), and record estimate, interval, and $p$-value.
5. Pair each row with the flash it frees (computed separately, from matched configuration pairs —
   not part of the OLS, and out of scope for this file).

One OLS is fit per cell, so twelve fits total for the three pooling modes times four heads. Each
fit is independent; there is no pooling of information across cells, which is deliberate — a knob's
effect can and does differ by head (`findings/220` §3), and fitting per cell is what lets the
figure show that.

---

## 7. What is deliberately not in this pipeline

To bound the scope, three things this OLS does **not** do:

- **No interaction terms.** The formula is main-effects only. Interactions (`findings/220` §5) are
  left in the residual by design; a forest plot of main effects wants an additive model
  (`findings/215` §5.3, §7.3). The interaction is characterized separately, by the paired-delta
  and interaction-matrix figures, not by this model.
- **No weighting.** Every configuration counts equally. Ordinary, not weighted, least squares.
- **No multiple-comparison correction.** The $p$-values are raw. Given the single-seed,
  single-fold caveat (`findings/215` §7.3), they are read as effect-size flags against the seed
  band rather than as a family of hypothesis tests that would need Bonferroni or similar. Applying
  a correction would imply an inferential reading the data does not support.

---

## 8. References

Check that the statsmodels and Patsy pages match your installed versions; both libraries revise
documentation and occasionally behavior (notably `t_test` return shapes, §5.2) across releases.

- statsmodels formula API and examples.
  <https://www.statsmodels.org/stable/example_formulas.html>
- statsmodels `RegressionResults` (attributes `params`, `pvalues`, `conf_int`).
  <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html>
- statsmodels `RegressionResults.t_test`.
  <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.t_test.html>
- Patsy formulas and the `C()` operator.
  <https://patsy.readthedocs.io/en/latest/formulas.html>
- Patsy categorical coding (treatment coding, reference-level selection).
  <https://patsy.readthedocs.io/en/latest/categorical-coding.html>
- Companion derivation: `findings/215_ols_derivation.md`.