# Background: the OLS model behind the knob-cost charts, derived

**Status:** Reference. Explains the statistical method used by `runs/knob_cost.py`
to produce the knob-cost forest and scatter figures. No results here; the numbers those
figures carry are in `findings/220_full_sweep_results_and_refitting.md`. The implementation —
which library calls produce these quantities — is the companion file
`findings/216_ols_implementation.md`.

**Context:** The knob-cost figures report, for each architecture knob, how much AUC a one-step
change costs and how much flash it frees. The AUC-cost numbers are coefficients of an ordinary
least squares (OLS) model fit over the 72 case-1 sweep configurations, one model per
(pooling, distance head) cell. This document derives every quantity used: the coefficient
estimator, its covariance, why the balanced factorial makes the estimates unconfounded, and why
an adjacent-step contrast such as `d_state 32 -> 16` needs `t_test` rather than arithmetic on two
coefficients.

**Claude comment:** the single most important thing in this file is §7 — what the confidence
interval does and does not claim. Every other section is standard linear-model theory; §7 is
where the single-seed, single-fold nature of the data bounds what any of it means, and it is the
part a committee will press on.

---

## 1. Notation and the model

Index the $n = 72$ case-1 configurations by $i$. Let $y_i$ be the AUC of configuration $i$ at a
fixed (pooling, head) cell. Each configuration is described by five categorical knobs. Writing the
knob levels as indicator (dummy) variables $x_{ij}$, the model is

$$
y_i = \beta_0 + \sum_{j=1}^{p-1} \beta_j\, x_{ij} + \varepsilon_i ,
$$

where $\beta_0$ is the intercept, the $\beta_j$ are level effects, and $\varepsilon_i$ is an
error term. In matrix form,

$$
\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\mathbf{y}\in\mathbb{R}^{n},\;\;
X\in\mathbb{R}^{n\times p},\;\;
\boldsymbol{\beta}\in\mathbb{R}^{p} .
$$

$X$ is the **design matrix**: row $i$ holds a leading $1$ for the intercept followed by the dummy
indicators for configuration $i$'s knob levels. How those dummies are built from the knob columns
— treatment coding, reference-level choice — is `findings/216` §3; here it is enough that $X$ has
full column rank, which §5 shows the balanced design guarantees.

The standard OLS assumptions (Gauss-Markov conditions):

1. **Linearity:** $\operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}$, so
   $\operatorname{E}[\mathbf{y}] = X\boldsymbol{\beta}$.
2. **Homoscedasticity and no correlation:**
   $\operatorname{Cov}(\boldsymbol{\varepsilon}) = \sigma^2 I_n$ — every error has the same
   variance $\sigma^2$ and errors are mutually uncorrelated.
3. **Full rank:** $\operatorname{rank}(X) = p$, so $X^\top X$ is invertible.

Normality of $\boldsymbol{\varepsilon}$ is **not** needed to estimate $\boldsymbol{\beta}$ or its
covariance (§2–§4). It is needed only for the $t$-distribution of the confidence interval (§6).
§7 discusses how well these assumptions actually hold for this data; the honest answer is "poorly
enough that the interval is descriptive, not inferential."

Reference: ordinary least squares overview and assumptions, Wikipedia,
<https://en.wikipedia.org/wiki/Ordinary_least_squares>; design matrix,
<https://en.wikipedia.org/wiki/Design_matrix>.

---

## 2. The estimator: minimizing squared error

OLS chooses $\hat{\boldsymbol{\beta}}$ to minimize the residual sum of squares (RSS),

$$
S(\boldsymbol{\beta})
= \lVert \mathbf{y} - X\boldsymbol{\beta} \rVert^2
= (\mathbf{y} - X\boldsymbol{\beta})^\top (\mathbf{y} - X\boldsymbol{\beta}) .
$$

Expanding, and using that $\boldsymbol{\beta}^\top X^\top \mathbf{y}$ is a scalar and therefore
equal to its own transpose $\mathbf{y}^\top X \boldsymbol{\beta}$,

$$
S(\boldsymbol{\beta})
= \mathbf{y}^\top\mathbf{y}
- 2\,\boldsymbol{\beta}^\top X^\top \mathbf{y}
+ \boldsymbol{\beta}^\top X^\top X \boldsymbol{\beta} .
$$

Differentiate with respect to $\boldsymbol{\beta}$ and set to zero. Using the vector-calculus
identities $\partial(\mathbf{a}^\top\boldsymbol{\beta})/\partial\boldsymbol{\beta} = \mathbf{a}$
and $\partial(\boldsymbol{\beta}^\top A \boldsymbol{\beta})/\partial\boldsymbol{\beta} =
2A\boldsymbol{\beta}$ for symmetric $A$,

$$
\frac{\partial S}{\partial \boldsymbol{\beta}}
= -2 X^\top \mathbf{y} + 2 X^\top X \boldsymbol{\beta}
= \mathbf{0} .
$$

This gives the **normal equations**

$$
X^\top X\, \hat{\boldsymbol{\beta}} = X^\top \mathbf{y} ,
$$

and, since $X^\top X$ is invertible under assumption 3,

$$
\boxed{\;\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}\;} .
$$

The second derivative $\partial^2 S / \partial\boldsymbol{\beta}\,\partial\boldsymbol{\beta}^\top
= 2X^\top X$ is positive definite, so this stationary point is the unique global minimum.

Reference: derivation of the normal equations, Wikipedia OLS §"Estimation",
<https://en.wikipedia.org/wiki/Ordinary_least_squares#Estimation>; a fuller treatment with the
matrix calculus spelled out is Faraway, *Practical Regression and Anova using R*, ch. 2,
<https://cran.r-project.org/doc/contrib/Faraway-PRA.pdf>.

---

## 3. Unbiasedness

The estimator is unbiased under assumption 1. Substituting $\mathbf{y} = X\boldsymbol{\beta} +
\boldsymbol{\varepsilon}$,

$$
\hat{\boldsymbol{\beta}}
= (X^\top X)^{-1} X^\top (X\boldsymbol{\beta} + \boldsymbol{\varepsilon})
= \boldsymbol{\beta} + (X^\top X)^{-1} X^\top \boldsymbol{\varepsilon} .
$$

Taking expectations and using $\operatorname{E}[\boldsymbol{\varepsilon}] = \mathbf{0}$,

$$
\operatorname{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta} .
$$

The estimation error is $\hat{\boldsymbol{\beta}} - \boldsymbol{\beta} = (X^\top X)^{-1} X^\top
\boldsymbol{\varepsilon}$, a fact §4 reuses directly.

The Gauss-Markov theorem strengthens this: among all linear unbiased estimators,
$\hat{\boldsymbol{\beta}}$ has the smallest variance (it is BLUE — best linear unbiased
estimator). This is why OLS is the default rather than one choice among many.
Reference: <https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem>.

---

## 4. The coefficient covariance matrix

The whole point of the forest plot's error bars is the covariance of $\hat{\boldsymbol{\beta}}$.
From §3, the error is a linear map of $\boldsymbol{\varepsilon}$. Write $M = (X^\top X)^{-1}
X^\top$, so $\hat{\boldsymbol{\beta}} - \boldsymbol{\beta} = M\boldsymbol{\varepsilon}$. For any
fixed matrix $M$ and random vector with $\operatorname{Cov}(\boldsymbol{\varepsilon}) = \sigma^2
I$,

$$
\operatorname{Cov}(M\boldsymbol{\varepsilon})
= M \operatorname{Cov}(\boldsymbol{\varepsilon}) M^\top
= \sigma^2 M M^\top .
$$

Now expand $MM^\top$, using that $(X^\top X)^{-1}$ is symmetric:

$$
M M^\top
= (X^\top X)^{-1} X^\top \, X (X^\top X)^{-1}
= (X^\top X)^{-1} (X^\top X)(X^\top X)^{-1}
= (X^\top X)^{-1} .
$$

Therefore

$$
\boxed{\;\operatorname{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2 (X^\top X)^{-1}\;} .
$$

The diagonal entries are the variances of individual coefficients; the off-diagonal entries are
covariances between coefficients. §6 shows the off-diagonals are exactly what a naive "subtract
two coefficients" contrast would ignore.

$\sigma^2$ is unknown and is estimated from the residuals (§6.1).

Reference: <https://en.wikipedia.org/wiki/Ordinary_least_squares#Finite_sample_properties>.

---

## 5. Why balance makes the estimates unconfounded

This is the property that lets the forest plot report "the effect of `expand`" as a single number
that does not depend on which other knobs are in the model. It follows from the design being a
**balanced full factorial**: every combination of knob levels appears the same number of times.

### 5.1 The claim

In a balanced design, the OLS estimate of one factor's effect is identical whether or not the
other factors are included in the model. Equivalently, the estimate of `expand`'s effect is not
biased by, and does not absorb, the effects of `d_state`, `n_layers`, `selective`, or
`discretization`.

### 5.2 Why it holds

Consider two factors, $A$ and $B$, with indicator column sets $X_A$ and $X_B$, plus the intercept.
The cross-product block of $X^\top X$ that couples them is $X_A^\top X_B$. Its $(k,\ell)$ entry
counts how many configurations have both level $k$ of $A$ and level $\ell$ of $B$:

$$
(X_A^\top X_B)_{k\ell} = n_{k\ell} ,
$$

the co-occurrence count. In a balanced full factorial, $n_{k\ell}$ is the **same** for every pair
$(k,\ell)$ — every $A$-level meets every $B$-level equally often. Concretely for this sweep:
`d_state` has 3 levels and `selective` has 2, and each of the $3\times2 = 6$ combinations appears
$72/6 = 12$ times.

When $n_{k\ell}$ is constant across $(k,\ell)$, the coupling block $X_A^\top X_B$ is a constant
matrix. After the intercept accounts for the grand mean — equivalently, after centering each
factor's indicators — the centered coupling block is **zero**. The centered factor subspaces are
orthogonal.

Orthogonal blocks make $X^\top X$ block-diagonal across factors (after removing the intercept
direction). A block-diagonal matrix inverts block by block, so in

$$
\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}
$$

the estimate of factor $A$'s coefficients depends only on $A$'s own block and on
$X_A^\top \mathbf{y}$ — the projection of the response onto $A$'s levels. The other factors'
columns contribute nothing to it. That is exactly the statement that the effect is unconfounded.

### 5.3 What this buys, and what it does not

It buys clean main-effect estimates: each `expand` coefficient is estimated by averaging over all
36 configurations at each `expand` level, and that average is not contaminated by the other knobs
being unevenly distributed, because they are evenly distributed by construction.

It does **not** remove interactions. Balance orthogonalizes the *main-effect* blocks; it says
nothing about whether a factor's effect is constant across another factor's levels. The knob-cost
model is additive (main effects only), so an interaction such as the selectivity crossover in
`findings/220` §5 shows up as model misspecification — it inflates the residual and therefore the
error bars, rather than appearing as its own term. §7.3 returns to this.

**Claude comment:** balance is why the whole sweep was run as a full factorial rather than
one-at-a-time (`findings/210` §2). One-at-a-time gives one delta per knob at a single point in
config space; the balanced factorial gives an effect averaged over 24 or 36 configurations per
level, with the orthogonality above guaranteeing that average is not an artifact of the other
knobs' settings.

Reference: orthogonality and balance in factorial designs, NIST/SEMATECH *e-Handbook of
Statistical Methods* §5.3.3, <https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm>;
Patsy's treatment of how factor columns are built, <https://patsy.readthedocs.io/en/latest/>.

---

## 6. Contrasts: why an adjacent step needs `t_test`

The `d_state` knob has three levels {8, 16, 32}. Treatment coding (`findings/216` §3) picks one
level as the reference — say 8 — and estimates coefficients for the other two **relative to that
reference**: $\hat\beta_{16}$ is the effect of 16 versus 8, and $\hat\beta_{32}$ is the effect of
32 versus 8. The knob-cost table reports adjacent steps, and one of them, `32 -> 16`, is not a
coefficient. It is a **contrast**: a linear combination of coefficients.

### 6.1 A contrast and its estimator

Write a contrast as $L\boldsymbol{\beta}$ for a row vector $L$. For `32 -> 16`,

$$
L\boldsymbol{\beta} = \beta_{32} - \beta_{16},
\qquad
L = (\,0,\;\dots,\;\underbrace{1}_{\beta_{32}},\;\dots,\;\underbrace{-1}_{\beta_{16}},\;\dots,\;0\,).
$$

Its point estimate is $L\hat{\boldsymbol{\beta}} = \hat\beta_{32} - \hat\beta_{16}$. So far this
*is* just subtracting two coefficients. The subtlety is entirely in the uncertainty.

### 6.2 The variance of a contrast

Because $L$ is fixed, the variance of $L\hat{\boldsymbol{\beta}}$ follows from the covariance
matrix of §4 by the same "fixed linear map of a random vector" rule used there:

$$
\operatorname{Var}(L\hat{\boldsymbol{\beta}})
= L \operatorname{Cov}(\hat{\boldsymbol{\beta}}) L^\top
= \sigma^2 \, L (X^\top X)^{-1} L^\top .
$$

Write this out for the two-coefficient contrast, using
$\operatorname{Var}(aU + bV) = a^2\operatorname{Var}(U) + b^2\operatorname{Var}(V) +
2ab\operatorname{Cov}(U,V)$ with $a=1$, $b=-1$:

$$
\operatorname{Var}(\hat\beta_{32} - \hat\beta_{16})
= \operatorname{Var}(\hat\beta_{32})
+ \operatorname{Var}(\hat\beta_{16})
- 2\operatorname{Cov}(\hat\beta_{32}, \hat\beta_{16}) .
$$

The covariance term is the crux. $\hat\beta_{32}$ and $\hat\beta_{16}$ are **both measured
relative to the same reference level 8**, so they share the sampling noise of that reference and
are correlated: $\operatorname{Cov}(\hat\beta_{32}, \hat\beta_{16}) \neq 0$. It is a genuine
off-diagonal entry of $\sigma^2 (X^\top X)^{-1}$.

### 6.3 Why you cannot combine two confidence intervals

A tempting shortcut is to take the individual intervals for $\hat\beta_{32}$ and $\hat\beta_{16}$
and combine their half-widths. That silently assumes the two estimates are independent — it drops
the $-2\operatorname{Cov}(\hat\beta_{32}, \hat\beta_{16})$ term. Since that covariance is
generally positive here (shared reference), dropping it **overstates** the contrast's variance and
yields an interval that is too wide, sometimes badly. The estimate would be right and the
uncertainty wrong.

`t_test` computes $\sigma^2 L(X^\top X)^{-1}L^\top$ directly from the full covariance matrix, so
it carries the covariance term correctly. This is the entire reason `runs/knob_cost.py` evaluates
adjacent-step contrasts through `model.t_test(...)` rather than subtracting coefficients and their
intervals. The mechanics of that call are `findings/216` §5.

### 6.4 Estimating $\sigma^2$ and the degrees of freedom

$\sigma^2$ is estimated from the residuals. With $\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}}$
the fitted values and $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ the residuals,

$$
\hat\sigma^2 = \frac{\mathbf{e}^\top \mathbf{e}}{\,n - p\,}
= \frac{\lVert \mathbf{y} - X\hat{\boldsymbol{\beta}} \rVert^2}{n - p} .
$$

The divisor is the **residual degrees of freedom**: $n$ observations minus $p$ estimated
parameters. For this model $n = 72$ and $p = 1 + 2 + 2 + 1 + 1 + 1 = 8$ (intercept, two `d_state`
contrasts, two `n_layers` contrasts, one each for `expand`, `selective`, `discretization`), so
$n - p = 64$. The divisor $n - p$ rather than $n$ is what makes $\hat\sigma^2$ unbiased; dividing
by $n$ would underestimate the variance because the fitted $\hat{\boldsymbol{\beta}}$ was chosen
to minimize exactly this residual sum.

---

## 7. The confidence interval, and what it does and does not claim

### 7.1 The interval

Under the normal-errors assumption, the standardized contrast follows a $t$-distribution with
$n-p$ degrees of freedom:

$$
\frac{L\hat{\boldsymbol{\beta}} - L\boldsymbol{\beta}}
     {\sqrt{\hat\sigma^2\, L (X^\top X)^{-1} L^\top}}
\;\sim\; t_{\,n-p} .
$$

Inverting this gives the 95% confidence interval the forest plot draws as an error bar:

$$
L\hat{\boldsymbol{\beta}}
\;\pm\;
t_{\,n-p,\,0.975}\,\sqrt{\hat\sigma^2\, L (X^\top X)^{-1} L^\top} .
$$

The $t$ multiplier, not the normal $z = 1.96$, is used because $\sigma^2$ is itself estimated
from the same 64 residual degrees of freedom. At $n-p = 64$ the difference is small
($t_{64,\,0.975} \approx 2.00$) but it is the correct quantity.

Reference: `t_test` documentation,
<https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.t_test.html>;
$t$-based intervals for linear models, Faraway ch. 3 (link in §2).

### 7.2 What the interval formally claims

The interval is a statement about repeated sampling of $\boldsymbol{\varepsilon}$ **with the
design $X$ held fixed**. In words: if the 72 configurations were held exactly as they are and only
the noise $\boldsymbol{\varepsilon}$ were redrawn, 95% of the intervals so constructed would cover
the true contrast. The configurations are treated as fixed; the randomness is entirely in the
error term.

### 7.3 Why that claim is weaker than it looks for this data

Two facts about the sweep make the interval **descriptive rather than inferential**, and both must
be stated wherever these numbers appear.

**The errors are not what the model assumes.** The knob-cost model is additive, but the sweep
contains real interactions — the selectivity crossover in `findings/220` §5 is a large one. An
interaction the additive model cannot represent lands in the residual $\mathbf{e}$. So
$\hat\sigma^2$ is not measuring noise; it is mostly measuring **unmodeled structure** — the
spread of a knob's effect across the other knobs' levels. The interval therefore describes "how
much this effect varies across the rest of the design," which is a dispersion, not a
sampling error.

**There is one seed and one fold.** Every $y_i$ is a single training run (seed 158) evaluated on a
single fold (case 1). The formal claim in §7.2 quantifies resampling of $\boldsymbol{\varepsilon}$,
but the only external, empirical handle on run-to-run variability is $\text{SEED\_SD} = 0.0132$,
the case-1 seed spread measured on the default configuration (`findings/210`, from the three-seed
measurement in `findings/130` §8). The OLS interval and $\text{SEED\_SD}$ answer different
questions: the interval is internal to the additive model's fit; $\text{SEED\_SD}$ is the
observed variation across actual re-runs.

### 7.4 How the charts handle this

The knob-cost figures draw **both**: the OLS interval as the error bar, and the
$\pm\,\text{SEED\_SD}$ band as a shaded reference. A knob's effect is treated as real only if it
clears **both** zero and the seed band. A coefficient whose OLS interval excludes zero but whose
magnitude is inside $\text{SEED\_SD}$ is reported as "not distinguishable from seed noise,"
because the tighter OLS interval reflects only the additive model's internal precision, not the
empirical run-to-run floor.

**Claude comment:** the honest one-sentence summary is that the $p$-values and intervals from this
OLS are best read as effect-size rankings with an uncertainty *scale*, not as hypothesis tests.
Treating a $p < 0.05$ here as a significance verdict would overclaim, because the errors violate
the independence and correct-specification assumptions the $p$-value rests on, and because one
seed and one fold cannot support a population inference. $\text{SEED\_SD}$ is the guard against
that overclaim and belongs on every figure.

---

## 8. Summary of what each figure quantity is

| figure quantity | symbol | source |
| :--- | :--- | :--- |
| knob-step AUC cost | $L\hat{\boldsymbol{\beta}}$ | contrast estimate, §6.1 |
| error bar (95%) | $\pm\,t_{n-p,0.975}\sqrt{\hat\sigma^2 L(X^\top X)^{-1}L^\top}$ | §7.1 |
| seed band | $\pm\,\text{SEED\_SD}$ | empirical, `findings/210` |
| "significant" | interval clears both zero and the seed band | §7.4 |

The library calls that produce $L\hat{\boldsymbol{\beta}}$, the interval, and the $p$-value are in
`findings/216_ols_implementation.md`.

---

## 9. References

All URLs verified against the library versions in use at time of writing; check that the
`statsmodels` and Patsy pages match your installed versions, since these libraries revise
documentation across releases.

- Ordinary least squares, estimation and finite-sample properties. Wikipedia.
  <https://en.wikipedia.org/wiki/Ordinary_least_squares>
- Design matrix. Wikipedia. <https://en.wikipedia.org/wiki/Design_matrix>
- Gauss-Markov theorem. Wikipedia.
  <https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem>
- Faraway, J. *Practical Regression and Anova using R.* Chapters 2–3 cover the normal-equation
  derivation and $t$-based intervals. <https://cran.r-project.org/doc/contrib/Faraway-PRA.pdf>
- NIST/SEMATECH *e-Handbook of Statistical Methods*, §5.3.3 on factorial-design orthogonality.
  <https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm>
- statsmodels `RegressionResults.t_test`.
  <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.t_test.html>
- Patsy documentation (formula and contrast construction).
  <https://patsy.readthedocs.io/en/latest/>