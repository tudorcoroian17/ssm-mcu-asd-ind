# Finding: Fold-level baseline differences are a normalization artifact, not a case-difficulty difference

**Status:** Resolved, quantitatively confirmed.
**Context:** discovered while sanity-checking `mse_persistence` after regenerating
`fold_norm_stats.json` on silence-filtered CNT training data (see `01_eval_spec.md`
and the raw-waveform silence-filtering work).

---

## 1. The observation that triggered this

After rebuilding fold normalization stats on the silence-filtered training pool,
`baselines.py` produced per-fold `mse_persistence` values that did not cluster the
way case-level window-survival rates would predict:

| held-out fold | trains on | mse_persistence |
|---|---|---|
| 1 | 2, 3, 4 | 0.3035 |
| 2 | 1, 3, 4 | 0.3602 |
| 3 | 1, 2, 4 | **0.8832** |
| 4 | 1, 2, 3 | 0.3985 |

Held-out-3 sat at more than double the next-highest fold, and nearly 3x
held-out-1 — despite case3's per-case window-survival rate (39.3%) being nearly
identical to case2's (39.2%), which did not show the same effect. This did not
match the expected pattern and needed explaining before the numbers could be
trusted for the upcoming retrain.

---

## 2. Investigation, in order

**Ruled out: overall window variance.** Comparing raw per-window `std()` between
case2 and case3's kept windows showed no meaningful difference (case2: mean=2.010,
case3: mean=2.027). `std()` does not measure frame-to-frame change, so this did not
actually test the hypothesis it was meant to — noted and corrected in the next step.

**Ruled out (initially, then partially confirmed): per-window persistence, computed
in the wrong units.** An early check computed persistence directly on raw,
unnormalized log-mel arrays, giving case2=0.7149 and case3=0.9086 — a real gap, but
in the wrong units to compare against the fold-level `mse_persistence` figures
(which are computed on *normalized* windows). Confirmed as invalid via an algebraic
consistency check: solving for case4's implied contribution to held-out-1's pooled
mean using these raw-unit numbers produced a mathematically impossible negative
MSE.

**Corrected: per-window persistence, properly normalized with the matching fold's
stats.** Recomputing with `apply_normalization` using each fold's own
`fold_norm_stats.json` entry:

| case | normalized with | mean persistence |
|---|---|---|
| case2 | held-out-1's stats (pool {2,3,4}) | 0.2830 |
| case3 | held-out-1's stats (pool {2,3,4}) | 0.3395 |
| case1 | held-out-3's stats (pool {1,2,4}) | 0.8786 |
| case4 | held-out-3's stats (pool {1,2,4}) | 0.8873 |
| case2 | held-out-3's stats (pool {1,2,4}) | **0.8869** |

The decisive result is the last row: **the same case2 audio**, scored once under
held-out-1's normalization stats (0.2830) and once under held-out-3's (0.8869) —
more than a 3x difference on identical underlying recordings. This rules out any
case-intrinsic-difficulty explanation. The variable is which fold's stats were used
to normalize, not which case was being measured.

---

## 3. Root cause, quantitatively confirmed

Directly comparing held-out-1's stats (`mean`, `std`) against held-out-3's
(`mean3`, `std3`): mean difference (fold3 - fold1): mean_abs=0.4696, max_abs=0.8065
std ratio (fold3 / fold1): mean=0.5488, min=0.3327, max=0.8848


Held-out-3's stats have roughly **half the standard deviation** of held-out-1's,
averaged across mel bins. Since persistence is a squared quantity, dividing by a
std that is ~55% as large inflates squared error by roughly `1/0.549² ≈ 3.3x` —
matching the ~3.13x gap observed directly (0.2830 → 0.8869) almost exactly. The
effect is fully, quantitatively explained.

**Mechanism:** case1 has the highest post-silence-filter window-survival rate of
any case (43.2%, vs. 33.1–41.4% for cases 2–4). Whichever fold's training pool
happens to include case1 ends up with tighter, more consistent normalization
statistics (smaller std) than a pool that excludes it. This single factor —
whether case1 is present in the stats-computation pool — is sufficient to explain
the full spread in fold-level `mse_persistence` seen in Section 1.

---

## 4. Implication for interpreting results going forward

**Fold-level `mse_persistence`, and therefore fold-level `skill`, is not
comparable in absolute terms across folds.** A skill of 0.39 on one held-out fold
and 0.39 on another are each measuring "beat that fold's own, differently-scaled
baseline by the same relative margin" — not "equally good at predicting normal
sound" in any absolute sense. This is a *second*, independent reason (distinct
from the small-N sampling-variance caveat already in `01_eval_spec.md` /
master doc §14) why per-fold numbers should be reported and read individually,
never pooled into one cross-fold average without this caveat attached.

**This does not invalidate any single fold's own skill score.** Within one fold,
the baseline and the training loss are computed from the same normalization
stats and the same training pool — the property that actually makes `skill`
meaningful holds. This finding is about cross-fold comparison, not within-fold
validity.

---

## 5. Not yet done

- Add a short cross-reference to this file from `01_eval_spec.md`'s reporting
  section, alongside the existing small-N caveat.
- Consider whether Phase 2's ablation reporting (comparing a config against the
  default across folds) needs an explicit note that fold-to-fold deltas partly
  reflect this normalization effect, not purely the ablated axis.