# Eval Spec — Data Split, Balancing, and Metrics

**Companion to `01_design_decisions.md`.** Resolves open items from Phase 1 §1.1 (channel policy) and §1.7 (evaluation) based on inspecting the actual ToyCar dataset on disk.

---

## 1. Data source split — resolved

The raw dataset splits each case into three subfolders: `AnomalousSound_IND`, `NormalSound_IND`, `NormalSound_CNT`, plus a top-level `EnvironmentalNoise_CNT` (background noise, `caseAll`, not tied to any single case).

**Verified on disk (case1, representative):** `NormalSound_CNT` contains ~36 hours of unique continuous audio per case (4 channels × ~9h each); `NormalSound_IND` covers only ~4.1 hours per case as fixed 11.0s clips. The ratio (~8.7×) rules out "IND is CNT fully chopped up" — IND is a curated subsample, not an exhaustive segmentation of CNT.

**Resolution:**

| source | role | why |
|---|---|---|
| `NormalSound_CNT` | **training pool**, sliced into fixed-length windows (window length is a free choice, not tied to 11s) | ~9× more unique normal audio per case than IND; matches the design doc §2 goal of maximizing self-supervised supervision density |
| `NormalSound_IND` + `AnomalousSound_IND` | **evaluation pool**, held-out case only | already fixed-length (11.0s), matched format, clean apples-to-apples AUC scoring per DCASE convention |
| `EnvironmentalNoise_CNT` | excluded from core manifest | not machine-sound, doesn't fit the normal/anomaly binary; park for a possible later noise-robustness check |

**Why this avoids leakage without needing to check CNT/IND overlap directly:** in any LOSO fold, the held-out case's `CNT` is never used for training (only cases 2/3/4's `CNT` is). Whether a case's own `IND` clips happen to be a subsample of its own `CNT` is therefore irrelevant — that case's `CNT` isn't in play when its `IND` is being scored.

## 2. Channel policy — resolved

**Decision: single channel only, `ch1`, applied consistently everywhere** (training windows and eval clips alike).

**Why:** all three target boards (Nucleo-H7S3L8, ESP32, RP2040) are single-microphone. Training or evaluating on multi-channel data (or averaging channels) would not match deployment reality. Using `ch1` alone — never averaging, never treating channels as independent samples — also sidesteps a leakage risk: channels 1–4 of the same physical event are near-duplicate recordings of the same anomaly instant, so treating them as independent train/test samples could put near-identical information on both sides of a split.

## 3. Anomaly counts per case — verified, no small-N caveat needed

From `AnomalousSound_IND`, channel-corrected (÷4):

| case | anomalous events | normal IND events (pre-balancing) |
|---|---|---|
| case1 | ~264 | ~1350 |
| case2 | ~265 | ~1350 |
| case3 | ~265 | ~1350 |
| case4 | ~265 | ~1350 |

All comfortably above the ~50-clip threshold flagged in Phase 1 §1.1 — no small-N confidence-interval caveat required for ToyCar per-fold AUC.

## 4. Eval-set balancing — 1:1, seeded

Natural eval ratio is ~5:1 normal:anomalous per held-out case. **Decision: balance to exactly 1:1** by randomly subsampling `NormalSound_IND` down to match the `AnomalousSound_IND` count for that case, with a **fixed, logged seed** (config-discipline requirement — must be reproducible, same as every other design choice in `01_design_decisions.md`).

This is not required for AUC/pAUC (see §5) — it's required to make precision/recall/F1/accuracy comparable across folds and across ablation configs (see §5). Without it, an apparent metric "drop" between two configs could just be an artifact of one held-out case having a coincidentally different natural class ratio than another.

## 5. Metrics — primary and secondary, with imbalance-robustness notes

**Primary (threshold-free):** AUC, pAUC (p=0.1). Rank-based — computed by sweeping the score threshold across all values, so class imbalance does not bias these. Headline metrics per Phase 1 §1.7, unaffected by the balancing decision in §4.

**Secondary (threshold-dependent):** precision, recall, accuracy, F1. Added to check for drops the rank-based metrics might not surface. Imbalance sensitivity varies by metric:

| metric | imbalance-sensitive? | why |
|---|---|---|
| accuracy | **yes, severely** | at natural ~5:1 ratio, "always predict normal" scores ~84% while detecting nothing (accuracy paradox) |
| precision | **yes** | for a fixed TPR/FPR, precision falls mechanically as anomalies get rarer relative to normals — false positives are drawn from the larger pool |
| F1 | **yes** (inherits from precision) | harmonic mean of precision and recall |
| recall (= TPR) | **no** | depends only on the anomaly population and the threshold; never touches normal-class count |

This is why §4's balancing decision matters specifically for this section: it removes the confound so that a genuine drop in precision/F1/accuracy across folds or configs reflects the model, not a shifting class ratio.

## 6. Threshold methods — compute all four secondary metrics under each

Per Phase 1 §1.7 / master doc §9.1, three threshold methods are implemented regardless:

1. **Percentile-based** (primary)
2. **Chi-square analytic** (cross-check)
3. **Labeled-anomaly-calibrated** (upper bound, flagged in writeup as using label information a real deployment wouldn't have)

**Requirement:** report precision, recall, accuracy, and F1 **separately under each threshold method**, clearly labeled. Do not report a single set of numbers without stating which threshold produced them — the same model can look materially different at different operating points.

## 7. Reporting requirements

- **Per-case, never a single averaged number** (Phase 1 §1.7, master doc §14 small-N caveat convention).
- Report: AUC, pAUC per case/fold (primary); precision/recall/accuracy/F1 per case/fold **per threshold method** (secondary).
- Log the balancing seed and the exact set of normal clips dropped during eval-set subsampling, per fold, into the run directory — same reproducibility bar as everything else in `01_design_decisions.md` and `01_design_decisions.md`'s config-hash discipline.

## 8. Bootstrap confidence intervals — complementary to fold-spread reporting

**Decision: yes, compute bootstrap CIs, per fold, for all six metrics.**

**Not redundant with §7's "mean ± spread across folds"** — that captures generalization variance across *which case got held out*. Bootstrap CIs capture a different source of uncertainty: given a single fold's finite eval set (~264 anomalies + ~264 balanced normals, post-§4), how much would that fold's own metrics vary from finite-sample noise alone. Report both; they answer different questions.

**Primary use case: Phase 2's ablation sweep.** With 300+ one-at-a-time configs, point-estimate differences between a config and the default (e.g. "0.85 vs 0.83 AUC") are not interpretable without knowing whether the gap exceeds sampling noise.

**Method:**

1. Resample the fold's `(score, label)` pairs with replacement.
2. Recompute all six metrics (AUC, pAUC, precision, recall, accuracy, F1) on the resample.
3. Repeat ~5,000–10,000× (cheap at N≈530/fold — milliseconds per resample).
4. Take the 2.5th/97.5th percentiles for a 95% CI, per metric, per fold.

**When comparing two configs on the same fold (the common Phase 2 case): use a *paired* bootstrap.** Draw the same resampled index set for both configs on each iteration, then look at the distribution of the *difference* in each metric, rather than computing two independent CIs and eyeballing overlap. Paired bootstrap has materially more statistical power for "is config A better than config B on this fold" than comparing two unpaired CIs — worth the small amount of extra bookkeeping (same indices, two score arrays) from the start rather than retrofitting later.

**Note for precision/recall/accuracy/F1 specifically:** these still require a threshold choice per §6 — bootstrap the metric under each threshold method separately, same as the point estimates.

## 9. Validation split — early stopping and threshold calibration

**Problem:** §1.5 requires an early-stopping validation signal that never touches the held-out case. §6's threshold methods (percentile, chi-square) need known-normal scores to calibrate against, and method 3 needs labeled anomalies — none of which should come from the test pool itself, or threshold calibration leaks into the metrics it's supposed to produce.

**Resolution: reuse the training cases' own `IND` clips, currently unused in each fold.** Training draws only from `CNT` (§1); evaluation draws only from the held-out case's `IND` (§1). That leaves each training case's `IND` — both `NormalSound_IND` and `AnomalousSound_IND` — untouched, already cached, already fixed-length, and by construction never overlapping the held-out case.

Three pools per fold, no data reused across purposes:

| pool | source | used for |
|---|---|---|
| train | `CNT`, training cases, normal only | gradient updates (next-frame prediction loss) |
| validation | `IND`, training cases, normal + anomaly | early stopping; threshold calibration (§6) |
| test | `IND`, held-out case, normal + anomaly, 1:1 balanced (§4) | final reported metrics |

Validation-normal feeds the percentile and chi-square threshold methods; validation-anomaly feeds the labeled-anomaly-calibrated method. No new data collection or caching required — this is a purpose assignment on data already inventoried.

---

*Resolved in session, 2026-08-01, while setting up the WSL2/PyTorch environment and running the first real inventory checks against the ToyCar dataset on disk.*
