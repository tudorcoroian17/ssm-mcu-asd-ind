# Eval Spec — Data Split, Balancing, and Metrics

**Companion to `01_design_decisions.md`.** Resolves open items from Phase 1 §1.1 (channel policy) and §1.7 (evaluation) based on inspecting the actual ToyCar dataset on disk.

**Status: partially superseded**. Written 2026-08-01, before the IND-only restart. §1 (data source split) and §9 (validation split) describe the CNT-based pipeline and no longer reflect this project — see 01_design_decisions.md §7 and §4b/§9 below for what replaced them. §2 (channel policy), §3 (anomaly counts), §4 (eval balancing), §5 (metrics), §6 (threshold methods), §7 (reporting), and §8 (bootstrap CIs) are unaffected and remain current. CNT-era text is kept as record, not deleted.

---

## 1. Data source split — resolved

The raw dataset splits each case into three subfolders: `AnomalousSound_IND`, `NormalSound_IND`, `NormalSound_CNT`, plus a top-level `EnvironmentalNoise_CNT` (background noise, `caseAll`, not tied to any single case).

**Verified on disk (case1, representative):** `NormalSound_CNT` contains ~36 hours of unique continuous audio per case (4 channels × ~9h each); `NormalSound_IND` covers only ~4.1 hours per case as fixed 11.0s clips. The ratio (~8.7×) rules out "IND is CNT fully chopped up" — IND is a curated subsample, not an exhaustive segmentation of CNT.

**Resolution — SUPERSEDED.** `CNT` is dropped entirely. Silence contamination in the `CNT` training pool made the original volume argument moot; see `01_design_decisions.md` §7.2 for the contamination finding and §7.3 for the decisive `IND`-only experiment (train/val skill gap collapsing from 0.369 to 0.003). Current pool assignment:

| Source | Role | Why |
| :--- | :--- | :--- |
| `NormalSound_IND`, training cases | Training pool (80%) and validation pool (20%), split by `data.val_split_seed` | Every frame is real signal by construction; no filtering needed |
| `AnomalousSound_IND`, training cases | Threshold-calibration pool (§6 method 3) | Never used for gradient updates or early stopping |
| `NormalSound_IND` + `AnomalousSound_IND`, held-out case | Evaluation pool, balanced 1:1 (§4) | Apples-to-apples AUC scoring per DCASE convention |
| `NormalSound_CNT` | Excluded entirely | Silence contamination, unevenly distributed across cases |
| `EnvironmentalNoise_CNT` | Excluded | Not machine sound; park for a later noise-robustness check |

The leakage argument in the original text still holds and is now simpler: in any LOSO fold, the held-out case contributes nothing to training, validation, or calibration — only to test.

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
4. **Cross-fold skill scores are not comparable in absolute terms**. Per-fold mse_persistence depends on which cases are in the normalization-stats pool, and that alone produced a 3.1× spread in the CNT era (findings/110). Under IND the spread is far smaller (ratio 0.0666–0.0756 across folds, runs/baselines/toycar_all_folds_k2_baselines.json), but the mechanism is unchanged. Report per-fold skill individually; never pool it into one cross-fold average without this caveat attached. This is a second, independent reason alongside the small-N caveat. 
5. **Phase 2's ablation deltas partly reflect this normalization effect**. When a config is compared against the default across folds, part of the fold-to-fold delta comes from differing baseline scaling rather than from the ablated axis. Note it explicitly in ablation tables.

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

**Resolution — REVISED for IND-only.** The original argument depended on training drawing exclusively from `CNT`, leaving training-case `IND` free. That no longer holds. The pools are now carved out of training-case `IND` directly, as implemented in `src/data/folds.py`:

| Pool | Source | Used For |
| :--- | :--- | :--- |
| **Train** | Training-case `IND` normals, 80% split by `val_split_seed` | Gradient updates (next-frame prediction loss) |
| **Validation (normal)** | Training-case `IND` normals, remaining 20% | Early stopping; percentile and chi-square threshold calibration (§6) |
| **Validation (anomaly)** | Training-case `IND` anomalies, all | Labelled-anomaly threshold calibration only (§6 method 3) |
| **Test** | Held-out case `IND`, normal + anomaly, 1:1 balanced (§4) | Final reported metrics |

Early stopping uses validation normals only — anomalies are reserved for calibration and never influence training. The held-out case is untouched by every pool above.

**Consequence worth stating explicitly:** Train and validation are now drawn from the same clip population, which is exactly why the train/val skill gap collapsed to 0.003 (`01_design_decisions.md` §7.3). That is a feature for training stability and a limitation for interpretation — a small gap here indicates the model is not overfitting the clip format, not that it generalizes to a new machine. Only the held-out test fold measures that.

---

*Resolved in session, 2026-08-01, while setting up the WSL2/PyTorch environment and running the first real inventory checks against the ToyCar dataset on disk.*

## 10. Threshold Calibration Is Population-Mismatched Under LOSO — A Real Deployment Finding

**Discovery context:** Identified while testing score fusion (`findings/130 §10.1`), though the implications extend far beyond fusion itself.

**Core mechanism:** Z-score calibration constants are computed exclusively on validation normals from the training cases, whereas test clips originate from an unseen machine. Consequently:
- Every held-out clip resides far from the training centroid.
- Calibration constants fail to map scores onto a comparable scale.
- **Symptom that exposed the issue:** On `case4`, `z_max(euclidean, mahalanobis)` should have rescued the two clips suppressed by Mahalanobis (as Euclidean ranked them highest); instead, it replicated the Mahalanobis ranking identically.
- **Rank fusion caveat:** While rank fusion is scale-free and immune to this shift, it is transductive and cannot operate per-clip on an MCU.

**Broad impact across threshold methods:** This discrepancy applies to every thresholding method in **§6**, not merely fusion. Any deployed threshold calibrated on normals from other machines will be systematically miscalibrated on the target machine. This provides empirical confirmation for the open question raised in master doc **§9.1** (*"does the threshold need to be per-machine-calibrated?"*).

**Reporting requirements:**
- When reporting the percentile and chi-square thresholds from **§6**, explicitly state that they are calibrated on training-case normals.
- Note that a real-world deployment would instead calibrate against normal operating audio collected directly from the installed unit post-installation.
- Frame the LOSO operating points as expectedly pessimistic: this represents a meaningful finding regarding domain transferability, not an experimental flaw in the protocol.
