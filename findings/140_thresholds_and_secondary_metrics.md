# Findings 140: Threshold Methods and Secondary Metrics

**Status:** Phase 1 §1.7 deferred item, now complete.

**Scripts:** `src/eval/thresholds.py` (all threshold methods, `run_case` / `run_case_advanced`), `src/data/folds.py` (`calib_normal` pool), `src/eval/embeddings.py` (`val_anomaly_emb`, `calib_normal_emb`), `checks/metrics/threshold_transfer.py` (population-shift diagnostic).

## 1. Purpose

AUC and pAUC (`findings/130`) measure ranking quality only. Neither answers the deployment question: at a real, fixed operating threshold, what precision, recall, accuracy, and F1 does the detector achieve? This document covers that gap: three baseline threshold methods per `01_eval_spec.md` §6, the cross-machine calibration failure they exposed, its fix, and five additional threshold-estimation methods evaluated against the fix.

*Claude comment: this finding changed twice over the course of building it — first the baseline methods looked badly broken, then a fix made them look very good, then testing the fix's robustness surfaced a real but narrower failure mode. All three stages are reported below rather than only the final state, because the middle stage (the cross-machine mismatch) is itself the most reportable result in this document.*

## 2. Baseline threshold methods

Per `01_eval_spec.md` §6, three methods:

1. **Percentile** — unsupervised, threshold set at a percentile of calibration-normal scores. Primary method.
2. **Chi-square** — analytic, Mahalanobis only. Squared Mahalanobis distance follows a chi-square distribution under a Gaussian-cluster assumption; `score_mahalanobis()` returns the square root, so the threshold is `sqrt(chi2.ppf(1-alpha, df=d_model))`. Cross-check, not a general-purpose method — the derivation does not extend to Euclidean or kNN distances.
3. **Calibrated** — sweeps threshold against validation normals + validation anomalies for best F1. Uses label information; described in the master doc and `01_eval_spec.md` §6 as an upper bound.

Secondary metrics (precision, recall, accuracy, F1) are reported per threshold method, never pooled into one unlabeled set.

## 3. First result: threshold calibration is population-mismatched under LOSO

Calibrating on validation-case normals (training-machine data) and testing on the held-out machine produced near-total failure. On case1: 12 of 27 pooling × head × method configurations classified **every** test clip as anomalous (precision 0.50, recall 1.00). Six more scored below chance on precision.

**Diagnosis.** `checks/metrics/threshold_transfer.py` computed an oracle F1 per configuration — the best F1 achievable at any threshold, using test labels directly. Oracle F1 averaged 0.87–0.97 across every head and fold. **The ranking was fine everywhere; the entire loss was threshold placement**, confirmed by the labelled-anomaly `calibrated` method performing no better than `percentile` in most configurations — if the problem were which percentile to pick, calibrated selection would have found a better one, and it consistently did not.

**Mechanism, two distinct causes:**

- **kNN under `mean` pooling** measures distance to a near-twin. Validation normals are a random 20% split of the same clip pool as training — same machine, same session. Every validation clip has a near-duplicate in the training reference set, giving artificially small kNN distances (~0.15) that don't represent what an unseen-machine clip scores (~0.9). Mean scale-shift ratio (test-normal minimum ÷ validation-normal median), averaged across folds: 4.10× for `knn_full`, 2.73× for `knn_clustered_16`.
- **Mahalanobis under `mean` pooling** amplifies exactly what it is chosen for. Whitening divides out high-variance training directions; the between-machine offset direction is high-variance within the training pool (which is why Mahalanobis handles the loudness confound well) but low-variance from an unseen fourth machine's perspective, so its distance gets multiplied up rather than down. Mean scale-shift ratio: 3.26×, with complete population separation (validation-normal max below test-normal min) in 3 of 4 folds.
- **`max` pooling and `euclidean` distance show almost no scale shift** (0.95–1.09× across every head). `max` pooling's embeddings are loose enough that within-population spread swamps the between-machine offset — the flip side of `max`'s known seed-instability and lower AUC (`findings/130` §2).

**Ranking inversion.** `findings/130` §11's deployment recommendation (`mean` + `knn_clustered_16`, chosen on AUC 0.9599) went degenerate in **all four folds** under this calibration — flags every clip as anomalous, every fold. The heads that scored worst on AUC (`euclidean` under `max` and `concat_mean_last` pooling) were the only ones producing a working detector from this calibration, purely because their scale happens not to drift across machines. Achieved (best of percentile/calibrated) vs. oracle F1, averaged across folds:

| pooling / head | achieved F1 | oracle F1 | loss | degenerate folds |
|---|---|---|---|---|
| max / euclidean | 0.841 | 0.902 | 0.062 | 0/4 |
| concat / euclidean | 0.823 | 0.875 | 0.053 | 0/4 |
| mean / euclidean | 0.810 | 0.938 | 0.128 | 2/4 |
| max / mahalanobis | 0.746 | 0.923 | 0.177 | 2/4 |
| max / knn_full | 0.746 | 0.906 | 0.161 | 1/4 |
| max / knn_clustered_16 | 0.730 | 0.907 | 0.178 | 3/4 |
| mean / mahalanobis | 0.667 | 0.947 | 0.280 | **4/4** |
| mean / knn_full | 0.665 | 0.974 | 0.309 | **4/4** |
| **mean / knn_clustered_16** | **0.661** | 0.964 | 0.302 | **4/4** |

**Chi-square cross-check validated independently.** The analytic threshold — computed from `d_model` alone, never touching the data — landed within 2.7–4.6% of the empirical validation-normal p95 for `mean`-pooling Mahalanobis across all four folds. Confirms the Gaussian-cluster assumption behind the covariance estimate holds for this pooling mode. Diverged 8–24% for `concat_mean_last`, consistent with its already-documented ill-conditioning (`findings/130` §2.4).

## 4. Fix: same-machine calibration

`01_eval_spec.md` §4's 1:1 test balancing discards roughly 1,085 held-out-case normal clips per fold that are never scored. This pool is same-machine and leakage-free — it simulates a real deployment recording its own normal operation after installation, and never touches the test set. Added as `calib_normal` in `folds.py`, threaded through `embeddings.py` and `thresholds.py`.

**A transcription bug was found and fixed during this work:** the initial implementation reported identical thresholds for p95, p99, and p99.5 (matching to sixteen significant figures across all 48 configurations), traced to the percentile argument not being forwarded into the `np.percentile` call — every call silently used the function's default. Fixed by passing the loop variable explicitly.

**Result, post-fix, at p95:**

| pooling / head | F1 (case1–4) | mean | factory F1 | oracle F1 | gap to oracle closed |
|---|---|---|---|---|---|
| **mean / knn_full** | 0.931 0.953 0.969 0.962 | **0.953** | 0.665 | 0.974 | 97% |
| **mean / knn_clustered_16** | 0.892 0.941 0.978 0.963 | **0.944** | 0.661 | 0.964 | 93% |
| mean / mahalanobis | 0.787 0.989 0.972 0.974 | 0.930 | 0.667 | 0.947 | 94% |
| max / mahalanobis | 0.787 0.912 0.962 0.981 | 0.911 | 0.746 | 0.923 | 93% |
| mean / euclidean | 0.770 0.869 0.957 0.969 | 0.891 | 0.810 | 0.938 | 63% |
| max / knn_clustered_16 | 0.736 0.880 0.960 0.974 | 0.887 | 0.730 | 0.907 | 89% |
| max / knn_full | 0.740 0.866 0.958 0.969 | 0.883 | 0.746 | 0.906 | 86% |
| max / euclidean | 0.686 0.894 0.932 0.972 | 0.871 | 0.841 | 0.902 | 97% |
| concat / knn_full | 0.500 0.813 0.298 0.980 | 0.648 | — | 0.912 | 4% |
| concat / knn_clustered_16 | 0.718 0.594 0.298 0.965 | 0.644 | — | 0.885 | 3% |
| concat / euclidean | 0.493 0.651 0.316 0.972 | 0.608 | — | 0.875 | −80% |
| concat / mahalanobis | 0.422 0.150 0.293 0.976 | 0.460 | — | 0.812 | −59% |

Same-machine calibration closes 86–97% of the oracle gap on every `mean`/`max` head, versus the 3–20% typical of cross-machine calibration. Answers master doc §9.1's open question ("does the threshold need to be per-machine-calibrated?") directly: yes.

**Practical consequence: head choice barely matters once calibration is correct.** `mean/knn_clustered_16` (0.944, 4 KB reference set) sits within 0.009 F1 of `mean/knn_full` (0.953, 810 KB). Factory calibration made the 4 KB head look unusable; same-machine calibration shows it losing almost nothing to the head 200× its size.

## 5. Percentile choice: p95 is the correct default, not a per-fold pick

| pooling / head | F1 @ p95 | F1 @ p99 | F1 @ p99.5 |
|---|---|---|---|
| mean / mahalanobis | 0.787 0.989 0.972 0.974 | 0.718 0.996 0.985 0.994 | 0.704 0.998 0.992 0.996 |
| mean / euclidean | 0.770 0.869 0.957 0.969 | 0.626 0.881 0.951 0.989 | **0.367** 0.881 **0.171** 0.994 |

Two competing effects. On the clean-separation folds (2–4), raising the percentile buys precision at no recall cost — every head improves monotonically toward p99.5. On case1 (the fold `findings/130` §9.3 already documented as failing diffusely), every head degrades as the percentile rises — there is less headroom before the threshold starts cutting into anomaly territory. `mean/euclidean` at p99.5 additionally collapses on case3 (F1 0.171) despite that fold otherwise being clean, for a different reason: with ~1,085 calibration clips, p99.5 is defined by roughly the top 5 order statistics, and a single unusual clip in that handful swings the threshold substantially.

**Reporting rule:** commit to one percentile in advance and apply it identically across folds. Selecting whichever percentile scores best per fold, after seeing test-set F1, is the same category of leakage as the `calibrated` method's label access in Section 3 — a real deployment cannot make that choice a priori. p95 is the value reported throughout this document for that reason, not because it happened to win most often.

## 6. Advanced threshold methods

Five additional estimators, organized around a false-alarm budget (`target_far`) rather than an arbitrary percentile:

- **False Alarm Budgeting** — `target_far` (e.g., 0.05 / 0.01 / 0.001) as the parameter the other methods solve against, converted from a deployment duty cycle when one is available.
- **Extreme Value Theory (POT/GPD)** — fits a Generalized Pareto distribution to the top 10% of calibration exceedances (Coles 2001; Siffer et al. 2017), extrapolates to the target quantile.
- **Parametric fitting** — Gamma / Log-Normal / Weibull / Chi, MLE-fit, selected by AIC. `chi` is included as the Euclidean-distance analogue of the Mahalanobis chi-square precedent (Section 3).
- **Kernel Density Estimation** — Gaussian KDE on calibration normals, quantile taken from a large resample.
- **MAD / IQR** — robust dispersion metrics (Iglewicz & Hoaglin 1993; Tukey fences), not quantile estimators — use only median and spread, 25–50% breakdown points.

**At p95-equivalent, all four quantile methods agree within 0.005 F1** (parametric 0.9113, KDE 0.9104, percentile 0.9088, EVT 0.9068, mean across mean/max pooling and all heads). Expected: at ~1,085 calibration clips, p95 is set by ~54 order statistics, already well-estimated by the simplest method.

**At p99-equivalent, parametric fitting produces a strictly better operating point** — not a tradeoff — on the two deployment-relevant heads:

| pooling / head | p95 (established default) | parametric, p99-equivalent | delta |
|---|---|---|---|
| mean / knn_full | 0.9534 | **0.9624** | +0.0090, at 1/5 the false-alarm rate |
| mean / knn_clustered_16 | 0.9436 | **0.9596** | +0.0160, at 1/5 the false-alarm rate |

Every other head loses ground at p99-equivalent (−0.02 to −0.08), so this is specific to these two configurations, not a general "always push to p99" result.

**At p999-equivalent, EVT and KDE both fail on a meaningful fraction of configurations.** 11 of 96 fold × head × pooling combinations produced precision = recall = 0, accuracy = 0.5 exactly — the signature of a threshold placed above every test clip, normal and anomaly alike. Reconstructed threshold values confirm direct blowup (e.g., `concat/mahalanobis`, case1: threshold 20,712 against test-normal scores of 25–60). **Parametric fitting never collapses this way** (worst case F1 = 0.299) because it fits its shape from the full ~1,085-point calibration sample rather than the ~108-point tail window EVT and KDE restrict themselves to. **Recommendation: do not report a p999-equivalent operating point from any method at this calibration-pool size.** The instability is a property of the sample size, not of any one estimator.

**MAD is the standout specifically on `concat_mean_last`** (0.686–0.863, versus 0.608–0.648 for percentile p95) — expected, since MAD's median/MAD statistics are immune to the heavy right tail already documented for that pooling mode (Section 7). Does not change the disqualification below; explains the mechanism rather than rescuing the candidate.

`IQR` scored consistently mediocre (~0.79 mean, below both MAD and p95) and isn't reported further.

## 7. EVT shape parameter as a diagnostic

The GPD shape parameter — fitted per fold, per pooling, per head, from the top ~108 calibration exceedances — was investigated as a predictor of p999-equivalent collapse, across all four folds (48 configurations).

**What holds, checked against all 48 points:** shape ≥ 0.676 predicts collapse with zero exceptions. Every configuration above that line produced F1 near zero; none escaped.

**What does not hold, and was initially overstated:** the reverse claim — that shape below some threshold predicts survival — does not hold. Two configurations collapsed at near-zero or negative shape:

| config, fold | shape | F1 |
|---|---|---|
| concat / knn_clustered_16, case3 | −0.033 | 0.022 |
| concat / knn_full, case3 | −0.006 | 0.030 |

*Claude comment: I stated a clean monotonic relationship after seeing case1 alone (12 points), and it did not survive the other 36. Recording that correction here rather than only the corrected version, because the failure mode — generalizing a diagnostic rule from one fold — is exactly the kind of thing this project has otherwise been careful about (see the seed-sweep and multi-fold discipline throughout `findings/130`).*

The middle zone (shape roughly −0.05 to +0.65) is not predictable from shape alone: `mean/knn_full` case3 (shape 0.366) scores F1 0.994; `max/knn_full` case3 (shape 0.091, much milder) scores F1 0.279. Shape is a one-directional flag for danger above ~0.67, not a general-purpose score.

**Pooling-level pattern.** Mean shape by pooling, per fold:

| pooling | case1 | case2 | case3 | case4 |
|---|---|---|---|---|
| max | 0.402 | 0.133 | 0.109 | −0.033 |
| mean | 0.265 | −0.183 | 0.317 | −0.179 |
| concat | 1.150 | −0.210 | 0.191 | −0.126 |

`max` pooling's average shape descends roughly monotonically with fold difficulty (case1 hardest → case4 easiest, matching the AUC-based ranking in `findings/130`). `mean` and `concat` do not — both put case3 above case1, despite case3 being the easier fold by every AUC and oracle-F1 measure so far. Unexplained; open item below.

**Deployment head, all four folds:** `mean/knn_clustered_16` shape is −0.078, −0.125, +0.248, −0.181. Case3 is its one positive-shape fold and simultaneously its best F1 (0.994 at p999-equivalent) of the four — a direct illustration that moderate positive shape does not predict failure in practice, even for the specific head this project depends on.

**Parametric family cross-validation.** `mean/mahalanobis` selects the `chi` family by AIC in 3 of 4 folds (case1, case3, case4; `lognorm` in case2). Given `chi` is the raw-distance analogue of the squared-Mahalanobis chi-square relationship already confirmed empirically in Section 3, this is a second, independent method arriving at the same distributional conclusion for the same head.

## 8. `concat_mean_last`: consolidated disqualification record

Independent evidence accumulated across this document and `findings/130`:

1. AUC unstable across seeds (`findings/130` §2).
2. Sub-chance AUC (0.368) for `mahalanobis` on case2 — ranking inverted, not just noisy.
3. Covariance condition numbers 1.9×10⁷–1.3×10⁸ (`findings/130` §2.4), against 1.5–7.5×10⁴ for `mean`.
4. Chi-square analytic threshold disagrees with empirical p95 by 8–24%, versus 3–5% for `mean` (Section 3).
5. Factory-calibrated thresholds degenerate on this pooling mode as often as on any other; same-machine calibration does not rescue it — mean F1 0.46–0.65 against oracle 0.81–0.91 (Section 4).
6. Higher percentiles are actively dangerous, not just unhelpful: F1 as low as 0.09–0.26 at p99/p99.5 (Section 6).
7. EVT shape parameter both the most extreme observed (up to 1.569, case1) and the most volatile across folds (1.150 average in case1 down to −0.210 in case2) of any pooling mode — and produces two collapses at shapes that predict safety everywhere else (Section 7).

**Recommendation:** drop `concat_mean_last` from the evaluation grid for any future work. Seven independent diagnostics agree; none partially rescue it.

## 9. Deployment recommendation

`mean` pooling + `knn_clustered_16`, thresholded at the 95th percentile of the held-out unit's own recorded normal operation (`calib_normal`, not training-case validation data). Mean F1 0.944 across folds, within 0.01 of the largest reference-set variant at 1/200th the memory footprint. Consider `parametric_p99_equiv` in place of plain percentile — it improves this exact configuration to 0.9596 at a stricter false-alarm rate, with no configuration-specific tuning beyond the existing AIC family selection.

Supersedes the AUC-only recommendation in `findings/130` §11, which was correct on ranking grounds but silent on the calibration question this document answers.

## 10. Open items

- **Two case3 `concat_mean_last` collapses at near-zero shape (Section 7) are unexplained.** Working hypothesis: the GPD shape parameter describes the tail *above* `u0` (the 90th-percentile cutoff) but says nothing about where `u0` itself sits relative to the anomaly distribution — `concat_mean_last`'s known erratic calibration behavior could place `u0` too high independent of the local tail shape. Check: pull `u0` for these two configurations against case3's known concat test-anomaly range.
- **Why `mean` and `concat` pooling's average GPD shape does not track fold difficulty, while `max`'s does** (Section 7). No mechanism proposed yet.
- **Cross-machine calibration was never directly re-tested with the five advanced methods.** Section 6's results all use `calib_normal` (same-machine). Worth one confirmatory run against `val_normal` to verify the advanced methods fail there too, rather than assuming it from the Section 3 mechanism.
- **`threshold_diagnostics.json` does not record `held_out_case` internally.** Each file is scoped correctly by its directory path at write time, but a file in isolation (e.g., handed off or archived separately) can't be identified without reconstructing a threshold from its parameters and matching against known results, as was necessary while assembling this document. One-line fix: add `held_out_case` to each diagnostic entry.