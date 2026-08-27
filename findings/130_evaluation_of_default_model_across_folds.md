# Default model evaluated across all four LOSO folds

**Status:** Partially resolved. The headline results are validated. One question — why fold
difficulty varies the way it does — remains open and requires a seed sweep to answer.

**Context:** First full evaluation of the `ssm-mcu-asd-ind` pipeline. The default configuration
(`configs/default.yaml`) was trained on all four leave-one-subject-out folds, then scored with
three pooling modes against four distance heads, for 12 combinations per fold.

**Scope note:** every number in this document comes from a single training run per fold at
seed 42. No result here has been replicated across seeds. Section 8 explains why that matters.

---

## 1. What was run

Training used plain Adam with no gradient clipping, no learning-rate schedule, and no weight
decay, matching the configuration that produced the reference numbers in
`01_design_decisions.md` §7.3. Each fold trains on the three non-held-out cases' IND normal
clips, split 80/20 into train and validation. The held-out case supplies a 1:1 balanced test
set of roughly 265 normal and 265 anomalous clips.

Held-out case 1 reproduced the historical reference almost exactly: train skill 0.5482, val
skill 0.5452, against the recorded 0.548 and 0.545. The loss curve descended smoothly across 49
epochs with no oscillation, confirming that the `val_mse` instability documented in
`01_design_decisions.md` §7.2 does not occur on IND data.

Scripts used in this investigation:

| Script | Purpose |
|---|---|
| `checks/metrics/eval_auc_pauc.py` | Pooling × distance-head sweep, 27 charts per case |
| `checks/metrics/onset_tail_contribution.py` | Protocol-memorization diagnostic |
| `checks/metrics/coefficient_of_variation.py` | Score and raw-feature dispersion per case |
| `checks/smoke/embedding_saturation.py` | Embedding-space geometry of extreme-scoring clips |

---

## 2. Results: AUC and pAUC across all folds

### Held-out case 1

| pooling | distance head | AUC | pAUC (p=0.1) |
|---|---|---|---|
| mean | euclidean | 0.9338 | 0.7249 |
| mean | mahalanobis | 0.8797 | 0.8606 |
| mean | knn_full | 0.9524 | 0.8697 |
| mean | knn_clustered_16 | 0.8965 | 0.9106 |
| max | euclidean | 0.9637 | 0.8460 |
| max | mahalanobis | 0.9602 | 0.8358 |
| max | knn_full | 0.9639 | 0.8502 |
| **max** | **knn_clustered_16** | **0.9647** | **0.8779** |
| concat_mean_last | euclidean | 0.9400 | 0.7431 |
| concat_mean_last | mahalanobis | 0.8746 | 0.8598 |
| concat_mean_last | knn_full | 0.9582 | 0.8791 |
| concat_mean_last | knn_clustered_16 | 0.9036 | 0.9101 |

### Held-out case 2

| pooling | distance head | AUC | pAUC (p=0.1) |
|---|---|---|---|
| **mean** | **mahalanobis** | **0.9962** | **0.9980** |
| mean | euclidean | 0.8707 | 0.8830 |
| mean | knn_full | 0.9886 | 0.9833 |
| mean | knn_clustered_16 | 0.9896 | 0.9810 |
| max | euclidean | 0.9524 | 0.8594 |
| max | mahalanobis | 0.9559 | 0.8690 |
| max | knn_full | 0.9550 | 0.8638 |
| max | knn_clustered_16 | 0.9553 | 0.8571 |
| concat_mean_last | euclidean | 0.3057 | 0.5345 |
| concat_mean_last | mahalanobis | 0.2503 | 0.5165 |
| concat_mean_last | knn_full | 0.8463 | 0.6988 |
| concat_mean_last | knn_clustered_16 | 0.7250 | 0.5661 |

### Held-out case 3

| pooling | distance head | AUC | pAUC (p=0.1) |
|---|---|---|---|
| **mean** | **euclidean** | **0.9923** | 0.9597 |
| mean | mahalanobis | 0.9888 | 0.9940 |
| mean | knn_full | 0.9887 | 0.9940 |
| mean | knn_clustered_16 | 0.9887 | 0.9939 |
| max | euclidean | 0.9670 | 0.8455 |
| max | mahalanobis | 0.9804 | 0.9012 |
| max | knn_full | 0.9779 | 0.8868 |
| max | knn_clustered_16 | 0.9823 | 0.9094 |
| concat_mean_last | euclidean | 0.9918 | 0.9569 |
| concat_mean_last | mahalanobis | 0.9710 | 0.9003 |
| concat_mean_last | knn_full | 0.9871 | 0.9857 |
| concat_mean_last | knn_clustered_16 | 0.9868 | 0.9838 |

### Held-out case 4

| pooling | distance head | AUC | pAUC (p=0.1) |
|---|---|---|---|
| **mean** | **euclidean** | **1.0000** | **1.0000** |
| mean | mahalanobis | 0.9925 | 0.9960 |
| mean | knn_full | 0.9925 | 0.9960 |
| mean | knn_clustered_16 | 0.9925 | 0.9960 |
| max | euclidean | 0.9999 | 0.9993 |
| **max** | **mahalanobis** | **1.0000** | **1.0000** |
| max | knn_full | 0.9973 | 0.9980 |
| max | knn_clustered_16 | 0.9963 | 0.9960 |
| **concat_mean_last** | **euclidean** | **1.0000** | **1.0000** |
| concat_mean_last | mahalanobis | 0.9925 | 0.9960 |
| concat_mean_last | knn_full | 0.9925 | 0.9960 |
| concat_mean_last | knn_clustered_16 | 0.9925 | 0.9960 |

### Covariance conditioning

The Mahalanobis head reports the covariance condition number and warns above 1e6. All four
`concat_mean_last` folds exceeded the threshold: case1 at 7.30e7, case2 at 2.28e7, case3 at
8.73e6, and case4 at 9.37e6. No `mean` or `max` fold triggered the warning.

This addresses the near-singularity concern raised in
`120_evaluation_of_pooling_methods.md` §7. The concern is real, and it is specific to
`concat_mean_last`, whose 128-dimensional embedding pairs two correlated halves derived from
the same sequence.

---

## 3. Why the results prompted an investigation

Three folds produce AUC above 0.99, and case4 reaches 1.0000 under three separate
configurations. Results this high warrant checking for a shortcut before accepting them.

The initial concern was that fold difficulty appeared to increase with case number. Under
`mean` pooling with Euclidean distance, the folds read 0.9338, 0.8707, 0.9923, and 1.0000.
Because the folds were trained in ascending case order, case identity and training order are
perfectly confounded in this run.

**That framing was imprecise.** The sequence is not monotonic — case2 scores lowest, below
case1. Section 7 revisits the framing after the investigations.

---

## 4. Ruled out: protocol memorization

`01_design_decisions.md` §7.4 flagged a risk specific to IND clips. Every clip shares a
near-identical structure: roughly 0.816s of onset silence, a motor-running section, and roughly
0.45s of tail silence. A model could learn this fixed timing instead of the engine sound.

The diagnostic scores each fold four ways: on the full clip, on the onset region alone (first 40
frames), on the tail alone (last 25 frames), and on the middle region with both brackets removed
(279 frames).

| case | full_clip | onset_only | tail_only | middle_only |
|---|---|---|---|---|
| 1 | 0.9338 | 0.8338 | 0.2938 | 0.9438 |
| 2 | 0.8707 | 0.7083 | 0.0969 | 0.9843 |
| 3 | 0.9923 | 0.4840 | 0.6154 | 0.9923 |
| 4 | 1.0000 | 0.7756 | 0.2439 | 1.0000 |

**`middle_only` matches or exceeds `full_clip` in every fold.** Removing the silence brackets
does not degrade performance — for case2 it improves AUC substantially, from 0.8707 to 0.9843.
If the model depended on clip timing, removing that timing would hurt. It doesn't.

The `onset_only` scores are non-trivial in three folds, which is consistent with motor startup
transients carrying genuine fault-correlated information. That is a second source of real
signal, not a leak.

`tail_only` inverts in three folds, scoring below 0.5. The tail is 25 of 344 frames, about 7% of
a mean-pooled embedding, so its influence on the full-clip score is small. The inversion is
unexplained but low-impact.

**Conclusion: the §7.4 protocol-memorization risk is resolved. The high AUC values do not come
from the silence brackets.**

---

## 5. Ruled out: training-loop state leakage

If `train_one_fold` reused model weights or optimizer state between folds, later folds would
start warm and appear artificially easy — in exactly the observed order.

A direct read of `src/train.py` rules this out. Inside `train_one_fold`, the code constructs
`SSMBackbone`, `PredictionHead`, and `torch.optim.Adam` fresh on every call. The `__main__` loop
calls the function once per case, so nothing persists across iterations. `set_seed(cfg['seed'])`
runs at the top of every call, so each fold initializes identically and independently.

**Claude comment:** I proposed this as the leading hypothesis before reading the file, and it was
wrong. Recording that here rather than quietly dropping it, since the reasoning that made it
plausible — order and case identity being confounded — still stands even though this particular
mechanism doesn't.

---

## 6. Investigating case4's narrow score distribution

The `mean`/`euclidean` score histogram for case4 shows normal clips in a very tight band around
1.0, much narrower than case1's. Two explanations compete: a genuinely homogeneous machine, or a
collapsed embedding space.

### 6.1 Score and raw-feature dispersion

Coefficient of variation (`cv`, standard deviation divided by mean) normalizes for the different
absolute scales across folds.

| case | normal score cv | anomaly score cv | raw clip_rms cv |
|---|---|---|---|
| 1 | 0.8786 | 0.8883 | 0.1249 |
| 2 | 0.0531 | 0.3314 | 0.0129 |
| 3 | 0.6877 | 0.5409 | 0.0802 |
| 4 | 0.0420 | 0.3026 | 0.0048 |

`clip_rms` measures the normalized log-mel features directly and never touches the trained model.
Case4's raw normal clips vary about 26 times less than case1's. The tight score cluster
therefore reflects a property of the recordings, not something the network introduced.

Collapse is also ruled out: case4's anomaly `cv` is 0.3026, roughly seven times its normal `cv`,
with scores spanning 1.41 to 8.92. A collapsed representation would compress both classes.

### 6.2 Clipping and gain check

Because `cv=0.0048` is extreme, the raw waveforms were checked for digital clipping or a fixed
gain ceiling. Eight normal clips per case, read directly from the source wav files:

| case | subtype | mean peak | peak cv | samples at or above 0.9999 |
|---|---|---|---|---|
| 1 | PCM_16 | 0.21182 | 0.22115 | 0 |
| 4 | PCM_16 | 0.35353 | 0.09009 | 0 |

No clipping anywhere, standard PCM_16 format, peaks well below the ceiling with continuous
values.

**The peak and RMS measurements diverge, and the divergence is informative.** Case4's peak `cv`
is 2.4 times tighter than case1's, but its RMS `cv` is 26 times tighter. If uniformity came from
a gain ceiling, both measures would tighten together. Instead, instantaneous peaks vary
normally while whole-clip energy is highly repeatable. The consistent quantity is case4's
**energy envelope across the full 11-second run**, not its raw waveform amplitude.

Case4 also records louder overall than case1 (mean peak 0.354 versus 0.212), consistent with
ordinary session-to-session variation in mic distance, gain, or motor loudness.

### 6.3 Extreme-scoring clips and fault type

Case4's two highest-scoring clips sit far above the rest, at 8.917 and 8.914, with the next
clip at 2.975. Cross-referencing `fault_table_detection.py`:

| clip | shaft | gears | tires | voltage |
|---|---|---|---|---|
| ab49 | Bent | Melted | Coiled (plastic) | Under |
| ab22 | Normal | Melted | Coiled (plastic) | Under |
| ab53 | Bent | Melted | Coiled (steel) | Over |

`ab49` and `ab53` share bent shaft and melted gears, but only `ab49` reaches the extreme tier.
The distinguishing factors are plastic versus steel coil and under- versus over-voltage. This
suggests melted gears combined with plastic coiling and under-voltage produce a markedly louder
failure mode. Treat this as a hypothesis worth checking, not an established mechanism.

Case1's top five clips are all Deformed-gear faults with a normal shaft (ab13, ab17, ab16,
ab09) — a different dominant failure signature, as expected for a different physical unit.

---

## 7. Falsified: two mechanisms for the fold-difficulty ordering

### 7.1 Self-consistency does not predict fold difficulty

**Hypothesis:** a case whose normal operation varies less should be easier to detect anomalies
in, because a distance-based detector can draw a tighter boundary.

**Prediction:** case2, with the lowest AUC, should show the highest raw RMS `cv`.

| case | raw clip_rms cv | mean+euclidean AUC |
|---|---|---|
| 1 | 0.1249 | 0.9338 |
| 2 | 0.0129 | 0.8707 |
| 3 | 0.0802 | 0.9923 |
| 4 | 0.0048 | 1.0000 |

**Result: falsified.** Case2 has the second-*lowest* `cv`, not the highest. Cases 1 and 3 are
also inverted relative to the prediction. Rank correlation is weak and the wrong sign.

### 7.2 Embedding saturation does not occur

**Hypothesis:** unusually loud clips drive embeddings into a saturated region, where
heterogeneous inputs land at a shared distant point. Cases 1 and 3 contain clips at roughly
twice typical energy (clip_rms max 1.9641 and 2.0733 against means near 1.0), while cases 2 and
4 do not. Case3's top five scoring clips include two *normal* clips, ranked above nearly every
anomaly.

**Prediction:** the top-scoring clips should show collapsed pairwise distances relative to the
rest of the test set.

| case | r(score, clip_rms) | pairwise dist, top-25 | pairwise dist, rest | L2 norm, top-25 | L2 norm, rest |
|---|---|---|---|---|---|
| 1 | +0.9565 | 2.3192 | 1.3499 | 7.3747 | 5.0200 |
| 2 | +0.7238 | 1.6782 | 1.0808 | 5.4828 | 5.3979 |
| 3 | +0.8875 | 4.6119 | 1.2182 | 5.2011 | 4.5371 |
| 4 | +0.3825 | 2.5305 | 1.5708 | 5.5476 | 5.4484 |

**Result: falsified.** Top-25 pairwise distances are *larger* than the rest in all four folds,
by nearly 4x for case3. Embedding norms are essentially unchanged for cases 2, 3, and 4. The
extreme-scoring clips are more spread out than the general population, not collapsed together.

### 7.3 What the falsification runs did establish

**Under `mean` pooling with Euclidean distance, the anomaly score is substantially a loudness
measurement.**

| case | r(score, clip_rms) | mean+euclidean AUC |
|---|---|---|
| 1 | +0.9565 | 0.9338 |
| 2 | +0.7238 | 0.8707 |
| 3 | +0.8875 | 0.9923 |
| 4 | +0.3825 | 1.0000 |

At r=+0.96, case1's detector produces scores nearly interchangeable with a raw energy meter.
This is the same failure `120_evaluation_of_pooling_methods.md` §1 documented on CNT data, where
PC1 correlated with loudness at r=-0.979. **The behavior survived the IND rebuild.**

Two consequences follow.

**Loudness dependence tracks worse performance, not better.** The least loudness-driven fold
(case4, +0.38) scores highest; the most loudness-driven (case1, +0.96) scores lowest. This
argues against the concern that high AUC values are inflated by a loudness shortcut. Real faults
are not reliably louder than normal operation, so a detector leaning on loudness performs worse.

**It explains case2's Euclidean-to-Mahalanobis gap concretely.** Euclidean scoring at r=+0.72
yields 0.8707. Mahalanobis whitens the dominant high-variance direction and yields 0.9962. This
reproduces §3 of the pooling finding on IND data, now with a measured correlation rather than an
inferred one.

---

## 8. The fold-difficulty ordering remains open

### 8.1 The trend is partly an artifact of fixing one scoring method

Selecting the best of the 12 combinations per fold rather than pinning `mean`/`euclidean`:

| case | best combination | AUC |
|---|---|---|
| 1 | max + knn_clustered_16 | 0.9647 |
| 2 | mean + mahalanobis | 0.9962 |
| 3 | mean + euclidean | 0.9923 |
| 4 | mean + euclidean | 1.0000 |

Case2 moves from worst to second-best, and the ordering disappears. The defensible claim is
narrower: **case1 is somewhat harder than the other three, which cluster at 0.99 and above.**
Case2's apparent weakness is specific to Euclidean scoring and is explained by §7.3.

### 8.2 Why no explanation is offered

Two mechanistic hypotheses were tested against these four points and both failed. With four
folds, one seed, and no repeated runs, the data cannot distinguish a real ordering from
run-to-run variation.

**Claude comment:** proposing a third hypothesis to test against the same four points risks
fitting a story to noise. The honest position is that the ordering is not yet established as a
phenomenon, so there may be nothing to explain.

### 8.3 What would settle it

Retrain each fold at three seeds and compare between-fold spread against within-fold spread. If
seed-to-seed variation within a case exceeds case-to-case variation, the question dissolves. If
fold differences survive, the ordering is real and worth a mechanism.

DCASE reports its own task-2 baselines as mean ± standard deviation over five independent
trials, which reflects how unstable single-run numbers are on this benchmark family. At roughly
one hour per fold, a three-seed sweep costs about 12 hours of GPU time.

---

## 9. Conclusions

**Validated:**

- The pipeline reproduces the historical reference result for held-out case 1 (train skill
  0.5482, val skill 0.5452) after a complete data-path rebuild.
- No protocol memorization. Removing the onset and tail silence brackets does not reduce AUC.
- No training-loop state leakage. Model, head, and optimizer are constructed per fold.
- No embedding collapse or saturation, and no clipping or gain artifacts in the source audio.
- Case4's tight score distribution reflects a genuinely repeatable energy envelope in that
  unit's normal operation, measurable in the raw features independent of the model.

**Established as a new concern:**

- Euclidean scoring under `mean` pooling is heavily loudness-driven (r = +0.38 to +0.96 across
  folds), reproducing the CNT-era finding. Mahalanobis and kNN heads mitigate it.
- `concat_mean_last` produces ill-conditioned covariance in every fold (1e6 to 1e7), and inverts
  entirely on case2 under both Euclidean and Mahalanobis scoring.

**Open:**

- Whether fold difficulty differs systematically at all, pending a multi-seed sweep (§8.3).
- Why `tail_only` scoring inverts in three of four folds (§4).
- Whether the `ab49`/`ab22` extreme-loudness fault-factor hypothesis holds up (§6.3).
- k-means seed sensitivity for `knn_clustered_16`, carried over unresolved from
  `120_evaluation_of_pooling_methods.md` §7.