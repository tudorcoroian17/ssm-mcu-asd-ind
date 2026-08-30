# Finding: default model evaluated across all four LOSO folds, three seeds

**Status:** Resolved. The original concern — that fold difficulty increased with case number —
was an artifact of single-seed measurement. Reference-set choice was checked and does not change
the recommendation (§13). Three open items remain, logged in §12, plus two new ones from §13.

**Context:** First full evaluation of the `ssm-mcu-asd-ind` pipeline. The default configuration
(`configs/default.yaml`) was trained on all four leave-one-subject-out folds at seeds 42, 158,
and 824, then scored with three pooling modes against four distance heads.

**Claude comment:** this document supersedes an earlier single-seed version. Several conclusions
in that version were wrong, and the sections where that happened say so explicitly rather than
quietly presenting the corrected result. The record of what failed is the more useful half.

---

## 1. What was run

Training used plain Adam with no gradient clipping, no learning-rate schedule, and no weight
decay, matching the configuration that produced the reference numbers in
`01_design_decisions.md` §7.3. Each fold trains on the three non-held-out cases' IND normal
clips, split 80/20 into train and validation. The held-out case supplies a 1:1 balanced test set
of 264 or 265 clips per class.

Held-out case 1 at seed 42 reproduced the historical reference almost exactly: train skill
0.5482, val skill 0.5452, against the recorded 0.548 and 0.545. The loss curve descended
smoothly across 49 epochs, confirming that the `val_mse` instability documented in
`01_design_decisions.md` §7.2 does not occur on IND data.

Scripts used:

| Script | Purpose |
| :--- | :--- |
| `src/eval/embeddings.py` | Per-fold embedding generation and persistence, all pooling modes |
| `src/eval/auc_pauc.py` | Pooling $\times$ distance-head sweep, 27 charts per case |
| `src/eval/footprint.py` | Flash and streaming-RAM footprint estimate |
| `checks/metrics/onset_tail_contribution.py` | Protocol-memorization diagnostic |
| `checks/smoke/coefficient_of_variation.py` | Score and raw-feature dispersion per case |
| `checks/smoke/embedding_saturation.py` | Embedding geometry of extreme-scoring clips |
| `checks/smoke/decay_half_life.py`, `checks/smoke/zero_state.py` | State-utilisation diagnostics |
| `checks/metrics/misranked_clips.py` | Anomalies ranked below normal clips, by fault code |
| `checks/metrics/score_fusion.py` | Pairwise fusion of distance heads |
| `checks/metrics/reference_set_ablation.py` | Train vs. validation-normal vs. combined reference sets |

Note on script paths. The evaluation code was restructured after most of this document's results were produced: checks/metrics/eval_auc_pauc.py became src/eval/auc_pauc.py, with embedding generation split into src/eval/embeddings.py. Results are unaffected — the numerical paths are identical — but the six checks/ scripts above still import get_embeddings from its old location and load checkpoints from the pre-restructure flat path. They must be repaired before any re-run.

---

## 2. Results: AUC across folds and seeds

Mean pooling only. See §2.2 for other pooling modes.

### 2.1 Mean pooling

| case | head | seed 42 | seed 158 | seed 824 | mean |
|---|---|---|---|---|---|
| 1 | euclidean | 0.9338 | 0.9241 | 0.9369 | 0.9316 |
| 1 | mahalanobis | 0.8797 | 0.7765 | 0.8419 | 0.8327 |
| 1 | knn_full | 0.9524 | 0.9420 | 0.9728 | 0.9557 |
| 1 | knn_clustered_16 | 0.8965 | 0.8819 | 0.9185 | 0.8990 |
| 2 | euclidean | 0.8707 | 0.9016 | 0.7832 | 0.8518 |
| 2 | mahalanobis | 0.9962 | 0.9962 | 0.9962 | 0.9962 |
| 2 | knn_full | 0.9886 | 0.9589 | 0.9756 | 0.9744 |
| 2 | knn_clustered_16 | 0.9896 | 0.9520 | 0.9744 | 0.9720 |
| 3 | euclidean | 0.9923 | 0.9845 | 0.8791 | 0.9520 |
| 3 | mahalanobis | 0.9888 | 0.9887 | 0.9887 | 0.9887 |
| 3 | knn_full | 0.9887 | 0.9888 | 0.9887 | 0.9887 |
| 3 | knn_clustered_16 | 0.9887 | 0.9887 | 0.9887 | 0.9887 |
| 4 | euclidean | 1.0000 | 1.0000 | 0.9811 | 0.9937 |
| 4 | mahalanobis | 0.9925 | 0.9925 | 0.9925 | 0.9925 |
| 4 | knn_full | 0.9925 | 0.9925 | 0.9818 | 0.9889 |
| 4 | knn_clustered_16 | 0.9925 | 0.9925 | 0.9552 | 0.9801 |

### 2.2 Head averaged across all four folds, per seed

| head | pooling | seed 42 | seed 158 | seed 824 | mean |
|---|---|---|---|---|---|
| **knn_full** | mean | **0.9806** | **0.9706** | **0.9797** | **0.9770** |
| knn_clustered_16 | mean | 0.9668 | 0.9538 | 0.9592 | 0.9599 |
| mahalanobis | mean | 0.9643 | 0.9385 | 0.9548 | 0.9525 |
| euclidean | mean | 0.9492 | 0.9526 | 0.8951 | 0.9323 |

`knn_full` under mean pooling is the strongest head at every seed. §11 covers why it is not the
deployment recommendation.

### 2.3 Max pooling is unstable on case1

| head | seed 42 | seed 158 | seed 824 | std |
|---|---|---|---|---|
| max + euclidean | 0.9637 | 0.8331 | 0.6369 | **0.134** |
| max + knn_full | 0.9639 | 0.8621 | 0.7071 | **0.106** |
| max + knn_clustered_16 | 0.9647 | 0.8392 | 0.7123 | **0.104** |

At seed 42, `max + knn_clustered_16` was case1's best combination at 0.9647. At seed 824 the
same combination scores 0.7123. **An earlier version of this document reported that combination
as case1's best configuration. That conclusion was an artifact of one seed.** Max pooling on
case1 is not a good configuration that underperformed once; it is unstable.

### 2.4 Covariance conditioning

The Mahalanobis head warns when the covariance condition number exceeds 1e6. Every
`concat_mean_last` fold exceeded it at every seed, ranging from 6.61e6 to 1.32e8. No `mean` or
`max` fold triggered the warning at any seed.

`concat_mean_last + mahalanobis` on case2 scores 0.2503, 0.3682, and 0.5409 across seeds — below
chance at two of three. This confirms the near-singularity concern from
`120_evaluation_of_pooling_methods.md` §7 as disqualifying rather than cautionary. The
128-dimensional embedding pairs two correlated halves derived from the same sequence.

**Recommendation: drop `concat_mean_last + mahalanobis` from the evaluation grid.**

---

## 3. Why the results prompted an investigation

Three folds produced AUC above 0.99, and case4 reached 1.0000 under three configurations.
Results this high warrant checking for a shortcut.

The initial concern was that fold difficulty appeared to increase with case number: at seed 42
under mean pooling with Euclidean distance, the folds read 0.9338, 0.8707, 0.9923, and 1.0000.
Because folds were trained in ascending case order, case identity and training order were
confounded.

That framing was imprecise from the start — the sequence is not monotonic, since case2 scores
below case1. §8 resolves the question with multi-seed data.

---

## 4. Ruled out: protocol memorization

`01_design_decisions.md` §7.4 flagged a risk specific to IND clips. Every clip shares a
near-identical structure: roughly 0.816s of onset silence, a motor-running section, and roughly
0.45s of tail silence. A model could learn this fixed timing instead of the engine sound.

The diagnostic scores each fold four ways: full clip, onset alone (first 40 frames), tail alone
(last 25 frames), and the middle with both brackets removed (279 frames).

| case | full_clip | onset_only | tail_only | middle_only |
|---|---|---|---|---|
| 1 | 0.9338 | 0.8338 | 0.2938 | 0.9438 |
| 2 | 0.8707 | 0.7083 | 0.0969 | 0.9843 |
| 3 | 0.9923 | 0.4840 | 0.6154 | 0.9923 |
| 4 | 1.0000 | 0.7756 | 0.2439 | 1.0000 |

**`middle_only` matches or exceeds `full_clip` in every fold.** Removing the silence brackets
does not degrade performance; for case2 it improves AUC from 0.8707 to 0.9843. If the model
depended on clip timing, removing that timing would hurt.

The `onset_only` scores are non-trivial in three folds, consistent with motor startup transients
carrying genuine fault-correlated information. That is a second source of real signal, not a
leak.

`tail_only` inverts in three folds. The tail is 25 of 344 frames, about 7% of a mean-pooled
embedding, so its influence on the full-clip score is small. The inversion is unexplained and
logged in §12.

**Conclusion: the §7.4 protocol-memorization risk is resolved.**

---

## 5. Ruled out: training-loop state leakage

If `train_one_fold` reused model weights or optimizer state between folds, later folds would
start warm and appear artificially easy, in exactly the observed order.

A direct read of `src/train.py` rules this out. Inside `train_one_fold`, the code constructs
`SSMBackbone`, `PredictionHead`, and `torch.optim.Adam` fresh on every call. The `__main__` loop
calls the function once per case, so nothing persists across iterations. `set_seed(cfg['seed'])`
runs at the top of every call.

**Claude comment:** I proposed this as the leading hypothesis before reading the file, and it was
wrong. The reasoning that made it plausible — order and case identity being confounded — still
stood, but the mechanism did not.

---

## 6. Case4's narrow score distribution

The `mean`/`euclidean` score histogram for case4 shows normal clips in a very tight band around
1.0, much narrower than case1's. Two explanations compete: a genuinely homogeneous machine, or a
collapsed embedding space.

### 6.1 Score and raw-feature dispersion

Coefficient of variation (`cv`) normalizes for the different absolute scales across folds.
Measured at seed 42.

| case | normal score cv | anomaly score cv | raw clip_rms cv |
|---|---|---|---|
| 1 | 0.8786 | 0.8883 | 0.1249 |
| 2 | 0.0531 | 0.3314 | 0.0129 |
| 3 | 0.6877 | 0.5409 | 0.0802 |
| 4 | 0.0420 | 0.3026 | 0.0048 |

`clip_rms` measures the normalized log-mel features directly and never touches the trained model,
so it is seed-independent. Case4's raw normal clips vary about 26 times less than case1's. The
tight score cluster reflects a property of the recordings.

Collapse is ruled out: case4's anomaly `cv` is 0.3026, roughly seven times its normal `cv`, with
scores spanning 1.41 to 8.92. A collapsed representation would compress both classes.

### 6.2 Clipping and gain check

Because `cv=0.0048` is extreme, the raw waveforms were checked for digital clipping or a fixed
gain ceiling. Eight normal clips per case, read from the source wav files:

| case | subtype | mean peak | peak cv | samples at or above 0.9999 |
|---|---|---|---|---|
| 1 | PCM_16 | 0.21182 | 0.22115 | 0 |
| 4 | PCM_16 | 0.35353 | 0.09009 | 0 |

No clipping, standard PCM_16 format, peaks well below the ceiling.

**The peak and RMS measurements diverge, and the divergence is informative.** Case4's peak `cv`
is 2.4 times tighter than case1's, but its RMS `cv` is 26 times tighter. A gain ceiling would
tighten both together. Instead, instantaneous peaks vary normally while whole-clip energy is
highly repeatable. The consistent quantity is case4's **energy envelope across the full
11-second run**, not its waveform amplitude.

Case4 also records louder than case1 (mean peak 0.354 versus 0.212), consistent with ordinary
session-to-session variation in mic distance, gain, or motor loudness.

---

## 7. Two falsified hypotheses for the fold-difficulty ordering

### 7.1 Self-consistency does not predict fold difficulty

**Hypothesis:** a case whose normal operation varies less should be easier to detect anomalies
in, because a distance-based detector can draw a tighter boundary.

**Prediction:** case2, with the lowest AUC at seed 42, should show the highest raw RMS `cv`.

| case | raw clip_rms cv | mean+euclidean AUC (seed 42) |
|---|---|---|
| 1 | 0.1249 | 0.9338 |
| 2 | 0.0129 | 0.8707 |
| 3 | 0.0802 | 0.9923 |
| 4 | 0.0048 | 1.0000 |

**Falsified.** Case2 has the second-lowest `cv`, not the highest. Cases 1 and 3 are also inverted
relative to the prediction.

### 7.2 Embedding saturation does not occur, except intermittently on case1

**Hypothesis:** unusually loud clips drive embeddings into a saturated region where heterogeneous
inputs land at a shared distant point. Cases 1 and 3 contain clips at roughly twice typical
energy; cases 2 and 4 do not.

**Prediction:** top-scoring clips should show collapsed pairwise distances relative to the rest.

Ratio of top-25 mean pairwise distance to the remainder. Values below 1.0 indicate collapse.

| case | seed 42 | seed 158 | seed 824 |
|---|---|---|---|
| 1 | 1.72 | **0.57** | **0.93** |
| 2 | 1.55 | 1.18 | 1.60 |
| 3 | 3.79 | 4.18 | 2.73 |
| 4 | 1.61 | 1.71 | 2.14 |

**Falsified for cases 2, 3, and 4 at every seed.** Case1 collapses at two of three seeds.

**An earlier version of this document declared saturation falsified in all four folds. That was
correct for seed 42 and wrong as a general statement.**

### 7.3 What the loudness measurements did establish, and its limits

Correlation between anomaly score and raw clip RMS, mean pooling with Euclidean distance:

| case | seed 42 | seed 158 | seed 824 | mean ± std |
|---|---|---|---|---|
| 1 | +0.9565 | +0.3843 | +0.8453 | 0.729 ± 0.248 |
| 2 | +0.7238 | +0.8165 | +0.7237 | **0.755 ± 0.044** |
| 3 | +0.8875 | +0.1940 | +0.6985 | 0.593 ± 0.293 |
| 4 | +0.3825 | +0.1347 | +0.6016 | 0.373 ± 0.191 |

Loudness dependence is real and substantial: 11 of 12 measurements exceed +0.13, most exceed
+0.6. This reproduces the CNT-era finding in `120_evaluation_of_pooling_methods.md` §1, where PC1
correlated with loudness at r=-0.979. **The behavior survived the IND rebuild.**

Only case2's correlation is stable enough (± 0.044) to characterize as a property of the fold
rather than of a particular run.

**An earlier version claimed that lower loudness dependence predicts higher AUC, based on case1
(+0.96, worst) versus case4 (+0.38, best) at seed 42. That relationship does not survive: at
seed 158 case1 drops to +0.38 while remaining the worst fold, and at seed 824 all four cases sit
between +0.60 and +0.85 with no correspondence to AUC ranking.**

### 7.4 Loudness latching damages pAUC specifically

Case1 shows a coherent mechanism linking loudness dependence to the false-positive region:

| seed | r | normals in top-25 | top-25 clip RMS | rest RMS | mean+eucl pAUC |
|---|---|---|---|---|---|
| 42 | +0.96 | 4 | 1.8427 | 1.0086 | 0.7249 |
| 824 | +0.85 | 4 | 1.8437 | 1.0086 | 0.7754 |
| 158 | +0.38 | 0 | 1.0050 | 1.0503 | **0.7861** |

When the detector latches onto loudness, loud normal clips enter the top ranks and pAUC falls.
When it does not, the top 25 are all genuine anomalies and pAUC is highest. Because pAUC
integrates only the low-false-positive region, the cost lands there rather than on AUC broadly.

---

## 8. Fold difficulty: resolved

Best of the 12 pooling × head combinations per fold, per seed:

| case | seed 42 | seed 158 | seed 824 | mean ± std |
|---|---|---|---|---|
| 1 | 0.9647 | 0.9420 | 0.9734 | **0.9600 ± 0.0132** |
| 2 | 0.9962 | 0.9962 | 0.9962 | **0.9962 ± 0.0000** |
| 3 | 0.9923 | 0.9898 | 0.9887 | **0.9903 ± 0.0015** |
| 4 | 1.0000 | 1.0000 | 0.9978 | **0.9993 ± 0.0010** |

**The ascending 1→4 ordering does not exist.** Case2 outranks case3 at every seed.

**Case1 is genuinely the hardest fold**, and it carries roughly 10x the seed variance of the
other three. Cases 2, 3, and 4 cluster tightly at 0.99 and above.

§9 shows that case1's difficulty is different in kind, not only in degree.

---

## 9. The residual error is a small set of specific clips

### 9.1 The 1/265 quantization

Several AUC values repeat identically across seeds. With 265 clips per class, one anomaly ranked
below every normal costs exactly 1/265 = 0.003774 of AUC. Verified empirically through the
discordant-pair count:

| case | head | AUC | discordant pairs | ÷ n_neg | clips in residual set |
|---|---|---|---|---|---|
| 2 | mahalanobis | 0.9962 | 265 | 1.00 | `ab22` 0003 |
| 3 | mahalanobis | 0.9887 | 791–795 | 3.00 | `ab08` 0005, `ab33` 0004, `ab33` 0005 |
| 4 | mahalanobis | 0.9925 | 530 | 2.00 | `ab22` 0001, `ab49` 0004 |

**The same clips form the residual set at both seeds tested.** Case4's `knn_full` and
`knn_clustered_16` bury the identical pair. These folds achieve perfect separation except for a
handful of specific, reproducible clips.

This reframes the ceiling results: they are not diffusely near-perfect. They are perfect apart
from three named clips in case3, two in case4, and one in case2.

### 9.2 Fault profiles of the residual clips

From `fault_table_detection.py`:

| code | shaft | gears | tires | voltage |
|---|---|---|---|---|
| ab22 | Normal | Melted | Plastic ribbon | Under |
| ab49 | Bent | Melted | Plastic ribbon | Under |
| ab08 | Normal | Normal | Steel ribbon | Over |
| ab33 | Bent | Normal | Steel ribbon | Normal |

Case4's two clips share melted gears, plastic ribbon, and under-voltage, differing only in shaft.
**`ab22` is also the single residual clip in case2** — the same fault profile defeats Mahalanobis
on two independent machines. Case3's pair share steel ribbon with normal gears.

This partly confirms the hypothesis in §6.3 of the earlier version, but the direction was
backwards. At seed 42 those clips were the *highest*-scoring under Euclidean (8.917 and 8.914);
under Mahalanobis they rank below every normal clip. Same clips, opposite extremes, depending
only on the metric.

Cross-era support: in the CNT-era catch-rate tables, `ab25` and `ab49` both had catch rate 0.0
under mean pooling, and `ab22` sat at 0.4. The same fault codes are hardest now, on rebuilt data.
That makes this a property of the fault type rather than of one pipeline.

### 9.3 Case1 fails differently in kind

Case1 has essentially no buried clips (0 under Euclidean, `knn_full`, and `knn_clustered_16`) but
211–257 anomalies overlapping at least one normal clip. Its errors are diffuse rather than
concentrated.

That is a qualitatively different regime from cases 2–4, and it explains both the lower AUC and
the higher seed variance: there is no clean separation to be stable about.

---

## 10. Score fusion does not help

**Hypothesis:** Euclidean and Mahalanobis fail on disjoint fault families, so fusing them should
recover the residual gap.

The disjointness is real:

| case | Euclidean struggles with | Mahalanobis buries |
|---|---|---|
| 2 | ab04, ab28 | ab22 |
| 3 | ab02, ab20, ab23 | ab08, ab33 |
| 4 | ab06, ab07 | ab22, ab49 |

No overlap in any fold. Case4 at seed 158 is starkest: Euclidean scores AUC 1.0000 with zero
errors, catching exactly the two clips Mahalanobis buries.

**But it does not convert into AUC.** Fixing one method a priori and averaging across folds
(seeds 42 and 158; the seed 824 fusion run was not completed):

| method (fixed a priori) | seed 42 | seed 158 |
|---|---|---|
| **knn_full alone** | **0.9806** | **0.9706** |
| rank_mean(knn_full, knn_clustered_16) | 0.9818 | 0.9654 |
| rank_mean(euclidean, mahalanobis) | 0.9720 | 0.9589 |
| rank_mean(euclidean, knn_full) | 0.9706 | 0.9687 |
| z_mean(mahalanobis, knn_full) | 0.9691 | 0.9460 |

No fusion beats plain `knn_full` at both seeds.

**The arithmetic explains why.** Rescuing every buried clip in case4 is worth at most 2/265 =
+0.0075. Fusing a strong head with a weaker one degrades the bulk of the ranking by more than
that. Case1 at seed 158 is clearest: `mahalanobis` alone scores 0.7765, and every fusion
involving it falls below `knn_full` alone.

The `best fusion per case` line printed by the script is selection bias — it takes the maximum
over 22 options evaluated on test data. It should not be read as a result.

**Claude comment:** I proposed fusion expecting a meaningful gain. The 1/265 ceiling was
calculable in advance and I should have run it before predicting. The disjointness finding stands
on its own; the fusion conclusion is negative.

### 10.1 Rank and z fusion diverge, and the reason matters for deployment

On case4, `rank_mean(euclidean, *)` reaches 1.0000 at both seeds while every `z_*` variant stays
at 0.9925 with `ab22` and `ab49` still buried. `z_max` should have rescued them, since Euclidean
ranks those clips highest, but it returns the Mahalanobis result exactly.

The cause is the calibration population. Z-scores use validation normals from the **training**
cases, while test clips come from a **different machine**. Every held-out clip sits far from the
training centroid, so the calibration constants do not put the heads on comparable scales. Rank
fusion is immune because it is scale-free, but it is transductive and cannot run per-clip on an
MCU.

**This affects thresholding beyond fusion: any deployed threshold calibrated on other machines'
normals will be miscalibrated on the target machine.** Real deployment would calibrate on
normals collected from the installed unit, which is legitimate but is not what the LOSO protocol
simulates. This belongs in `01_eval_spec.md`.

---

## 11. Deployment recommendation

Head storage cost at `d_model=64`:

| head | stored parameters | fp32 size | mean AUC (3 seeds) |
|---|---|---|---|
| euclidean | 64 centroid | 256 B | 0.9323 |
| **knn_clustered_16** | **16 × 64 references** | **4 KB** | **0.9599** |
| mahalanobis | 64 centroid + 64×64 inverse covariance | 16.4 KB | 0.9525 |
| knn_full | 3240 × 64 training embeddings | 810 KB | 0.9770 |

`knn_full` is the most accurate head but is not deployable at 810 KB — larger than the backbone
itself (~156 KB int8, per `helpers/vis_inspect_model.py`).

**Recommend `mean` pooling with `knn_clustered_16`.** It beats Mahalanobis at every seed while
using a quarter of the memory and avoiding conditioning problems entirely. Report `knn_full` as
an accuracy upper bound.

---

## 12. Open items and known defects

### Script defects to fix

- **`train.py` checkpoint naming omits the seed. CLOSED** — will not fix, by decision. Seed is not an ablation axis for this project and the working seed is fixed at 158. The three-seed sweep in this document was archived by hand to archive/SEED_{42,158,824}/ between runs. The hazard remains latent: if a future phase varies seed programmatically without first adding 'seed': cfg['seed'] to train_config_hash, runs will collide silently. Recorded in 00_index.md amendment 7.
* **`src/eval/embeddings.py` set `model.pooling_mode` instead of `model.pooling` (FIXED):**
  * `SSMBackbone._pool()` reads `self.pooling`; the loop was setting an unused attribute, causing every pooling mode to silently fall back to mean-pooled embeddings.
  * **Symptom:** Identical AUC and pAUC across all three pooling modes in `eval/results.csv`.
  * **Scope:** Affected only the post-restructure seed-158 artifacts, which have now been regenerated. The pooling comparisons in §2.2 and §2.3 predate the restructure and remain unaffected.

* **`src/eval/footprint.py` double-counted parameters (FIXED):**
  * Summed `model.parameters()` twice (once as "weights" and once as "biases"), reporting $2\times$ the true Flash footprint (303 KB INT8 instead of ~152 KB).
  * Omitted the largest transient tensors from peak-RAM estimation because `_scan()` and `discretize()` are implemented as plain methods rather than `nn.Module` boundaries.
  * **Resolution:** Replaced with an analytical streaming-execution estimate. The ~156 KB INT8 figure cited in §11 was derived independently via `helpers/vis_inspect_model.py` and was always correct.

### Suggested improvement

- Persist embeddings, not just scores. DONE. src/eval/embeddings.py now persists train, validation-normal, validation-anomaly, and test embeddings plus labels and normalization stats, per fold per pooling mode, to runs/case{N}/<hash>/embeddings/emb_<pooling>.npz, with a manifest.csv recording which pool each clip belongs to. Scores are persisted separately to scores/. All downstream analysis is now pure NumPy with no model reload.

### Questions still open

- Why `tail_only` scoring inverts in three of four folds (§4).
- Whether the melted-gears + plastic-ribbon + under-voltage profile is systematically the hardest
  fault type, or whether `ab22`/`ab49` recurrence is coincidence (§9.2).
- The seed 824 fusion run was not completed (§10).
- Whether `train_emb`'s geometry is systematically tightened by training, or whether case2's
  +42% spread increase under `val_normal` is fold-specific (§13.1). Untested at other seeds.
- Why `knn_full` on case1 degrades when `train+val` strictly extends `train` as the reference
  set (§13.3) — expected monotonic behavior did not hold.
- Whether `concat_mean_last`'s already-poor conditioning (§2.4) worsens further under a smaller
  reference set (§13.4). Not yet tested.

---

## 13. Reference set: train-only versus validation-normal versus combined

`train_emb` is generated from clips the model was directly optimized against. Validation
normals come from the same three non-held-out cases but received no gradient, raising the
question of whether `train_emb`'s geometry is artificially tightened by fitting rather than
representative of normal operation generally. Checked at seed 158, `mean` pooling, all four
folds, three reference sets: `train` (3240 clips), `val_normal` (810 clips), `train+val` (4050
clips).

### 13.1 The tightening effect is real but fold-dependent

`cov_trace`, a pooling-independent measure of reference-set spread, comparing `val_normal`
against `train`:

| case | train | val_normal | change |
|---|---|---|---|
| 1 | 0.5939 | 0.6243 | +5.1% |
| **2** | 1.5604 | **2.2188** | **+42.2%** |
| 3 | 0.3794 | 0.3773 | −0.5% |
| 4 | 0.9689 | 0.9722 | +0.3% |

Case2 shows a large, unambiguous widening: validation normals occupy a looser region than the
embeddings the model was fit against. Cases 3 and 4 show no meaningful difference. The
hypothesis is confirmed as a real effect, not confirmed as a general property of training — it
appears specific to case2's fold.

### 13.2 The scoring effect tracks the geometry effect, in both directions

Case2, where geometry shifted most, is also where scoring shifted most, but not uniformly across
heads:

| head | train | val_normal | train+val |
|---|---|---|---|
| euclidean | 0.9016 | 0.8790 | 0.8975 |
| mahalanobis | 0.9962 | 0.9961 | 0.9962 |
| knn_full | 0.9589 | 0.9584 | 0.9593 |
| knn_clustered_16 | 0.9520 | **0.9580** | **0.9589** |

A centroid-based head (`euclidean`) is hurt by a looser reference population, since the centroid
becomes a less precise summary. A local kNN head (`knn_clustered_16`) benefits from seeing the
true extent of normal variation. Cases 3 and 4, where geometry barely moved, show correspondingly
flat scores across every head. The relationship between reference-set geometry and downstream
AUC is mechanistically coherent, but it does not resolve to "train" or "val" being better in
general — it depends on both the fold and the head.

### 13.3 Two results outside the expected pattern

**Case1, `knn_full`: combining reference sets performs worse than either alone.** Train alone
scores 0.9420, val_normal alone scores 0.9371, but `train+val` scores 0.8827 — an 8-point drop
from adding more reference data. `knn_full` uses the k-th nearest neighbor distance, and since
`train+val` is a strict superset of `train`, every test point's score can only decrease or stay
equal when val is added. If the added points happen to sit close to specific borderline
anomalies without similarly tightening genuinely normal test points, those anomalies' scores
shrink toward the normal cluster and the ranking degrades, even though each individual score
moved in the direction superset inclusion guarantees. Not a bug; a property of nearest-neighbor
scoring under reference-set growth. Logged as open in §12.

**Case3, `knn_full`: AUC and pAUC move in opposite directions.** `val_normal` gives 0.9888 →
0.9930 AUC but 0.9940 → 0.9629 pAUC. The overall ranking improves slightly while the
low-false-positive region — the part that determines a usable alarm threshold — regresses. A
change that looks like an improvement on the headline metric can be a regression on the metric
that matters operationally. Logged as open in §12.

### 13.4 Conditioning: directionally confirmed, not yet disqualifying

Mahalanobis condition number, train vs val_normal, `mean` pooling:

| case | train | val_normal |
|---|---|---|
| 1 | 5.00e4 | 5.46e4 |
| 2 | 3.38e4 | 5.73e4 |
| 3 | 1.47e4 | 1.62e4 |
| 4 | 7.50e4 | 7.69e4 |

Higher for `val_normal` in all four cases, consistent with the smaller n/d ratio (810 samples
against `mean` pooling's 64 dimensions, versus train's 3240). None cross the 1e6 instability
threshold under `mean` pooling. This was not tested under `concat_mean_last`, whose baseline
conditioning already sits at 1e6–1e8 (§2.4) — that pooling mode is where a smaller reference set
would most plausibly push an already-borderline fold over the threshold. Untested; logged in
§12.

### 13.5 Decision

**Kept `train` as the default reference set.** The tightening effect is real and mechanistically
understood, but it improves some heads while hurting others within the same fold, and the one
case with an unambiguous geometric shift (case2) does not produce an unambiguous scoring
improvement. Measured at a single seed. Given §7–§8's demonstration that single-seed conclusions
on this pipeline have twice been wrong in this document already, a one-seed result favoring
either reference set is not enough to change a default.