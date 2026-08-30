# Finding: Pooling and distance metric interact — the fault-type blind spot is metric-specific, not architectural

**Status:** Resolved. Supersedes the pooling-only analysis in this file's original version.
**Context:** Option 3 evaluation on the retrained held-out-1 checkpoint (trained on
silence-filtered CNT data, see `110_dataset_fold_stats.md`). Follows the original AUC/PCA
investigation on the unfiltered model, which found an inverted, collapsed embedding space.

**Revision note:** this document originally recommended `max` pooling as an unconditional
default, based on a Euclidean-distance-only comparison. A follow-up test with Mahalanobis
distance overturned that recommendation. The full picture — three distance/scoring methods,
all three pooling modes — is below.

Scope: CNT era. Produced before the IND-only restart (01_design_decisions.md §7). Retained as record. The findings are correct for the pipeline they describe, but that pipeline is not this project's — see the per-file note below.

Superseded by findings/130 for all recommendations. Its headline conclusion — mean pooling + Mahalanobis — does not carry to IND, where Mahalanobis places third of four heads (0.9525 mean AUC against knn_clustered_16's 0.9599 and knn_full's 0.9770). What does carry: the mean pooling choice, the loudness-confounding finding (§1, reproduced on IND in findings/130 §7.3), the concat_mean_last conditioning concern (§7, now disqualifying per findings/130 §2.4), and the 16-reference clustering result (§6, now the deployment recommendation). The k-means seed-sensitivity item left open in §7 is resolved and reclassified: it is process-level nondeterminism, not seed sensitivity — see findings/130 §12.

---

## 1. Re-establishing the baseline: AUC on the filtered model, mean pooling, Euclidean distance

Centroid + Euclidean distance over `mode='pooled'` embeddings, held-out-1:

| | old (unfiltered model) | new (filtered model, mean pooling) |
|---|---|---|
| AUC | 0.235 (inverted; 0.765 flipped) | **0.807** |
| pAUC(p=0.1) | 0.484 | 0.537 |
| outlier cluster (test set) | 22 points | **0** |
| PC1-vs-loudness correlation | 0.678 (mixed population) | -0.979 (near-total, single population) |

Training on silence-filtered data resolved the original inversion and outlier cluster entirely,
confirming the original problem was substantially a training-data-quality issue.

The embedding's dominant PCA axis (74.1% of variance) is almost perfectly explained by clip
loudness (r=-0.979), but PC1 alone is a much weaker classifier (raw AUC 0.380, flipped 0.620)
than the full embedding (0.807) — loudness dominates the representation's *variance* without
dominating its *usefulness*. This observation turned out to be the key to the Mahalanobis
result in Section 3.

---

## 2. Pooling comparison under Euclidean distance

| pooling | AUC | pAUC(p=0.1) | PC1-loudness corr |
|---|---|---|---|
| mean | 0.8073 | 0.5373 | -0.979 |
| max | 0.8111 | 0.6335 | -0.554 |
| concat_mean_last | 0.2755 (0.7245 flipped) | 0.5337 | -0.601 |

Under Euclidean distance, `max` pooling looked like the clear winner — comparable AUC to
`mean`, meaningfully better pAUC (+18% relative). The mechanistic story: `max` preserves brief,
extreme per-channel moments that `mean` averages away, which should help detect transient
mechanical faults specifically.

**This conclusion does not survive the Mahalanobis re-test (Section 3) and should not be acted
on as originally stated.**

---

## 3. Distance metric comparison — Mahalanobis reverses the pooling ranking

Same three pooling modes, Mahalanobis distance (`sqrt(diff @ inv(cov) @ diff)`) in place of
Euclidean:

| pooling | Euclidean AUC / pAUC | Mahalanobis AUC / pAUC | Δ AUC | Δ pAUC |
|---|---|---|---|---|
| **mean** | 0.807 / 0.537 | **0.929 / 0.692** | **+0.122** | **+0.155** |
| max | 0.811 / 0.634 | 0.707 / 0.580 | -0.104 | -0.054 |
| concat_mean_last | 0.275→0.725 / 0.534 | **0.922 / 0.702** | — | +0.168 |

**`mean` and `concat_mean_last` both improve dramatically under Mahalanobis; `max` gets worse.**
`mean` now has the best AUC of any combination tested; `concat_mean_last` has the best pAUC,
narrowly ahead of `mean` (0.702 vs 0.692 — a gap close enough it may not be reliably real).

### Why this happens

Euclidean distance weights every embedding dimension equally. Section 1 established that the
dominant embedding direction (74% of variance under mean pooling) is a loudness proxy with
comparatively low standalone predictive power — Euclidean distance was letting that
high-variance, low-information direction dominate the score, drowning out lower-variance
directions carrying more of the real signal. Mahalanobis distance divides by the inverse
covariance, specifically down-weighting high-variance directions relative to low-variance
ones — exactly correcting the imbalance Euclidean distance was suffering from under `mean`
and `concat_mean_last`.

**`max` pooling responds differently because its embedding distribution has a different shape.**
Max pooling captures rare, extreme single-frame values, producing a heavier-tailed,
non-elliptical embedding distribution (visible directly in the score histograms — max's
Mahalanobis scores span 8–90+ with a scattered tail, versus mean's tight 40–145 band).
Mahalanobis distance's whitening assumes the training covariance meaningfully describes the
shape of "normal" — a reasonable assumption for mean pooling's roughly Gaussian cloud, a poor
one for max pooling's outlier-driven distribution. The inverse-covariance weighting appears
miscalibrated for that shape, actively hurting rather than helping.

---

## 4. Fault-type analysis — the blind spot is specific to max+Euclidean, not general

A script bug caused the first round of fault-type breakdowns (all three pooling modes) to
silently reuse `max`+Euclidean scoring regardless of which pooling mode was nominally being
tested — the caught/missed dictionaries were byte-identical across all three runs. Fixed by
removing a leftover hardcoded `model.pooling = 'max'` block that predated the multi-pooling
loop. Re-run after the fix produced genuinely different breakdowns per configuration.

### Per-factor catch-rate delta (max − min across levels), by configuration

| factor | max + Euclidean | mean + Mahalanobis | concat_mean_last + Mahalanobis |
|---|---|---|---|
| **shaft** | **0.508** (Bent=0.059, Normal=0.567) | 0.026 (flat) | 0.132 (mild) |
| **tires** | **0.208** (coiled helps) | 0.154 (**inverted** — normal helps) | 0.202 (**inverted** — normal helps) |
| gears | 0.119 (noise) | 0.222 | 0.143 |
| voltage | 0.156 (noise) | 0.074 | **0.244** (over > under) |

**The sharp, near-binary shaft blind spot (Bent=0.059 vs Normal=0.567 under max+Euclidean) is
not present under either Mahalanobis-scored configuration** — `mean` shows no shaft effect at
all (Δ=0.026), `concat_mean_last` shows a mild residual (Δ=0.132), roughly a quarter the size.

**The tire effect not only shrinks but reverses direction.** Under max+Euclidean, coiled tires
were *easier* to detect than normal tires — consistent with the "brief periodic transient"
mechanism proposed in the original analysis. Under both Mahalanobis configurations, normal
tires are *easier* to detect than coiled tires. This is a genuinely different failure mode, not
a weaker version of the same one.

**A new pattern emerges under `concat_mean_last`+Mahalanobis that has no counterpart under the
other two configurations:** voltage condition shows the largest effect of any factor for this
combination (Δ=0.244, over-voltage easier than under-voltage) — invisible in every other
configuration tested.

**Conclusion: the tire/shaft mechanism documented in the original version of this file was real
and correctly diagnosed, but scoped incorrectly.** It is a property of max pooling's
representation combined with Euclidean scoring — not a general limitation of the detector.
Each configuration tested has its own, different sensitivity profile across the four damage
factors; none has been shown to be free of *all* such effects, only free of *this specific* one.

---

## 5. Recommendation

**`mean` pooling + Mahalanobis distance** is the strongest overall combination tested: highest
AUC (0.929), pAUC essentially tied for best (0.692 vs concat's 0.702), and no dominant
per-factor blind spot in the breakdown above. **This supersedes the original recommendation of
`max` pooling**, which was correct only under Euclidean distance and not stated as such at the
time.

`concat_mean_last` + Mahalanobis is a close alternative with a marginally better pAUC, at the
cost of double the embedding width (2×d_model) — a real on-device memory cost for a
deployment-feasibility project. Given the pAUC gap between it and `mean` is small enough to be
within noise, `mean` is the more defensible default absent a specific reason to spend the extra
memory.

**Action:** set `pooling: mean` (i.e., leave the architecture default as-is — no change needed
there) and implement Mahalanobis distance as Option 3's scoring method, replacing Euclidean.
This requires storing the training-embedding covariance (or its inverse) alongside the
centroid — a real, non-trivial increase in Option 3's on-device memory footprint relative to
Euclidean (64×64 covariance vs. a single 64-length vector, per Phase 1 §1.6's own cost table),
worth weighing explicitly in Phase 5's deployment accounting rather than assumed free.

This recommendation is further stress-tested against a third scoring method (k-NN) in Section 6
below — it holds.

---

## 6. k-NN as a memory-cheaper alternative to Mahalanobis — tested, does not change the recommendation

Motivation: Mahalanobis's 64×64 covariance matrix (16KB) was flagged as a real on-device memory
cost (Phase 1 §1.6). k-NN was tested as a potentially cheaper alternative, in two forms:

- **Full k-NN** (distance to k-th nearest neighbor among *all* training embeddings) — a
  ceiling/reference measurement, not deployment-realistic. At held-out-1's scale (9,246
  training windows × 64 floats × 4 bytes ≈ 2.37MB), this is actually ~150x *more* expensive
  than Mahalanobis, not cheaper.
- **Clustered k-NN** (training embeddings compressed to 16 representative points via k-means,
  k=1) — the actual deployment-realistic candidate, at 16×64×4B = 4KB, roughly a quarter of
  Mahalanobis's cost.

### Results, k=5 (full) / k=1 (clustered, 16 references)

| pooling | Mahalanobis AUC/pAUC | kNN full AUC/pAUC | kNN clustered (16 refs) AUC/pAUC |
|---|---|---|---|
| **mean** | **0.929 / 0.692** | 0.828 / 0.570 | 0.818 / 0.591 |
| max | 0.707 / 0.580 | **0.772 / 0.619** | 0.756 / 0.608 |
| concat_mean_last | **0.922 / 0.702** | 0.365 (inverted) / 0.601 | 0.301 (inverted) / 0.559 |

**Under `mean` pooling — the current recommendation — k-NN underperforms Mahalanobis in both
forms, by a wide margin on AUC (0.929 vs 0.818–0.828).** Even the unrealistically expensive
full-training-set version doesn't close the gap; this is not a compression-loss story, k-NN's
scoring *shape* is simply a worse match for mean pooling's embedding than Mahalanobis is. The
mechanism: k-NN, like Euclidean, treats every embedding dimension equally and has no analog to
Mahalanobis's covariance-based down-weighting of the loudness-dominated axis (Section 3) — it
inherits Euclidean's blind spot, just measured locally instead of against one centroid.

**Under `max` pooling, k-NN is the best method tested** (AUC 0.772 vs Mahalanobis's 0.707) —
consistent with Section 3's finding that max pooling's heavy-tailed, non-elliptical
distribution is poorly served by Mahalanobis's Gaussian-shaped covariance assumption; k-NN
makes no such distributional assumption and handles it better. Still well below mean+Mahalanobis
overall (0.772 vs 0.929), so this does not change the top-line recommendation.

**Under `concat_mean_last`, k-NN performs badly (AUC inverts to 0.30–0.36) — a genuine negative
result, not noise.** Nearest-neighbor distance in this embedding's higher-dimensional space
(2×d_model=128, with real correlation between the mean-half and last-half since both derive
from the same sequence) is a known weak spot for k-NN-style methods — sparse local
neighborhoods become more likely as dimensionality grows relative to sample count, degrading
the "nearest neighbor = normal" assumption the method relies on.

### A separate, reusable finding: 16-reference clustering costs almost nothing in accuracy

Comparing full k-NN to its 16-reference clustered compression, across all three pooling modes,
the accuracy loss is consistently small: mean (AUC −0.010, pAUC actually *improves* +0.021),
max (AUC −0.016, pAUC −0.011), concat (proportionally similar degradation to the full version).
**Worth remembering independent of which scoring method ultimately gets used** — if Mahalanobis
turns out to have its own problems on a future fold (e.g., a near-singular covariance, see
Section 7), 16-reference clustered k-NN is a validated fallback that recovers most of full
k-NN's signal at a small fraction of its memory cost. Not yet checked for seed-sensitivity
(k-means with only 16 clusters may vary meaningfully run to run) — treat the specific numbers
above as indicative, not final, until that's confirmed.

**Conclusion: k-NN does not change the recommendation.** `mean` pooling + Mahalanobis distance
remains the strongest combination tested, now confirmed against a third scoring method rather
than only two.

---

## 7. Not yet done

- **No configuration has been shown free of all fault-factor sensitivity** — `mean`+Mahalanobis
  has the flattest profile of those tested, not a flat one. A factor with a real but smaller
  effect could still be present and not yet identified.
- **This entire investigation is held-out-1 only.** Whether the tire/shaft/voltage patterns, or
  the pooling×metric interaction itself, generalize to the other three LOSO folds is untested.
- **Mahalanobis's covariance estimate should be sanity-checked for near-singularity** given the
  training-embedding sample size, per the concern already flagged in Phase 1 §1.6 — not yet
  done for this checkpoint. If it turns out unstable, clustered k-NN (Section 6) is a validated
  fallback.
- **The voltage effect under `concat_mean_last`+Mahalanobis is new and unexplained** — worth a
  mechanistic hypothesis if `concat_mean_last` is ever reconsidered, but not pursued further
  here given `mean` is the current recommendation.
- **k-means seed-sensitivity for the 16-reference clustering has not been checked** — the
  Section 6 clustered-kNN numbers used a single fixed seed; worth confirming stability across
  2–3 seeds before treating the compression result as fully validated.
