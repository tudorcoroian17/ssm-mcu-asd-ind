# Activation int8: boundaries are nearly free, scan is genuinely hard, fidelity still does not predict AUC

**Feeds:** `06_phase_5_quantization_and_head.md` (rung 2, activation
quantization); `findings/150` (whose four flagged tensors -- delta, A_bar,
B_bar, h -- are exactly the `scan` group here); `findings/520`/`521` (weight
quantization precedent this extends with a second, independent case).

## Result

A diagnostic ablation (weights stay fp32; this isolates the activation
effect, and is not itself a deployment scheme -- see the reasoning that
motivated it) splits the model's quantizable activations into two groups and
tests each with the best-case (on-the-fly) int8 scale, per-tensor and
per-channel:

- **`boundaries`** (conv output, gates, block output, final norm) is close to
  a clean pass, with one attributable exception: max pooling.
- **`scan`** (delta, A_bar, B_bar, h, B, C, y_scan -- the recurrence internals
  `findings/150` flagged) is not clean under any of three variants tried, but
  the primary metric (mean pooling / knn_full) stays near flat throughout.
- Correcting a granularity bug for the scan group's 4D tensors cut embedding
  error by up to 4x, but did **not** improve AUC/pAUC -- the second
  independent case, after `findings/521`, of embedding fidelity and
  anomaly-detection accuracy moving in opposite directions.

Scope: `f4cd557b7e3b`, case 1, run `16662b29beb3`, on-the-fly scale only.
Ranges-based (deployment-realistic) static scales not yet tested.

## Method

`checks/smoke/quant/activation_quant_parity.py` and `activation_quant_auc.py`
build an `ActivationQuantizer` and pass it into `SSMBackbone.forward`, which
now calls `quantizer.apply(name, tensor)` at every site the existing
`range_recorder` hook already touches (a source change to `ssm_block.py` and
`backbone.py` -- activations, unlike weights, cannot be perturbed by editing
`state_dict()`, since they do not exist until the forward pass runs). `h` is
quantized every timestep, not on the recorder's `t % 10` subsample, since the
quantized state is what feeds the next update.

Two groups, matching `range_recorder`'s names:

- `scan`: `.delta`, `.A_bar`, `.B_bar`, `.h`, `.B`, `.C`, `.y_scan`
- `boundaries`: `.u_post_conv_silu`, `.z_gate`, `.y_gated`, `.block_output`,
  `final_norm_output`

`--scale-source onthefly` computes each tensor's scale from its own runtime
min/max -- a best-case probe, not deployable (the MCU cannot recompute a
scale per frame). `--scale-source ranges` (not yet run) would use fixed
scales from `findings/150`'s `ranges.json`.

## A granularity bug, caught and fixed before trusting the result

Per-channel granularity for a weight tensor scales along axis 0 (output
channel). The first attempt at per-channel for activations naively reused
"last axis," which is correct for 3D tensors (`delta`, `B`, `C`,
`u_post_conv_silu`, ... -- d_inner genuinely is their last axis) but wrong for
`A_bar`, `B_bar`, and `h`: these are shaped `(..., d_inner, d_state)`, so the
last axis is d_state (16 values, the recurrence's internal state size), not
d_inner (128 values, the real channel axis analogous to the weight scripts'
per-channel scaling). The fix adds a name-based exception
(`CHANNEL_AXIS_MINUS2_SUFFIXES`) so these three tensors scale along d_inner
instead.

Consequence of the bug, quantified in "Finding 3" below: the wrong axis
already looked like an improvement over per-tensor, so this was not a
silent-no-op bug -- it would have supported a real conclusion on the wrong
evidence had the axis mismatch gone unnoticed.

## Finding 1: boundaries are close to free, except max pooling

Embedding error (mean_abs_err), boundaries vs weight-only int8
(`findings/520`) for comparison:

| pooling | boundaries | weight-only (520) |
| :--- | ---: | ---: |
| mean | 1.28e-2 | 2.22e-2 |
| max | 6.92e-2 | 2.23e-2 |
| concat_mean_last | 2.61e-2 | 1.82e-2 |

AUC/pAUC: every mean-pooling and concat-pooling row is positive (mean/knn_full
pAUC +0.044, concat/knn_full pAUC +0.053), consistent with the same
incidental-smoothing effect `findings/520`/`521` already documented. Max
pooling is uniformly negative across all four distance heads (-0.012 to
-0.029 AUC) -- four heads agreeing on sign is a real, attributable pattern,
not noise. Max pooling selects one frame's value out of 344 with no
averaging to dilute a bad frame's error; mean pooling spreads the same error
across all 344 frames and it washes out. This is boundaries' one weak point,
and it is mechanistically explained rather than mysterious.

## Finding 2: scan is genuinely hard, as findings/150 predicted

Per-tensor scan embedding error (mean_abs_err 9.65e-2 to 2.13e-1) is 4-10x
weight-only's and 3-8x boundaries', with `max` pooling's `max_abs_err`
reaching 1.5 -- larger than the embedding's own value range. AUC/pAUC under
per-tensor scan is the messiest result in this project's quantization work
so far: `mean`/`mahalanobis` and `concat_mean_last`/`mahalanobis` show large
positive swings (+0.12, +0.17 AUC) that are a known numerical-instability
artifact, not signal -- `score_mahalanobis` (`src/eval/auc_pauc.py`) prints
its condition-number warning unconditionally (a plain `print`, confirmed by
reading the source, not a deduped `warnings.warn`), and a condition number of
1.8e7 means the covariance inversion is near-singular and amplifies noise
disproportionately. Discount both Mahalanobis rows for this reason.

The one non-Mahalanobis result worth taking seriously: `concat_mean_last`/
`knn_clustered_16` pAUC collapses from 0.8506 to 0.7232 (-0.1274) while its
AUC barely moves (+0.0033). AUC integrates the whole ROC curve; pAUC at
max_fpr=0.1 restricts to the low-false-positive region a deployed detector
actually operates in. Flat AUC with collapsed pAUC means the quantization
noise reshuffled rankings specifically among the hardest normal clips near
the decision boundary, while barely denting the aggregate statistic -- exactly
the kind of degradation a headline AUC number hides.

## Finding 3: fixing the granularity bug cut embedding error sharply but did not fix AUC -- confirming this is a real pattern, not a fluke

| scan variant | mean-pool mean_abs_err | max-pool mean_abs_err | mean/knn_full dAUC | worst secondary result |
| :--- | ---: | ---: | ---: | ---: |
| per-tensor | 9.65e-2 | 2.13e-1 | +0.0062 | pAUC -0.1274 (concat/knn_clustered_16) |
| per-channel, wrong axis (d_state) | 1.31e-1 | 1.87e-1 | -0.0002 | recovered: pAUC +0.0122 |
| per-channel, correct axis (d_inner) | **5.76e-2** | **5.61e-2** | -0.0020 | dAUC -0.0563 (max/euclidean) |

The corrected axis roughly quarters max-pooling's embedding error relative to
per-tensor -- confirming d_inner was the axis that mattered, decisively. But
it has the *worst* max-pooling AUC result of the three variants, and does not
recover the pAUC collapse the way the wrong axis accidentally did. This is
the same inversion `findings/521` found for weights (per-channel improves
fidelity, does not improve and can slightly hurt AUC), now observed
independently on activations, with a much larger fidelity swing -- so it is a
pattern in this problem, not a coincidence specific to weight quantization.

Hypothesis, not established: per-channel scaling fits each channel's scale to
that channel's typical range. Max pooling exists to find the single largest
value across the whole clip -- close to an atypical excursion for whatever
channel produces it, by construction. A scale fitted to the typical case may
represent that atypical peak worse, not better, while a cruder shared scale
(per-tensor, or the wrong-axis per-channel) leaves more headroom for outliers
at the cost of average-case accuracy. Plausible and worth stating, but not
verified against an alternative explanation.

## Finding 4: the metric that anchors the deployment claim did not move

Across per-tensor, wrong-axis, and correct-axis scan variants, `mean`/
`knn_full` -- the primary pooling/head combination this project's paper leads
on -- stayed within noise throughout: +0.0062, -0.0002, -0.0020. Every
concerning number in this ablation (the pAUC collapse, the max-pooling
swings, the Mahalanobis artifacts) sits on a secondary combination. Per
`findings/521`'s own precedent, the primary metric is what the go/no-go
decision anchors on.

## Consequence

On the metric the deployment claim rests on, none of the three scan variants
or the boundaries group has shown a problem, even though scan visibly
perturbs the embedding and several secondary metrics. This is a diagnostic
result, not a deployment recommendation: all of it uses on-the-fly scales,
which the MCU cannot compute, and weights are fp32 throughout. The
deployment-realistic version -- `--scale-source ranges`, static scales from
`findings/150`'s calibration -- has not been run, and is expected to perform
closer to the messier per-tensor profile than the cleaner per-channel one,
since `ranges.json` holds no per-channel statistics and static per-tensor
scales cannot adapt per clip the way on-the-fly scales do.

## Open items

- **Ranges-based (static) scale test not yet run.** This is the actual
  deployment question; on-the-fly is a ceiling probe.
- **One fold.** As with `findings/520`/`521`, not blocking further Python
  work but needed before the final accuracy table.
- **The max-pooling-vs-per-channel hypothesis is unverified.** A direct test
  (e.g. checking whether the quantization error specifically at each clip's
  argmax frame is larger under per-channel than per-tensor) would confirm or
  refute it rather than leaving it as a plausible story.
- **Boundaries' max-pooling weakness and scan's per-channel max-pooling
  weakness may or may not share a cause.** Both hit max pooling, but boundaries
  hurts it under every granularity while scan's correct-axis per-channn is the
  outlier that hurts it worst -- these could be the same underlying mechanism
  (single-frame sensitivity) or two different ones. Not disentangled here.

<!-- Claude comment: the throughline worth carrying into the paper is now
supported twice, independently, in weight and activation quantization: better
reconstruction fidelity is not a reliable proxy for anomaly-detection accuracy
in this model, and can point the wrong way. That is a more interesting and
more defensible claim than "quantization works" -- it is a specific, repeated,
mechanistically-motivated result about what fidelity metrics do and do not
tell you for this task, which is exactly the kind of finding a deployment-
feasibility paper benefits from having twice rather than once. -->