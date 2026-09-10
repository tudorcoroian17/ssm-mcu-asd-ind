# Weight-only int8 quantization preserves AUC on the case-1 fold

**Feeds:** `06_phase_5_quantization_and_head.md` (rung 1, post-training int8);
`00_index.md` open item 10 (is int8 backbone quantization lossless);
`findings/225` (which assumed this question was still open when interpreting
the footprint figures).

## Result

Per-tensor symmetric int8 quantization of the backbone weights preserves
anomaly-detection AUC and pAUC on the case-1 fold, with no systematic
degradation across any pooling or distance head. This validates Phase 5's
rung-1 scheme offline, before any change to the export script or the C
backbone.

Scope: `f4cd557b7e3b`, case 1, run `16662b29beb3` -- the same config the
Phase 4 C port and `findings/430` parity used. One fold only.

## Method

Two Python checks, both reusing the existing model and eval pipeline
unchanged and quantizing the loaded weights in place after `load_state_dict`,
so any measured difference is quantization error rather than pipeline
divergence:

- `checks/smoke/weight_quant_parity.py` -- embedding parity. Quantizes each
  weight tensor to int8 and dequantizes it to fp32, then diffs the pooled
  embedding against the fp32 model through the same forward pass
  `src/eval/parity_vectors.py` uses.
- `checks/smoke/weight_quant_auc.py` -- AUC/pAUC impact. Builds fp32 and int8
  embeddings into separate stores (the fp32 baseline is never overwritten),
  then scores both with the four distance heads and three poolings from
  `src/eval/auc_pauc.py`.

Quantization scheme: per-tensor symmetric int8, zero-point 0, range plus or
minus 127, scale = `max(abs(W)) / 127`. Applied to the projection weights
(`in_proj`, `conv`, `x_proj`, `dt_proj`, `out_proj`) and the RMSNorm weights.
The recurrence parameters `A` and `D` and all biases stay fp32 for this pass
(see "What this pass deliberately excludes").

## Finding 1: per-tensor weight quantization is well behaved

Per-tensor SNR ranges from 36 to 52 dB across all thirteen quantized tensors
-- ordinary int8 territory, no outlier tensor demanding per-channel scales.
The recurrence-feeding projections (`x_proj`, `dt_proj`) sit mid-pack at 38 to
44 dB, not at the bottom.

## Finding 2: the scan survives int8 weights numerically

The pooled embedding drifts by a mean absolute error of `2.2e-2` and a maximum
of `7.8e-2`, against embedding values spanning roughly -1 to 2.2. A drift of
this size -- rather than an error compounding across the 344-timestep
recurrence into something large -- is the signature of a recurrence that
survives int8 weights.

This error is roughly 130 times larger than the `5.99e-4` fp32 C-port parity
in `findings/430`, which is expected and not a regression: that figure
measured float32 implementation-order noise, whereas quantization is lossy by
design. The two are not comparable, and the `findings/430` figure is not a
valid pass/fail bar for quantization.

## Finding 3: the drift does not move AUC

Across all twelve pooling-by-head combinations, both AUC and pAUC change by
small, uniformly positive amounts (AUC delta +0.0002 to +0.0068; every pAUC
delta positive). The primary `knn_full` head is effectively unmoved: 0.9683 to
0.9695 (mean pooling), 0.9686 to 0.9702 (concat pooling).

The uniformly positive direction is not evidence that quantization improves
the model. The magnitudes sit inside the fold's own noise band (a different
KMeans seed in `knn_clustered_16`, or a different test split, would move AUC
comparably). The correct reading is that int8 is statistically
indistinguishable from fp32 on this fold, with noise leaning slightly
positive.

Suggested writeup phrasing: weight-only int8 quantization preserved AUC to
within plus or minus 0.007 across all pooling and head combinations on the
case-1 fold, with no systematic degradation.

## What this pass deliberately excludes

`A` (the precomputed `-exp(A_log)` recurrence matrix) and `D` stay fp32 here.
`findings/150` section 6 identifies `A_bar = exp(delta * A)` as the tensor
whose resolution near 1 is load-bearing for the long-half-life state channels
that `findings/130` proved matter. Folding `A` into quantization is a separate,
deliberate probe -- with `decay_half_life.py` re-run against the result, per
`findings/150` recommendation 3 -- not a default to enable silently.

Note that `dt_proj.weight` is quantized in this pass, so `delta` is already
mildly perturbed even though `A` is not. The clean AUC result therefore
already reflects some perturbation of the recurrence half-lives, not none.

## Consequence

- Phase 5 rung 1 is a go on this fold. Per-tensor is sufficient; per-channel
  is not needed and would add complexity the data does not call for.
- `00_index.md` open item 10 ("is int8 backbone quantization lossless") moves
  from open to answered-for-case-1: yes, within noise, for weights only.
- `findings/225`'s caveat -- that its Pareto frontier plots int8-scale flash
  against fp32-measured AUC, and is only fair if int8 is lossless -- is now
  supported on this fold. It is not yet supported across folds.

## Open items

- **One fold only.** `findings/140` measured 2.3 to 6 times cross-machine
  score shift in this dataset family. A clean case-1 result is sufficient to
  commit the scheme to the C port, but the cross-machine AUC story needs the
  other three folds before the deployment claim rests on it.
- Weights only. Activation quantization (Phase 5 rung 2 and beyond) is
  untouched and is where `findings/150`'s `block_output` and `delta`
  dynamic-range concerns actually apply.
- `A`/`D` quantization not yet tested (see "What this pass deliberately
  excludes").

<!-- Claude comment: the single most useful thing this result unblocks is
honest framing for the paper. "Quantization was free" is exactly the result
that makes a deployment-feasibility claim compelling, and it is now defensible
on one fold with a stated plus-or-minus. The temptation to report the positive
deltas as an improvement is the trap -- it would be a claim you cannot defend
and a reviewer would rightly challenge. Report no-degradation, not gain. -->

## Update — Phase 5 Phase 1 refactor dropped backbone timing further

Backbone compute dropped again, to ~3,921 us/frame (2,352,814 cyc/frame,
~7.9x headroom against the 32,000 us budget) -- a further ~16% reduction
from this finding's original ~4,678 us/frame figure above. This happened
as a side effect of `06_phase5_nucleo_deployment_matrix.md` Phase 1's weight
plumbing refactor (`SSM_InitWeights()`, `ssm_blocks_working[]`), not of
anything in this finding -- noted here rather than there because this is
where the board's timing baseline lives.

No quantization is involved in this change; parity against `findings/430`'s
reference is unchanged (max abs error 5.99e-4, mean abs error 1.45e-4), so
the speedup is a pure side effect of how the weights are addressed, not a
change in what's computed.

Hypothesis, not confirmed: the old `ssm_blocks` was a `const` struct living
in external flash (XSPI) alongside the weight arrays it points to; the new
`ssm_blocks_working` is a plain mutable global in internal SRAM. The struct
itself only holds pointers, and the weight arrays it points to did not move
-- but `ssm_block_step` re-reads those pointer fields repeatedly inside its
loops, and in this Debug (`-O0`) build the compiler likely does not hoist
that load out of the loop. If so, every one of those repeated struct-field
reads moved from an external-flash access to an internal-SRAM access. Not
verified against a disassembly or an optimized build -- worth confirming
directly if this number is ever cited in the paper, so it is correctly
attributed to this refactor rather than to a later quantization scheme.

<!-- Claude comment: worth a two-minute check before writing this into the
paper as fact -- either build with -O1 and see if the gap closes (an
optimizer should hoist this regardless of which array the struct lives in,
which would confirm the -O0-specific hypothesis), or just diff the
disassembly of ssm_block_step's inner loop between the two versions. Cheap
to confirm, and "internal SRAM is faster than external flash for repeated
small reads" is exactly the kind of specific, checkable claim a reviewer
might ask about if it ends up in a results table. -->