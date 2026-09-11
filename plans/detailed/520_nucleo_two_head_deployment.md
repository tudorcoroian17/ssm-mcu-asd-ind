# Phase 5: two-head deployment on Nucleo-H7S3L8

## Goal

Deploy two quantized backbones on the Nucleo, each producing on-device
euclidean and knn_16 scores plus a thresholded decision:

1. **`weight_act_boundaries_ranges`** (already flashed, backbone-only):
   weight int8 all-tensors per-tensor + activation-boundaries int8. Reuses
   the canonical fake-quant `ssm_backbone.c`.
2. **`true_int8`** (does not exist as C yet): real int32-accumulate
   arithmetic, LUT-based nonlinearities, `h` storage width selectable
   between int8 and int16 via a `#define` in the generated header, one
   canonical `.c` source for both widths -- same convention
   `export_ssm_weights.py` already uses for every other scheme-specific
   decision.

Both heads are scored **on-device**: centroid + 16 KMeans cluster centers
+ an EVT-derived threshold are baked in as C constants; the MCU outputs
the raw score (for AUC/pAUC, computed host-side by collecting scores
across the test set) and a thresholded decision (for on-device precision/
recall/accuracy/F1).

## Decisions locked in this session

- Scoring location: **on-device**, not host-side.
- `true_int8` ships **both** `h=int8` and `h=int16` as two deploy folders
  built from one canonical source, toggled by a `#define`.
- Threshold method: EVT/GPD tail (`src/eval/thresholds.py::evt_threshold`),
  fit on this scheme's `calib_normal` split (same held-out machine, not
  train, not the balanced test split). Target false-alarm rate defaults to
  `p99_equiv` (1%) -- the middle of this project's existing `FAR_BUDGETS`,
  and the most defensible default absent a stated deployment duty cycle.
  *Claude comment: flagging this as a default, not a decision you've made
  -- say if you want a different FAR budget.*
- Reference embeddings (centroid, cluster centers, calibration scores) are
  recomputed **under each scheme's own quantized forward pass** -- never
  reused from the fp32 baseline. Comparing a device's quantized query
  embedding against an fp32-derived reference would be a domain mismatch
  (flagged when this was first scoped).

## Steps

1. **Generic reference-head export** (`mcu/export_reference_heads.py`) --
   done this turn. Works for either scheme by computing full train/test/
   calib_normal embeddings under that scheme's exact quantization, fitting
   centroid + KMeans-16, computing the EVT threshold, and writing
   `ssm_head_ref.h/.c` + a diagnostics JSON into the scheme's deploy
   folder.
2. **Generic on-device distance-head module**
   (`mcu/ssm_head_src/ssm_distance_head.h/.c`) -- done this turn. Scheme-
   agnostic: takes a pooled embedding, returns both scores and both
   decisions. Copied into every scheme's deploy folder by the export
   script, same convention as `ssm_backbone_src`.
3. **Run step 1 for `weight_act_boundaries_ranges`.** Backbone parity for
   this scheme is already established (`reference_embedding.npy` exists).
   Only the head export is new.
4. **True-int8 C port** (`mcu/ssm_true_int8_src/ssm_backbone.h/.c`) --
   next turn. Ports `true_int8_sim.py`'s `quantized_block_step`/
   `quantized_forward` to C: int32-accumulate matmuls, two LUTs
   (softplus, SiLU) per layer, fixed-point `A_bar`/`B_bar`/`h` recurrence,
   `h` type and clip range chosen by `#define SSM_H_WIDTH_INT16`.
5. **True-int8 weight/scale/LUT export** (`mcu/export_true_int8_weights.py`)
   -- next turn. Mirrors `export_ssm_weights.py`'s structure but emits the
   much larger scale/LUT set `true_int8_sim.py` calibrates. Run twice
   (`--h-width int8`, `--h-width int16`) into two deploy folders.
6. **Run step 1 (reference heads) for both `true_int8` folders.**
7. **`main.c` wiring**, both schemes: after `SSMBackbone_GetPooled`, call
   `SSMDistanceHead_Score`, transmit `{euclidean_score, knn16_score,
   euclidean_decision, knn16_decision}` over UART instead of the raw
   embedding.
8. **On-device validation**: feed the full test split through
   `audio_sender`, collect device scores/decisions, recompute AUC/pAUC/
   precision/recall/accuracy/F1 host-side from the collected scores, diff
   against step 1/6's `head_export_diagnostics.json`.
9. **Findings docs**: one per scheme, numbered continuing from `531`
   (`findings/532_boundaries_head_deployment.md`,
   `findings/533_true_int8_head_deployment.md`), each documenting the
   export parameters, threshold, and on-device-vs-host metric comparison.

Steps 1-3 are ready now. Steps 4-9 are next.