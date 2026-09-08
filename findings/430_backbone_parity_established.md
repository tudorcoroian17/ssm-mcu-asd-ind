# Backbone parity established on Nucleo-H7S3L8

**Feeds:** `05_phase_4_backbone_port.md` step 4 ("parity test against parity_vectors.npz").

## Result

The fp32 C port of `f4cd557b7e3b` (case 1, run `16662b29beb3`) matches the PyTorch streaming
reference to a max absolute error of `5.99e-4` and a mean absolute error of `1.45e-4` on
`pooled_mean`, against embedding values in the roughly [-2, 2] range. Compared clip:
`1200010003_ToyCar_case2_normal_IND_ch1_0003.wav`, the exact clip
`parity_vectors_streaming.npz` was recorded from.

## What it took to get here

The first attempt produced a systematic mismatch (mean absolute error 1.2, every one of the 64
on-device values negative against a mixed-sign reference). Root cause: the C port fed raw log-mel
frames into the backbone. The model runs on `apply_normalization(log_mel, mean, std)`
(`src/features/baselines.py`), a per-channel z-score using that fold's training-data statistics
(`src/features/stats.py`) -- not on raw log-mel. `src/eval/parity_vectors.py` applies this before
ever calling the model; the C port had no equivalent step. Adding `SSMBackbone_NormalizeFrame()`
(exported `norm_mean`/`norm_std` computed the same way `parity_vectors.py` computes them, so they
match that fold's frozen `.npz` exactly) resolved it in one fix.

## What this number does and doesn't mean

`findings/250`'s 0.2627 tolerance is in log-mel units, derived for the Phase 3 feature-pipeline
parity check -- a different pipeline stage, not a valid comparison point for this embedding-space
error. There is currently no `findings/250`-equivalent tolerance for backbone-level error (a
noise-injection sweep showing where embedding perturbation starts moving AUC, analogous to
`findings/250`'s method). This result is judged instead by scale (0.03% of the embedding's own
range) and by shape (small, mixed-sign, scattered error -- consistent with float32
implementation-order differences between PyTorch and hand-written C, not a remaining bug).

## Bottom line

Phase 4's core question -- does the fp32 C port reproduce the trained backbone -- is answered:
yes, to a precision consistent with ordinary cross-implementation float32 arithmetic. Step 5
(measure flash/RAM/cycles) is next.

**Claude comment:** an embedding-level tolerance analogous to `findings/250` would be worth
deriving eventually, but I'd wait for Phase 5: quantization will add its own error on top, and a
tolerance set now, before that second error source exists, would likely need re-deriving anyway
once both are in the picture together.