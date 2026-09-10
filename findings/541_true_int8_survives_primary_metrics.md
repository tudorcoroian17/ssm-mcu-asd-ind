# Real int8 arithmetic (not fake-quant) survives the primary and deployment-target metrics

**Feeds:** `plans/detailed/510_nucleo_quant_deployment_matrix.md` (whose
"Open items" named true int8 arithmetic explicitly out of scope -- this
finding is what reopened that, and Phase 7/8 there now scope the C work);
`findings/522`/`523` (fake-quant already flagged `scan`'s tensors as the
hard part; this is a strictly harder test of exactly those tensors);
`findings/520`/`521` (the embedding-distance-vs-AUC precedent this result
extends a third time).

## Result

`checks/smoke/quant/true_int8_sim.py`/`true_int8_auc.py` simulate genuine
quantized inference -- int32-accumulate matmuls, LUT-based nonlinearities,
direct fixed-point discretize math -- through the entire backbone,
including the recurrence, which every prior scheme in this project
deliberately left in fp32. This is not fake-quant: no intermediate value is
computed in float and then rounded; the arithmetic chain itself is integer.

At `h` stored as int8, three of four distance heads hold within noise of
fp32; `euclidean` alone degrades. At `h` stored as int16, all four heads
hold, including `euclidean`. The two heads chosen for MCU deployment
(`knn_full`'s primary equivalent, `knn_clustered_16`) are unaffected at
either width -- the discretization tested here does not currently need
`euclidean`, and the h-width choice is therefore not blocking, only a
question of margin.

Scope: `f4cd557b7e3b`, case 1, run `16662b29beb3`. Embedding-parity check
on one clip; AUC check on the full train/test split (n_test=528).

## Method, briefly

- Weights: int8, per-tensor (same math as `checks/smoke/quant/weight_quant_parity.py`).
- Every matmul and the depthwise conv: int8 x int8 -> int64 accumulate (no
  overflow risk) -> rescale to real units -> requantize to int8 with a
  calibrated output scale.
- `softplus` and both `silu` usages: 256-entry LUTs, exact for any int8
  input, built from calibrated input/output scale pairs.
- Discretize (`A_bar`, `B_bar`): direct fixed-point math, not a LUT --
  `delta_q * A_real` and `delta_q * B_q` computed as genuine products,
  rescaled, then requantized.
- `h`: the swept tensor. int8 uses the full +/-127 range; int16 uses
  +/-32767, both derived from `ranges.json`'s calibrated `h` range.
- Calibration gap this required: `ranges.json` only recorded POST-activation
  ranges. Pre-activation ranges (needed for a LUT's input side) were
  captured via forward hooks on the existing `dt_proj`, `conv`, and each
  block's `norms[i]` submodules -- no `ssm_block.py` edits.
- Deliberate exceptions, not oversights: `RMSNorm`'s arithmetic (sum of
  squares, rsqrt) and log-mel extraction both stay fp32 -- no clean int8
  form exists for either, the same category of problem as an int8 FFT.
  Both operations' outputs are still quantized to int8 before the next
  stage.

## Embedding parity (one clip, ceiling probe before the AUC check)

| h width | max_abs_err | mean_abs_err |
| :--- | ---: | ---: |
| int8 | 6.01e-1 | 1.79e-1 |
| int16 | 3.55e-1 | 1.15e-1 |

Both roughly 5-8x the largest fake-quant figure in this project
(`findings/523`'s `3.80e-2`) -- expected, not alarming on its own: fake-quant
rounds a handful of named tensors once each and computes everything else in
exact fp32 between them; this quantizes dozens of intermediates per frame,
chained, none of which get to "average out" in float. Per `findings/520`'s
own precedent, embedding distance is not a verdict by itself -- the AUC
table below is.

## AUC/pAUC, both widths (full train/test split)

| head | int8 dAUC | int8 dpAUC | int16 dAUC | int16 dpAUC |
| :--- | ---: | ---: | ---: | ---: |
| euclidean | -0.0482 | -0.0247 | +0.0014 | +0.0273 |
| mahalanobis | +0.1318 | +0.1029 | +0.0403 | +0.0042 |
| knn_full | -0.0076 | -0.0064 | +0.0073 | +0.0376 |
| knn_clustered_16 | +0.0208 | -0.0126 | +0.0051 | +0.0288 |

`mahalanobis`'s swings are discounted for the same confirmed reason as
every prior scheme in this project (`findings/522`): `score_mahalanobis`'s
covariance inversion is ill-conditioned on this fold (>1e6, unconditional
print in `src/eval/auc_pauc.py`), and any perturbation, regardless of
source, is amplified unpredictably through it. The swing shrinking from
`int8` to `int16` (+0.1318 -> +0.0403) tracks the overall reduction in
embedding perturbation, consistent with that mechanism, not new information
about it.

`knn_full` and `knn_clustered_16` -- the primary metric and the deployment
target -- hold within noise at BOTH widths. `euclidean` is the only head
that fails at `int8` (-0.0482, well outside the noise band) and the only
one that needed `int16` to recover, fully (+0.0014).

## The precisely attributable finding

Comparing `euclidean`'s two rows in isolation from everything else that
changed between the two runs: the *only* variable swept was `h`'s storage
width. `euclidean` failed at the narrower width and fully recovered at the
wider one, while every other head was already fine at the narrower width
and stayed fine at the wider one. That isolates the failure to `h`'s
resolution specifically -- not the LUTs, not the chained matmuls, not the
discretize math -- a genuine "found the limit and know which knob controls
it" result, not a vague "it got worse."

## Consequence

For the two heads actually planned for deployment (`knn_full`/`knn_clustered_16`),
`h` at int8 is already sufficient -- `int16` is safety margin, not a
requirement, at a cost of roughly 4 KB extra on this board (trivial here,
worth re-checking against the RP2040's tighter budget once that board's
phase starts). This is the first evidence in this project that real,
chained integer arithmetic -- not just int8 rounding noise -- survives
through the full recurrence on the metrics this project actually depends
on. The C implementation scope for this is now tracked in
`plans/detailed/510_nucleo_quant_deployment_matrix.md`'s Phase 7.

## Open items

- **One fold**, as always -- case 1 only.
- **`euclidean`'s int16 recovery mechanism is inferred from covariation,
  not independently re-verified** -- a clean, well-supported inference, not
  a second confirmed experiment.
- **No C implementation yet.** Three real gaps before this can be ported:
  LUT export, calibration-constant export, and `h`'s storage type
  genuinely changing in `SSMBackbone_State` (not a macro swap, unlike every
  prior scheme). Scoped in the plan doc, not started.

<!-- Claude comment: the most citable single sentence from this finding,
if the paper needs one, is that the failure mode this project's hardest
theoretical worry (findings/150 section 6, the recurrence's sensitivity)
predicted turned out to be real, but narrow and fixable with a small,
well-understood change -- not the open-ended fragility that section
worried about. That is a better and more specific story than either "it
works" or "it's fragile" on their own. -->