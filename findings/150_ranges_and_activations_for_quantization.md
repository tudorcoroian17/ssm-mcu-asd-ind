# Findings 150: Activation Ranges for Quantization

**Status:** Phase 1 §1.5 deferred item, now complete.

**Scripts:** `checks/smoke/activation_ranges.py` (`RangeRecorder`, `run_case`). Edits to `src/models/ssm_block.py` (`discretize`, `_scan`, `forward` — optional `range_recorder`/`name_prefix` parameters) and `src/models/backbone.py` (`forward` — optional `range_recorder`).

**Model config (all four folds):** `d_model=64`, `d_state=32`, `d_conv=4`, `expand=2` (→ `d_inner=128`), `n_layers=4`, `selective=True`, `discretization='zoh'`, seed 158.

**Output:** `runs/case{N}/<hash>/ranges.json`, 27 named tensors per fold (4 blocks × `{A_bar, B_bar, delta, h, u_post_conv_silu, block_output}`, plus `model_input`, `final_norm_output`, `head_output`).

## 1. Purpose

Two different things get quantized in Phase 5, and only one needs this file. **Weights** are static — quantizable directly from `state_dict()`, no forward pass required. **Activations** are not — their range depends on what data flows through the model, and can only be measured by running representative clips through it. This document is that measurement.

It also targets a specific question directly. Master doc Section 11's central open item — *does the scan arithmetic survive int8* — is about four specific tensors: `delta`, `A_bar`, `B_bar`, `h`. `delta` feeds `exp()`/`expm1()`, both of which amplify quantization error near zero; `h` is the persistent state carried across an entire clip, so error there compounds rather than resetting each frame. Those four tensors are why this file exists; everything else recorded alongside them is useful context.

## 2. Method

- **Exact global min/max**, O(1) memory, tracked across the full calibration pass.
- **Percentiles from a bounded random sample** — 2,000 elements per batch, concatenated across the pass — since storing every element of tensors like `h` (`batch × d_inner × d_state` per timestep) across a full validation pass is gigabytes.
- **`h` sampled every 10th timestep**, not every timestep. Justified directly by `findings/130`'s decay-half-life results: state changes on the order of hundreds of milliseconds to seconds, so a 10-frame stride loses essentially nothing while cutting recorder overhead 10×.
- **Calibration pool: training-case validation-normal clips** (`fold['val']`, `label == 'normal'`) — the same pool `train.py` uses for early stopping, *not* the `calib_normal` held-out-case pool `findings/140` uses for threshold calibration. Deliberate: activation-range calibration should reflect data from machines the model has seen, mirroring how a real deployment would calibrate quantization before knowing which specific new unit it will run on. See Section 6 for why this choice needs a caveat given what `findings/140` found about cross-machine shift.
- **No retraining required** — loads existing seed-158 checkpoints directly.

## 3. Finding 1: `block_output` grows 8–18× with depth, and the growth is not fold-stable

`block_output` is what each layer adds into the residual stream — the tensor a quantized residual-add would actually operate on. `final_norm_output` stays tame (±3.5–4.5 in every fold) because RMSNorm divides by whatever RMS it's handed, but that normalization happens **once**, at the very end (`x = x + block(norm(x))` — the per-block `norm()` only ever sees the residual stream, never rescales what gets added to it). `block_output` itself passes through no such correction.

| case | block0 \|max\| | block1 | block2 | block3 | growth 0→3 |
|---|---|---|---|---|---|
| 1 | 4.23 | 6.71 | 17.20 | 49.80 | 11.8× |
| 2 | 3.98 | 3.44 | 43.00 | 71.46 | 17.9× |
| 3 | 5.97 | 20.68 | 6.50 | 96.39 | 16.1× |
| 4 | 4.50 | 12.90 | 14.44 | 33.67 | 7.5× |

`block3` additionally varies **2.33×** across folds on its positive extreme (max 15.2–35.3) and its negative tail is genuinely populated, not a single outlier — `p1` sits at 32–63% of the minimum in every fold (e.g., case3: `p1 = -39.77`, `min = -96.39`), meaning at least 1% of all sampled elements are in that extreme range, not one rare frame.

**Consequence:** a single quantization scale, whether shared across layers or fixed from one fold's calibration, is wrong in one direction or the other — sized for block0's real ±5 range, it clips block3 catastrophically; sized for block3's ±96, it discards nearly all of block0's resolution. **Recommend per-layer, per-fold-margined quantization scales for `block_output` specifically**, not a shared scheme. This is new information the project didn't have before this file existed — nothing measured up to Phase 1 predicted it.

## 4. Finding 2: `delta`'s dynamic range does not fit in 8 bits, even robustly

Excluding the extreme 1% (p1–p99 ratio):

| block | delta p99, range across 4 folds | typical dynamic range |
|---|---|---|
| block0 | 3.89 – 5.38 | ~10–11 bits |
| block1 | 2.17 – 3.65 | ~5–7 bits |
| block2 | 6.94 – 8.48 | ~12–13 bits |
| block3 | 6.40 – 8.77 | ~10–15 bits (case4 block3 reaches ~15 bits) |

Blocks 0, 2, and 3 all need 10+ bits of dynamic range in the middle 98% of values alone, headed for a tensor with 8. **A single uniform int8 scale per `delta` tensor will lose most of its resolution at one end of the range or the other**, in three of the four layers, in every fold checked. Candidates worth testing in Phase 5, roughly in order of implementation cost: per-channel rather than per-tensor quantization (individual `d_inner` channels may span a much narrower range than the tensor as a whole), or a log/asymmetric scheme specific to this one tensor.

## 5. Cross-tensor pattern: block1 is consistently the mildest layer

Not previously connected across the two tensors it shows up in. `delta`'s dynamic range (Section 4) and `A_bar`'s tail extent (Section 6) both single out block1 as the best-behaved layer, in every one of the four folds, independently:

| block | delta p99 range (4 folds) | A_bar p99 range (4 folds) |
|---|---|---|
| block0 | 3.89 – 5.38 | 0.9708 – 0.9782 |
| **block1** | **2.17 – 3.65** | **0.7756 – 0.9044** |
| block2 | 6.94 – 8.48 | 0.9819 – 0.9909 |
| block3 | 6.40 – 8.77 | 0.9672 – 0.9968 |

*Claude comment: this connection isn't something either metric showed on its own — it only shows up putting the two side by side while writing this document. Two independent tensors agreeing on which layer is "calmer," across all four folds, is a much stronger signal than either alone. Worth investigating what block1 is doing differently (a less selective regime, a different effective dt-rank utilization, something else) before Phase 4 — if it's structural, it might be the natural layer to quantize least conservatively, or a lead on why the other three behave the way they do.*

## 6. Finding 3: `A_bar`'s median-tail imbalance is a risk to the state-persistence result specifically

`A_bar = exp(deltaA)` is bounded to (0, 1] by construction, so it looks like the easy tensor — fixed range, no outlier problem. The distribution inside that range is not uniform, though: median sits low (0.001–0.15 depending on block/fold), while p99 climbs to 0.78–0.997. A uniform 256-level quantizer over [0, 1] gives ≈0.0039 resolution per step everywhere — which is a problem specifically near `A_bar ≈ 1`, because the relationship between `A_bar` and effective half-life is highly nonlinear there (`half_life ∝ 1/(1 − A_bar)`): a single quantization step near 0.999 corresponds to a far larger jump in effective time constant than the same step near 0.1.

This matters because `findings/130`'s `zero_state.py` and `decay_half_life.py` diagnostics already proved this exact mechanism is load-bearing — skill collapses from +0.54 to between −2.3 and −3.3 when the state is zeroed, specifically because some channels sit with `A_bar` very close to 1 (half-lives reaching 0.3–15 seconds, one channel past 100 seconds). If int8 quantization coarsens resolution right at that boundary, it could quietly degrade the long-half-life channels toward the fast-decay ones without breaking the model outright.

**Recommendation — the single most important check to run once Phase 4 has a quantized forward pass: re-run `decay_half_life.py` against the quantized model and confirm the tail half-lives survive.** Cheap, decisive, and directly tests the risk this section identifies rather than leaving it theoretical.

## 7. Finding 4 (bonus — tangential to quantization): a possible dead-unit pattern in `u_post_conv_silu`

Every block, every fold, the minimum is **−0.2785** to four decimal places — SiLU's exact theoretical floor (SiLU(x) = x·sigmoid(x) bottoms out at x ≈ −1.278). `p1` sits at the same floor in every single row, meaning at least 1% of sampled values — in every layer, every fold — are pinned there, not just occasionally dipping near it.

Consistent with a genuine dead-unit pattern: SiLU is asymptotically flat approaching 0 as x → −∞, so a channel driven to x = −2 and one driven to x = −10 by the conv are indistinguishable downstream, contributing nothing but a fixed offset. Separate from the quantization question this document is otherwise organized around, but a legitimate side-finding worth a cheap follow-up: print per-channel (not tensor-wide) min/max for this tensor and check whether it's a small fixed set of channels permanently collapsed versus a wider rotating set. If it's the former, that's a capacity/pruning lead worth having if Phase 4's memory budget gets tight.

## 8. What's not a concern

`model_input`, `final_norm_output`, and `head_output` are modest and consistent across all four folds — no fold-to-fold volatility, no extreme tails, nothing flagged. These are the tensors this exercise confirms are *not* the risk; Phase 5 can treat them with a standard per-tensor scheme without the caveats above.

## 9. Recommendations for Phase 4/5

1. Per-layer, per-fold-margined quantization scale for `block_output` — not shared across layers, not fixed from a single fold's calibration (Section 3).
2. Test per-channel or log-scale quantization for `delta` specifically before falling back to a shared per-tensor scheme (Section 4).
3. **Decisive check, do this first once a quantized forward pass exists:** re-run `decay_half_life.py` against it, confirm long-tail half-lives survive (Section 6).
4. Optional, not blocking: per-channel min/max on `u_post_conv_silu` to characterize the possible dead-unit pattern (Section 7).
5. Investigate block1's consistent mildness across `delta` and `A_bar` — may inform which layer(s) can tolerate more aggressive quantization (Section 5).

## 10. Open items

- **Calibration used training-case data, not held-out-case data — and `findings/140` already showed this distinction matters for final scores.** `findings/140` Section 3 found `mean`-pooling Mahalanobis scores shift 2.3–3.9× between training-case and held-out-case normals, and kNN distances shift up to 6× — both driven by internal representation differences between machines the model has and hasn't seen. Nothing in this document rules out `block_output` or `delta` showing a similar or larger shift on genuinely unseen-machine data. Given `block_output` already varies 2.33× fold-to-fold on *training*-case calibration alone (Section 3), the true worst-case range Phase 5 needs to handle on deployment hardware could be wider than what's captured here. Worth one confirmatory pass calibrating on `calib_normal` (the held-out-case pool `findings/140` defined) to check whether the ranges materially change before trusting these as final quantization bounds.
- Per-channel dynamic range not yet measured for any tensor — this document only reports per-tensor statistics. Given `delta`'s per-tensor range already looks unworkable for uniform int8 (Section 4), per-channel analysis is the natural next step rather than a nice-to-have.
- The `u_post_conv_silu` dead-unit follow-up (Section 7) has not been run.