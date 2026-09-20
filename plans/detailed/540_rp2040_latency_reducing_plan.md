# RP2040 latency reduction plan

Board: Arduino Nano RP2040 Connect (Cortex-M0+, 133 MHz, no FPU)
Model: `f4cd557b7e3b` (case 1, run `16662b29beb3`), selective SSM, `d_model=64`, `d_state=16`, `expand=1`, `d_conv=4`, `n_layers=2`, `dt_rank=4`, Euler discretization
Created: 2026-09-18
Status: Planning. No step is started.

## Goal

Reduce per-hop compute time on the RP2040 to 32 ms or less, which is the hop duration. If that isn't reachable, document the real-time factor and the trade-offs that lead to it.

Real-time factor (RTF) is compute time per hop divided by hop duration. An RTF of 1.0 or less means the board keeps up with the audio stream.

## Baseline

Per-hop timings measured on the RP2040 (hardware timer, microsecond resolution):

| Stage | fp32 | true_int8_h8 | Nucleo, for scale |
|---|---|---|---|
| Feature | 38.3 ms | 38.3 ms | ~0.5 ms |
| Normalize | 0.41 ms | 0.42 ms | — |
| Backbone | 124.5 ms | 74.5 ms | ~3.3–5.0 ms |
| Head (once per clip) | 298 µs | 365 µs | a few µs to tens of µs |

Per-hop totals (feature + normalize + backbone):

| Setup | Total per hop | RTF | Speedup needed |
|---|---|---|---|
| fp32 | 163.2 ms | 5.1 | 5.1× |
| true_int8_h8 | 113.2 ms | 3.5 | 3.5× |

## Findings so far

1. The feature stage alone (38.3 ms) exceeds the 32 ms budget. A backbone that takes 0 ms still misses the target, so the feature stage needs a fix regardless of what else changes.
2. The gap to the Nucleo isn't only clock speed. The clock ratio is about 4.5× (600 MHz versus 133 MHz). The feature stage is about 77× slower and the backbone 15–38× slower. The remainder likely comes from the H7's hardware FPU versus software floating point on the Cortex-M0+.
3. Matrix multiplies probably aren't the dominant backbone cost. Estimate, assuming the standard Mamba block layout: about 35K multiply-accumulates (MACs) per hop. At 74.5 ms (about 9.9M cycles) that's about 280 cycles per MAC. A tight scalar int8 loop on an M0+ takes roughly 5–8 cycles per MAC. The fp32-to-int8 speedup of only 1.67× (124.5 ms to 74.5 ms) is consistent with this.
4. The `h=int8` storage width breaks `euclidean` distance, and `h=int16` recovers it (see the Nucleo findings). The `true_int8_h8` timing may therefore not be a deployable configuration.

> **Claude comment:** Finding 3 is an estimate built from an assumed block layout. Verify the MAC count against `ssm_weights.c` before relying on it. If the estimate holds, kernel tuning is the wrong first target.

## Unverified assumptions

Resolve each one before acting on the steps that depend on it.

| Assumption | Affects | How to verify |
|---|---|---|
| The official Mbed core links slow libgcc soft-float routines, not the RP2040 bootrom routines | L1.3 | Check the linker map for the source of `__aeabi_fmul` and `expf` |
| `A = -exp(A_log)` is already precomputed in the RP2040 build | L1.2 | Grep the generated `ssm_weights.c` and `ssm_backbone.c` for `expf` |
| The Mbed core exposes the second core cleanly | L2.1 | Check whether core 1 can be launched from your firmware |
| Weights and hot code can be placed in SRAM with a section attribute | L1.4 | Read the core's linker script |
| The backbone has about 35K MACs per hop and about 35 KB of int8 weights | Finding 3 | Count from `ssm_weights.c` |
| FFT size and mel bin count | L1.1 | Read the feature-pipeline source |

## Plan

### Phase 0: Profile

Every later phase depends on this one.

**L0.1: Split backbone timing into sub-stages.**

- [ ] Add accumulated per-hop timers for: `in_proj`, `conv1d`, `x_proj`, `dt_proj`, `out_proj`, discretize, scan (state update plus `C·h`), nonlinearities (softplus, SiLU), and RMSNorm.
- [ ] Use the same hardware timer the firmware already uses.
- [ ] Transmit the sub-stage times over the existing protocol, or log them once per clip.
- [ ] Record results in the results log.

For example, if the scan and nonlinearities account for 50 ms and the matrix multiplies for 5 ms, skip L1.6 and prioritize L1.2 and L1.3.

**L0.2: Split feature timing into sub-stages.**

- [ ] Time window, RFFT, power spectrum, mel filterbank, and log separately.

**L0.3: Time `true_int8_h16` on the RP2040.**

- [ ] Flash and time the h16 setup so the comparison uses a configuration that passes the accuracy checks.

### Phase 1: Changes that need no retraining

**L1.1: Move the feature stage to fixed point.** Expected to be the largest single gain.

- Replace the fp32 chain with: q15 window, q15 real FFT (`arm_rfft_q15`), integer power spectrum, sparse integer mel filterbank, and integer log2 (`clz` plus a small lookup table).
- Estimate: single-digit milliseconds total, down from 38.3 ms.
- Risk: q15 FFT scales down at each stage, so quiet high-frequency bins lose resolution. `arm_rfft_q31` keeps more precision, but the M0+ has no 32×32→64 multiply instruction, so it's slower.
- Verify: compare log-mel output against the fp32 reference, then compare end-to-end AUC from `scores.csv`.

**L1.2: Precompute `A = -exp(A_log)`.** Skip if the assumption table shows it's already done.

- With `d_inner=64` and `d_state=16`, the fp32 path calls `expf` 1,024 times per layer, or 2,048 per hop.
- Estimate: at about 2,000 cycles per soft-float `expf`, that's about 4M cycles, or about 30 ms.
- Verify: re-run parity (reference: about 6e-4 max absolute error on `pooled_mean` against `parity_vectors_streaming.npz`).

**L1.3: Check which soft-float library the build links.**

- The RP2040 bootrom contains optimized routines for float arithmetic, `exp`, `log`, and `sqrt`. The Pico SDK and the arduino-pico core use them by default.
- Inspect the linker map. If the Mbed core uses the slower libgcc routines, evaluate calling the bootrom routines directly.

**L1.4: Place hot code and weights in SRAM.**

- Weights and code run from external flash through a 16 KB XIP cache. The int8 weights (about 35 KB, estimate) don't fit in it, so every hop streams them through the cache.
- The RP2040 has 264 KB of SRAM. Copy weights to RAM at startup and place the hot loops in RAM.
- Read the linker script before choosing an attribute.

**L1.5: Raise the optimization level on hot files.**

- Arduino builds default to `-Os`. Try `__attribute__((optimize("O3")))` on hot functions, or `#pragma GCC optimize("O3")` at the top of `ssm_backbone.c`.
- Measure the effect, because it varies.

**L1.6: Hand-tune the int8 kernels for the M0+.** Do this only if L0.1 shows the matrix multiplies matter.

- The M0+ has no SIMD and a single-cycle 32-bit multiply.
- Load four weights with one 32-bit `LDR`, unpack with `SXTB` and shifts, and unroll the loop.

**L1.7: Replace remaining fp32 operations in the int8 path.**

- Convert per-layer dequantization scales to fixed-point multiply-and-shift pairs.
- Avoid 64-bit intermediates, because each one calls a software multiply routine.

### Phase 2: Parallelism and clock

**L2.1: Use both cores.**

- Pipeline option: run feature extraction on core 1 while core 0 runs the backbone for the previous hop. Throughput becomes `max(feature, backbone)` per hop, with one hop of extra latency.
- Split option: split the scan by channel across cores, since the 64 channels are independent. `x_proj` needs a cross-channel reduction, so it needs a synchronization point.

**L2.2: Overclock (ablation only).**

- The RP2040 commonly runs at 200 MHz or more, for up to about 1.5× on everything.
- Check the flash clock divisor and the UART timing before trying it.
- Report overclocked results separately from stock-clock results.

> **Claude comment:** The thesis also reports power, and an overclock changes energy per hop. Keeping it out of the headline numbers avoids a muddled comparison with the other boards.

### Phase 3: Changes that need retraining

**L3.1: Reduce the hop rate.**

- Stack 2–4 consecutive feature frames into one model input, so the hop becomes 64–128 ms. Backbone cost per second of audio drops by the stacking factor.
- Alternatively, reduce `n_mels` or `n_fft`.

**L3.2: Shrink the model.**

- Matrix-multiply cost scales roughly with `d_model²` and scan cost with `d_state`. Candidates: `d_model=32`, `d_state=8`, or one layer.
- Use the existing sweep infrastructure to compare accuracy and footprint.

## Measurement protocol

Apply this to every step that changes the code:

1. Record the baseline timing for the current build.
2. Apply one change.
3. Re-run parity (numerically exact changes) or the accuracy comparison (numeric-format changes).
4. Re-run the timing harness.
5. Add a row to the results log, including the parity or AUC result.

Define the accepted accuracy loss for fixed-point features before starting L1.1: ______ (fill in).

## Results log

| Date | Step | Setup | Feature | Backbone | Total per hop | RTF | Parity or AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-18 | Baseline | fp32 | 38.3 ms | 124.5 ms | 163.2 ms | 5.1 | — | |
| 2026-09-18 | Baseline | true_int8_h8 | 38.3 ms | 74.5 ms | 113.2 ms | 3.5 | — | h8 breaks euclidean |

## Stop conditions

- **Success:** total per hop is 32 ms or less with accuracy within the accepted loss.
- **Fallback:** if Phases 0–2 don't reach 32 ms, report the best RTF and add L3.1 as a documented design trade-off.

## Open questions

1. What are the FFT size and mel bin count in the RP2040 feature pipeline?
2. Is the `A` precompute already in this build?
3. What accuracy loss is acceptable for fixed-point features?

## Claude comments

> **Claude comment:** Recommended order: L0.1 to L0.3, then L1.1, L1.2, L1.3, and L1.4. Together they attack the two likeliest bottlenecks, soft-float features and expensive exponentials, without touching the model.

> **Claude comment:** Reaching 32 ms at `d_model=64` with two layers is ambitious on a chip without an FPU. An honest report such as "1.4× real time, with these trade-offs" is a legitimate dissertation result. It's more informative than hiding the miss.

> **Claude comment:** If L1.3 shows the Mbed core uses slow soft-float and L2.1 turns out to be awkward on it, switching to the arduino-pico core would help with both. It's a heavy change, so consider it only if profiling shows soft-float dominates.