# RP2040 latency sweep on the Mbed core

Records the corrected RP2040 latency sweep: 100 setups on 4 models, run on the
Mbed core with the stock float library. It explains each result with the cost
model from finding 571. Finding 570 records the harness bug that made the first
sweep invalid. This sweep ran after the fix.

## Summary

- True int8 arithmetic is faster than fp32 on all four models. The speedup
  ranges from 1.5 times to 4.0 times, and it depends on the model.
- The deployed model (`16662b29beb3`) takes 126.3 ms per hop in `full_fp32`
  and 74.5 ms per hop in `true_int8_h8`. The speedup is 1.70 times.
- The hop period is 32 ms. The deployed `true_int8_h8` model needs 113.1 ms
  per hop (feature, normalize, and backbone). The real-time factor is 3.5.
- The int8 hop time is almost flat across the four models (74.5 to 86.8 ms for
  the `qab` and selective forms). All four models have 2,048 scan iterations
  per hop, and the scan is 76% of the int8 hop.
- `qabar` (classic models only) removes 2 of the 3 requantize calls from each
  scan iteration. It cuts the backbone time by 40% to 46%.
- Weight-only int8 quantization makes the RP2040 slower than fp32: 1.33 times
  slower for `w_proj` and 1.85 times slower for `w_all` on the deployed model.
  The cost model predicts these times within 2%.
- In `w_all` setups, `SSM_A` computes `-expf(...)` on every scan iteration.
  This costs about 60 ms per hop, which is 26% of the `w_all` hop.
- h16 is 1.4% to 3.4% slower than h8 on every model pair. The cost model
  predicted 0.15%. The cause is not identified.
- The feature stage takes 37.3 to 38.4 ms per hop in every setup. It is longer
  than the 32 ms hop period on its own.
- The first manual `full_fp32` reading in finding 570 (42.8M µs) was correct.
  The sweep gives 43.5M µs.

## What ran

| Item | Value |
| --- | --- |
| Harness | `test_harness/cycle_sweep_rp2040.py` in `ssm-mcu-asd-deploy`, with the fixed upload step (`--input-dir`, finding 570) |
| Results | `test_harness/output_rp2040_naive/cycle_sweep_rp2040_summary.csv`, one row per setup |
| Board and core | Arduino Nano RP2040 Connect, `arduino:mbed_nano` 4.4.1 |
| Toolchain | GCC 7.2.1, `-Os`, libgcc soft-float |
| Clock | 125 MHz nominal (124.79 MHz measured, finding 571) |
| Setups | 100, all with status `OK` |
| Clips per setup | 2 (`n_clips_ok` is 2 in every row) |
| Timer | `TIMERAWL`, 1 µs resolution, read around each stage |

The folder name marks these results as the baseline before any latency
optimization (plan 540).

### Models

| Hash | Recurrence | $D$ | $E$ | $N$ | $L$ | fp32 weights (bytes) | MACs per hop | Scan iterations per hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `16662b29beb3` | Selective (deployed) | 64 | 64 | 16 | 2 | 131,924 | 30,208 | 2,048 |
| `086acf0275b8` | Selective | 64 | 128 | 8 | 2 | 237,908 | 56,320 | 2,048 |
| `238a49a973a3` | Classic | 64 | 64 | 16 | 2 | 111,700 | 25,088 | 2,048 |
| `b77482e85dc3` | Classic | 64 | 128 | 8 | 2 | 213,460 | 50,176 | 2,048 |

$D$ is `SSM_D_MODEL`, $E$ is `SSM_D_INNER`, $N$ is `SSM_D_STATE`, and $L$ is
`SSM_N_LAYERS`. The two selective models also have `SSM_DT_RANK` = 4. The
classic models have no `x_proj` and no `dt_proj`, so their MAC count per layer
is $2DE + EK + ED$. The scan iteration count $L \cdot E \cdot N$ is 2,048 for
all four models.

### Setup families

| Prefix or suffix | Meaning |
| --- | --- |
| `full_fp32` | fp32 weights and fp32 arithmetic. The baseline. |
| `w_proj_*` | int8 weights for `in_proj`, `conv`, `x_proj`, `dt_proj`, `out_proj`, and the RMSNorm weights. Each weight is dequantized at each use, inside the MAC loop. Selective models only. |
| `w_all_*` | Everything in `w_proj_*`, and also `conv_b`, `dt_proj_b`, `D`, and `A_log`. The macro `SSM_A` computes `-expf(A_log)` at each use. |
| `pertensor`, `perchannel` | Scale granularity. |
| `_actboundaries` | Adds one `ssm_quant_dequant` call at each of four activation points. |
| `true_int8_h8`, `true_int8_h16` | Real int8 arithmetic. The suffix is the width of the recurrence state `h`. |
| `qab` (classic models) | `A_log`, `dt`, and `B` ship as int8. The firmware computes `A_bar` and `B_bar` on every frame. |
| `qabar` (classic models) | `A_bar` and `B_bar` are computed at export time and ship as int8. The firmware has no discretize step. |
| `euclidean_fp32`, `euclidean_int8`, `knn16_*` | Head variant. The head runs once per clip, at most 4.6 ms. |

## Measurement conventions

Per-hop values are the mean backbone time per clip divided by 344 hops. The
344 hops per clip is the value from plan 540. The firmware prints `hops=` on
USB `Serial`, so you can check it. A cycle count is derived as
$\text{cycles} = t_{\mu s} \cdot f_{\text{MHz}}$ with the nominal 125 MHz.

### Run-to-run variation

Builds that differ only in the head (same backbone source) show this spread in
`mean_backbone_us`:

| Backbone family | Spread |
| --- | --- |
| `true_int8_*` | 0.05% to 0.1% |
| `w_*` (fake-quant) | 0.1% to 0.2% |
| `full_fp32` | 0.5% to 1.7% |

A difference below about 2% between two fp32-family rows is not significant.
The feature stage, which has the same source in every setup, varies by 3%
(12.82M to 13.22M µs per clip).

**Claude comment:** My guess is that the variation comes from the code
placement in flash. Each setup relinks, so the feature and backbone code land
at different addresses, and the XIP cache handles them differently. This is not
tested.

## Results

Backbone time in ms per hop. Rows use `euclidean_int8` for `true_int8_*` and
`euclidean_fp32` for the other setups.

| Setup | `16662` (selective) | `086acf` (selective) | `238a49` (classic) | `b77482` (classic) |
| --- | --- | --- | --- | --- |
| `full_fp32` | 126.3 | 214.7 | 115.1 | 206.2 |
| `w_proj_pertensor` | 168.3 | 294.2 | not built | not built |
| `w_proj_perchannel` | 168.9 | 295.6 | not built | not built |
| `w_all_pertensor` (selective or `qab`) | 234.1 | 361.5 | 223.1 | 348.8 |
| `w_all_pertensor` (`qabar`) | not built | not built | 139.7 | 253.3 |
| `w_all_perchannel` (selective or `qab`) | 229.1 | 358.4 | 219.6 | 353.9 |
| `w_all_perchannel` (`qabar`) | not built | not built | 141.3 | 258.0 |
| `w_all_pertensor_actboundaries` (selective or `qab`) | 229.8 | 358.3 | 216.4 | 338.9 |
| `w_all_pertensor_actboundaries` (`qabar`) | not built | not built | 141.8 | 255.1 |
| `true_int8_h8` (selective or `qab`) | 74.5 | 86.4 | 75.0 | 86.8 |
| `true_int8_h8` (`qabar`) | not built | not built | 40.3 | 51.7 |
| `true_int8_h16` (selective or `qab`) | 76.8 | 88.9 | 76.4 | 88.0 |
| `true_int8_h16` (`qabar`) | not built | not built | 41.7 | 52.9 |

Speedup of true int8 over `full_fp32` on the same model:

| Model | h8 | h16 |
| --- | --- | --- |
| `16662` (selective) | 1.70 times | 1.65 times |
| `086acf` (selective) | 2.49 times | 2.41 times |
| `238a49` (classic), `qab` | 1.54 times | 1.51 times |
| `238a49` (classic), `qabar` | 2.86 times | 2.76 times |
| `b77482` (classic), `qab` | 2.38 times | 2.34 times |
| `b77482` (classic), `qabar` | 3.99 times | 3.90 times |

### Real-time factor

The real-time factor is the sum of the feature, normalize, and backbone times
per hop, divided by the 32 ms hop period. The head is excluded because it runs
once per clip.

| Model and setup | Feature | Normalize | Backbone | Total | Factor |
| --- | --- | --- | --- | --- | --- |
| `16662` `full_fp32` | 37.9 ms | 0.4 ms | 126.3 ms | 164.7 ms | 5.1 |
| `16662` `true_int8_h8` | 38.2 ms | 0.4 ms | 74.5 ms | 113.1 ms | 3.5 |
| `086acf` `true_int8_h8` | 37.5 ms | 0.4 ms | 86.4 ms | 124.3 ms | 3.9 |
| `238a49` `true_int8_h8_qabar` | 37.9 ms | 0.4 ms | 40.3 ms | 78.6 ms | 2.5 |
| `b77482` `true_int8_h8_qabar` | 38.0 ms | 0.4 ms | 51.7 ms | 90.1 ms | 2.8 |

The best configuration in the sweep is still 2.5 times slower than real time.

### Footprint of the deployed model

The budgets are 16,777,216 bytes of flash and 270,336 bytes of RAM. Finding 550
describes the method.

| Setup | Flash (bytes) | RAM (bytes) |
| --- | --- | --- |
| `full_fp32_euclidean_fp32` | 256,372 | 64,528 |
| `w_proj_pertensor_euclidean_fp32` | 165,392 | 64,528 |
| `w_all_pertensor_euclidean_fp32` | 158,240 | 64,528 |
| `true_int8_h8_euclidean_int8` | 166,340 | 57,240 |
| `true_int8_h16_euclidean_int8` | 166,352 | 59,288 |

## Findings

### True int8 is faster, and the speedup depends on the model

The int8 hop time is almost flat: 74.5 ms to 86.8 ms for the selective and
`qab` forms of all four models. The fp32 hop time is not flat: 115.1 ms to
214.7 ms. The reason follows from the cost model (finding 571):

- The fp32 hop is mostly float MACs. It grows with the MAC count. The two
  models with $E$ = 128 have about twice the MACs of the two with $E$ = 64.
- The int8 hop is mostly the scan loop, and every model has 2,048 scan
  iterations per hop. The integer MACs are a small part of the hop.

So the speedup is larger for the models with more MACs. This answers the open
question in finding 570 about whether the result depends on the model
dimensions. It does.

### `qabar` removes two requantize calls from each scan iteration

A `qab` scan iteration computes `A_bar` and `B_bar` and requantizes both. A
`qabar` iteration reads them from baked int8 arrays. It still requantizes
`new_h`. The cost model gives:

| Scan iteration (Mbed) | Cycles |
| --- | --- |
| `qab` (measured, finding 571) | 3,617 |
| `qabar` (model: 1 requantize, 4 `fmul`, 1 `fadd`, 2 `i2f`) | 1,561 |

The scan iteration is 2.32 times cheaper. The hop is 1.86 times faster on
`238a49` (75.0 to 40.3 ms) and 1.68 times faster on `b77482` (86.8 to 51.7 ms).
The hop gain is smaller because the MACs, the epilogue, and the requantize
calls outside the scan do not change.

`qabar` only exists for the classic (`selective=False`) models. In the deployed
selective model, `A_bar` depends on the input through `delta`, so it cannot be
computed at export time.

### Weight-only int8 makes this MCU slower

On the deployed model, `w_proj` takes 1.33 times as long as `full_fp32`, and
`w_all` takes 1.85 times as long. All four models show the same direction
(1.22 to 1.94 times slower for `w_all`). Two mechanisms explain it:

1. **Dequantization inside the MAC loop.** The macros are of the form
   `(float) w_q[i] * w_scale[0]`. Each weight use adds one `i2f` (69.13
   cycles) and one `fmul` (163.31 cycles), which is 232.44 cycles. The int8
   weights need less flash bandwidth than the fp32 weights, which gives back
   some of this cost. The model uses the warm-flash MAC cost (296.08 cycles)
   as the base.
2. **`expf` on every scan iteration (`w_all` only).** `SSM_A` expands to
   `-expf((float) A_log_q[...] * A_log_scale[0])`. It runs once per scan
   iteration: 2,048 times per hop. Each call costs about 3,682 cycles
   (`expf`, `i2f`, `fmul`). The total is 7.54M cycles, or 60.4 ms per hop.
   That is 26% of the `w_all` hop.

Model against sweep, on the deployed model:

| Setup | Sweep | Model | Difference |
| --- | --- | --- | --- |
| `w_proj_pertensor` | 168.3 ms | 168.5 ms | +0.1% |
| `w_all_pertensor` | 234.1 ms | 229.7 ms | −1.9% |
| `w_all_perchannel` | 229.1 ms | 229.7 ms | +0.3% |

**Claude comment:** Precomputing `A = -exp(A_log)` in the export script is an
exact substitution. It would remove 60 ms from the `w_all` hop (229 to about
169 ms). The `w_all` setup would still be 1.34 times slower than `full_fp32`.
The gain from weight-only quantization on this MCU is flash space (158,240
bytes against 256,372 bytes), not latency. Run the parity check (finding 542)
after you change the export script.

### Activation boundaries are not resolvable

The `_actboundaries` setups add 576 `ssm_quant_dequant` calls per hop on the
deployed model. At 780.89 cycles each, that is 0.45M cycles, or 3.6 ms. This is
1.6% of the `w_all` hop. The measured differences against `w_all_pertensor`
are −3.0% to +1.5% across the four models, with both signs. The effect is below
the run-to-run variation. The data are consistent with the model, but the
sweep cannot confirm the effect. This closes the open question in finding 570.

### h16 costs more than the cost model predicts

| Model and form | h8 (ms) | h16 (ms) | Difference | Cycles per scan iteration |
| --- | --- | --- | --- | --- |
| `16662` selective | 74.50 | 76.77 | +2.27 ms (+3.1%) | 139 |
| `086acf` selective | 86.39 | 88.91 | +2.52 ms (+2.9%) | 154 |
| `238a49` `qab` | 75.00 | 76.42 | +1.42 ms (+1.9%) | 87 |
| `238a49` `qabar` | 40.33 | 41.72 | +1.39 ms (+3.4%) | 85 |
| `b77482` `qab` | 86.76 | 87.99 | +1.23 ms (+1.4%) | 75 |
| `b77482` `qabar` | 51.69 | 52.93 | +1.23 ms (+2.4%) | 75 |

The microbenchmark predicted 5 cycles per scan iteration (+0.14%). The
observed values are 15 to 30 times larger. Observations:

- The sign is positive on all six pairs.
- For one model, `qab` and `qabar` give the same absolute penalty (1.42 and
  1.39 ms on `238a49`; 1.23 and 1.23 ms on `b77482`). So the cause is in code
  that both forms share, not in the `A_bar` and `B_bar` requantize calls.
- The penalty is larger for the selective models than for the classic models
  with the same dimensions.
- h16 adds 12 bytes of flash and 2,048 bytes of RAM on the deployed model.

**Claude comment:** My hypothesis is that soft-float cost depends on the size
of the operand. The microbenchmark used the same `s_h` for h8 and h16, so its
h16 states stayed small. Real h16 states can reach ±32,767. Finding 571
proposes a magnitude sweep to test this. The weight-only result gives weak
support. The dequantization cost measured in the sweep is about 172 cycles per
weight, and the model assumed 232. The `i2f` on an int8-range value may be
cheaper than on the ±2¹⁸ values that the microbenchmark used. This is not
tested either.

### The feature stage is now the largest unexplained item

The feature stage takes 37.3 to 38.4 ms per hop in every setup, about 4.7M
cycles. It is longer than the 32 ms hop period on its own. It uses the same
source in every setup, so the sweep gives no information on its internals.
Stage timers are the next step (plan 540, L0.2).

### The first manual fp32 reading was correct

Finding 570 records a first manual `full_fp32` reading of 42,817,694 µs and
treats it as an anomaly. The sweep gives 43,459,319 µs for the same setup,
which is 1.5% higher. That difference is inside the fp32 variation above. The
first reading was the real fp32 time. The later values of 25.6M µs came from a
stale `true_int8_h8` binary.

## Cost model against the sweep

The model is the method in finding 571, with the corrected channel epilogue
(4 `i2f`, 7 `fmul`, 1 `fadd` per channel). "Cold" uses the cold-flash MAC cost
and "warm" uses the warm-flash MAC cost. The difference is (model − sweep) /
sweep.

| Model | Setup | Sweep (ms) | Model, cold | Model, warm |
| --- | --- | --- | --- | --- |
| `16662` | `full_fp32` | 126.3 | 124.5 (−1.4%) | 111.9 (−11.4%) |
| `16662` | `true_int8_h8` | 74.5 | 77.5 (+4.1%) | 74.5 (0.0%) |
| `086acf` | `full_fp32` | 214.7 | 215.6 (+0.4%) | not used |
| `086acf` | `true_int8_h8` | 86.4 | 90.4 (+4.6%) | 84.7 (−2.0%) |
| `238a49` | `true_int8_h8` `qab` | 75.0 | 74.6 (−0.5%) | 72.1 (−3.9%) |
| `238a49` | `true_int8_h8` `qabar` | 40.3 | 40.9 (+1.4%) | 38.3 (−4.9%) |
| `b77482` | `true_int8_h8` `qab` | 86.8 | 86.3 (−0.5%) | 81.2 (−6.4%) |
| `b77482` | `true_int8_h8` `qabar` | 51.7 | 52.6 (+1.7%) | 47.5 (−8.1%) |

- The `full_fp32` rows use the cold model, because the fp32 weights (121 KB
  per hop) do not fit in the 16 KB cache.
- For the six int8 rows, the cold model is within −0.5% to +4.6% of the sweep.
- The model was built from the deployed model's microbenchmark costs. It
  predicts three other models with different dimensions and two recurrence
  forms.
- The classic `qab` and `qabar` predictions use the scan cost from the
  selective microbenchmark. A classic scan iteration has the same operations,
  so the cost is the same.
- The classic-model counts come from `ssm_true_int8_classic_src/ssm_backbone.c`.
  They are: MACs $L(2DE + EK + ED)$, requantize calls outside the scan
  $L(5E + 2D) + D$, and $3E + D$ output rescales per layer.
- The model does not account for the per-channel `delta_real` line when
  $N = 8$. In the microbenchmark it is spread over 16 iterations. The
  correction is +0.3%.

## Open questions

- **The cause of the h16 penalty.** See above. The next test is the magnitude
  sweep in the microbenchmark, then stage timers in the real firmware.
- **The XIP cache hit rate for the real weight access pattern.** It decides
  between the warm and cold predictions for the int8 rows.
- **The number of hops per clip.** The table uses 344. Check the `hops=` value
  that the firmware prints.
- **Classic `full_fp32` and `w_*` rows are not modeled.** I did not read the
  classic fp32 backbone source.
- **Two clips per setup.** The run-to-run variation for fp32 rows is up to
  1.7%. More clips would tighten the fp32 rows.
- **Power was not measured on the RP2040.** Finding 560 describes the Nucleo
  method only.

**Claude comment:** For the dissertation, quote latency in ms per hop and state
the core, the compiler, and the clock in the same table. The result that the
gain from quantization on an FPU-less core depends on how the requantize step
is computed is more useful than one speedup number. The requantize step needs
a float divide (69% of its cost on this core). A comparison with the
`arduino-pico` build (finding 571) would show that the float runtime, not the
integer arithmetic, sets the hop time.