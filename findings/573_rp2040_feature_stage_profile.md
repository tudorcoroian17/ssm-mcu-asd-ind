# RP2040 feature-extraction stage profile

Records the per-stage timing of the feature-extraction pipeline on the Arduino
Nano RP2040 Connect, on the Mbed core and on the `arduino-pico` core. It
describes the profiling sketch and script, gives the results, and checks them
against the cost model in finding 571. Finding 572 records the backbone sweep,
where the feature stage takes about 38 ms per hop in every setup.

## Summary

- The feature stage takes 36.84 ms per hop on the Mbed core and 21.12 ms per
  hop on the `arduino-pico` core, both at 125 MHz. The `arduino-pico` core is
  1.74 times faster.
- The 1024-point real FFT (`arm_rfft_fast_f32`) is 71.7% of the stage on Mbed
  (26.40 ms) and 77.4% on `arduino-pico` (16.35 ms). No other stage is larger
  than 10%.
- Both cores are longer than the 32 ms hop period, or close to it: Mbed by 15%
  and `arduino-pico` at 66% of it. On Mbed, the feature stage alone does not
  fit in one hop.
- The `logf` call costs about 4,350 cycles on Mbed and about 840 on
  `arduino-pico`. The 64 calls per hop are 6.2% and 2.2% of the stage.
- The stage times are reproducible. The mean total per hop differs by 0.1% on
  Mbed and 0.03% on `arduino-pico` between five clips.
- Operand size has a small effect. The `assemble` time rises with hop energy
  (correlation +0.8). The `rfft` time varies by 0.5% between hops on Mbed and
  0.3% on `arduino-pico`.
- The two builds give different bits in about 8% of the log-mel values.
  The largest difference is 9.54e-7 (1 to 2 float32 steps), and the mean
  difference is 3.3e-8. Against a float64 reference, both builds have a
  maximum error of about 2.3e-6 (2.5 to 5 float32 steps).
- The cost model from finding 571 predicts `assemble`, `power`, and `mel_dot`
  within 2% to 16% and `rfft` within 4% (Mbed) and 14% (`arduino-pico`).

## What ran

| Item | Value |
| --- | --- |
| Sketch | `ssm-mcu-asd-deploy/rp2040-feature-profile/` |
| Script | `ssm-mcu-asd-deploy/test_harness/feature_profile.py` |
| Results | `ssm-mcu-asd-deploy/test_harness/output_feature_profile/<label>/`, with the labels `mbed` and `pico` |
| Clips | 5 clips of `case1`, the first five test clips of model `16662b29beb3` in name order (0004, 0007, 0010, 0019, 0021) |
| Hops | 344 per clip, 1,720 per core |
| Clock | 125 MHz on both cores |
| Data link | D-SUN adapter on `Serial1`, 115,200 baud, one ACK per hop |
| Timer | `TIMERAWL`, 1 µs resolution |

The cores and the toolchains are the ones from finding 571 (Mbed: GCC 7.2.1,
libgcc float. `arduino-pico`: GCC 16.1.0, bootrom float). The sketch prints its
compiler version on USB `Serial` at start. The output files do not record it.

## How the profiler works

The profiler answers one question: where do the 36.84 ms of a hop go? It runs
only the feature pipeline, so it does not depend on the backbone, the head, or
the setup files of the sweep. The sweep keeps its own sketch folder untouched.

### Pipeline stages

The pipeline (`feature_pipeline.c`) turns one 512-sample hop into 64 log-mel
values. It keeps the previous hop, so each frame has 50% overlap. The constants
are in `mcu_feature_contract.h`:

| Symbol | Meaning | C constant | Value |
| --- | --- | --- | --- |
| $N_{\text{FFT}}$ | FFT size | `MEL_N_FFT` | 1024 |
| $H$ | Hop size | `MEL_HOP_LENGTH` | 512 |
| $M$ | Mel bins | `MEL_N_MELS` | 64 |
| $L$ | Non-zero mel weights | `mel_offsets[64]` | 997 |

The profiler times five stages:

| Stage | Code | Operations per hop |
| --- | --- | --- |
| `assemble` | `FeaturePipeline_BeginFrame` | For each of the 1,024 samples: 1 `i2f`, 1 scale by 2⁻¹⁵, 1 `fmul` by the Hann window |
| `rfft` | `arm_rfft_fast_f32` | One 1024-point real FFT |
| `power` | `arm_cmplx_mag_squared_f32` and the two edge bins | 511 bins × (2 `fmul`, 1 `fadd`), plus 2 `fmul` |
| `mel_dot` | 64 calls of `arm_dot_prod_f32` | $L$ = 997 float MACs |
| `mel_log` | 64 calls of `logf` | 64 × (1 `logf`, 1 `fadd`) |

The `assemble` stage divides by 32768.0f in the source. GCC compiles this as a
multiply, because 32768 is a power of two. The measured time confirms it. A
divide would cost 0.70M cycles, and the measurement is 0.45M.

### Timer marks

The timer marks are in a copy of `feature_pipeline.c`. Two macros from
`feature_profile.h` do all the work:

- `FP_START(v)` declares a timestamp variable `v` and reads the timer.
- `FP_LAP(stage, v)` reads the timer once. It adds the time since the last
  read of `v` to one entry of `fp_stage_us[]`, then resets `v`.

One timer read therefore ends one stage and starts the next. The mel loop has
two marks per bin: one after the dot product and one after `logf`. The stage
totals for the 64 bins are accumulated in the array. There are about 136 timer
reads per hop. The `other` row (0.02% of the total) is the time between the
first read and the last read that no stage claims. It shows that the marks do
not disturb the measurement.

`fp_now()` has a compiler barrier before and after the timer read. This stops
the compiler from moving ordinary loads and stores across the read.

Every line that the profiler added to `feature_pipeline.c` contains `FP_` or
`feature_profile.h`. The script uses this to detect drift (see below).

### Data flow

1. The host sends a clip with the same protocol as the sweep: a header
   (`AUDI` magic and sample count), then one hop at a time. After each
   hop, the board returns the 64 log-mel floats of that hop (256 bytes)
   and then `ACK!`.
2. For each hop, the sketch clears `fp_stage_us`, reads the timer, runs
   `FeaturePipeline_BeginFrame` and `FeaturePipeline_FinishFrame`, and reads
   the timer again. It stores the five stage times and the total (6 values of
   4 bytes) in a table of 352 rows. This is 8.4 KB of RAM.
3. Outside the timed region, the sketch updates a fingerprint of the output:
   an FNV-1a hash over the raw bytes of all log-mel floats, and a float sum of
   all log-mel values.
4. After the last hop, the sketch sends one packet: the magic `PROF`, the
   hop count, the stage count, the fingerprint, the float sum, and the
   table. The host recomputes the FNV-1a hash of all received log-mel
   frames and compares it with the fingerprint, which checks the
   transfer. It saves each clip as `<clip>_logmel.npy`.
5. The script writes one CSV row per hop and one summary per clip. The summary
   reports the mean, median, minimum, maximum, and standard deviation of each
   stage, its share of the total, its mean cycles, and the correlation between
   the stage time and the RMS of the hop.

The define `RESET_OVERLAP_PER_CLIP` in the sketch (default 1) clears the
overlap at the start of each clip with one all-zero hop through
`FeaturePipeline_BeginFrame`. Then frame 0 of every clip starts from zeros,
which is what the reference assumes. The deployed sketch does not do this.
The return of the log-mel frames adds about 22 ms per hop on the 115,200
baud link, outside the timed region.

### Drift check

The profiling sketch uses copies of four files from the deployed sketch and a
modified copy of `feature_pipeline.c`. The script compares them to the
deployed files before it runs. For `feature_pipeline.c`, it first removes the
lines that contain `FP_` or `feature_profile.h`. The check covers
`feature_pipeline.c`, `feature_pipeline.h`, `mcu_feature_contract.h`,
`audio_ingest.cpp`, and `audio_ingest.h`. It prints a warning that names any
file that differs. This catches the case in which the deployed feature code
changes after the profiler was built.

### How to reproduce

1. Copy `audio_ingest.cpp`, `audio_ingest.h`, `feature_pipeline.h`, and
   `mcu_feature_contract.h` from the deployed sketch to
   `rp2040-feature-profile/`. Add `feature_profile.h`, the modified
   `feature_pipeline.c`, and `rp2040-feature-profile.ino`.
2. Build and flash. For the Mbed core, use a fixed build path (finding 570):

```text
arduino-cli compile --fqbn arduino:mbed_nano:nanorp2040connect --build-path <build-dir> <sketch-dir>
arduino-cli upload -p <upload-port> --fqbn arduino:mbed_nano:nanorp2040connect --input-dir <build-dir> <sketch-dir>
```

   For the `arduino-pico` core, select the board and the CPU speed in the
   Arduino IDE.

3. Wait about 5 seconds for the board to boot. Run the script from
   `test_harness`, with one label for each build:

```text
python feature_profile.py --com-port <D-SUN port> --label mbed --n-clips 5
```

4. Compare the log-mel frames with the host reference:

```text
   python logmel_compare.py --label mbed --label pico
```

   One clip takes about 50 seconds, because each hop crosses the 115,200 baud
   link. Do not run the script during a cycle sweep. Both use the board and
   the ports.

## Results

Mean per hop over 1,720 hops. Cycles are milliseconds × 125 MHz.

| Stage | Mbed (ms) | Mbed (M cycles) | Mbed share | pico (ms) | pico (M cycles) | pico share | Mbed / pico |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `assemble` | 3.58 | 0.448 | 9.7% | 1.85 | 0.231 | 8.8% | 1.94 |
| `rfft` | 26.40 | 3.300 | 71.7% | 16.35 | 2.043 | 77.4% | 1.62 |
| `power` | 1.84 | 0.230 | 5.0% | 0.97 | 0.121 | 4.6% | 1.90 |
| `mel_dot` | 2.73 | 0.342 | 7.4% | 1.48 | 0.185 | 7.0% | 1.85 |
| `mel_log` | 2.28 | 0.285 | 6.2% | 0.47 | 0.059 | 2.2% | 4.84 |
| `other` | 0.007 | 0.001 | 0.02% | 0.005 | 0.001 | 0.02% | |
| **Total** | **36.84** | **4.605** | | **21.12** | **2.640** | | **1.74** |

The feature stage in the backbone sweep takes 37.3 to 38.4 ms per hop on Mbed
(finding 572). The profile total is 1% to 4% lower. The profiling build has a
different binary. The layout sensitivity in finding 571 is a possible cause. It
is not tested.

### Reproducibility between clips

Mean total per hop, in ms:

| Clip | Mbed | pico |
| --- | --- | --- |
| 0004 | 36.843 | 21.119 |
| 0007 | 36.861 | 21.117 |
| 0010 | 36.848 | 21.118 |
| 0019 | 36.831 | 21.118 |
| 0021 | 36.824 | 21.112 |
| Spread | 0.037 (0.1%) | 0.007 (0.03%) |

### Data dependence

| Stage | Mbed: std / mean | Mbed: corr with RMS | pico: std / mean | pico: corr with RMS |
| --- | --- | --- | --- | --- |
| `assemble` | 4.6% | +0.80 | 2.6% | +0.78 |
| `rfft` | 0.5% | +0.60 | 0.3% | −0.05 |
| `power` | 1.0% | +0.02 | 0.6% | −0.06 |
| `mel_dot` | 1.1% | −0.05 | 1.5% | −0.03 |
| `mel_log` | 1.2% | +0.05 | 1.6% | +0.37 |
| **Total** | 0.8% | +0.75 | 0.4% | +0.45 |

- The variation between hops is small. Only `assemble` shows a clear
  dependence on the hop energy, which matches the operand-size result in
  finding 571 (`i2f` and `fmul` cost more for larger values).
- The audio is very quiet. In the first 23 hops of clip 0004 the RMS is about
  1.6 LSB (int16 units). So these clips test the small-operand end. Louder
  audio can cost a little more in `assemble`. The size of the effect is 3 to 4
  cycles per sample at most.
- Some hops show an outlier of about +0.35 ms in the short stages. For example,
  the maximum `power` time on Mbed is 2.19 ms against a median of 1.84 ms.
  The size is about the same in three stages. An interrupt is a possible cause.
  It is not tested. The per-hop data keep these outliers. The microbenchmark
  removes them with its best-of-3 rule.

## Cost model for the feature stage

The method is the one in finding 571: count the operations in the source and
multiply by the measured cost of each. The "before the run" column holds the
predictions that I wrote before the measurement.

| Stage | Count | Unit cost, Mbed (cycles) | Before the run (M cycles) | Measured (M cycles) |
| --- | --- | --- | --- | --- |
| `assemble` | 1,024 samples | 395.75 (`i2f` 69.13, 2 × `fmul` 163.31) | 0.405 (or 0.70 if the divide stays) | 0.448 (+10.5% over the model) |
| `rfft` | about 23,040 operations | 138.0 (mean of `fmul` and `fadd`) | 3.3 to 3.6 (a guess for the remainder) | 3.300 (+3.8% over the operation-count model of 3.18) |
| `power` | 511 bins | 439.31 (2 × `fmul`, `fadd`) | 0.225 | 0.230 (+2.1%) |
| `mel_dot` | 997 MACs | 289.83 to 348.19 (warm to cold) | 0.29 | 0.342 (343 cycles per MAC) |
| `mel_log` | 64 calls | not in finding 571 | 0.15 to 0.25 (a guess) | 0.285 (4,460 cycles per call) |

And for the `arduino-pico` core, using the `arduino-pico` costs at 125 MHz:

| Stage | Count | Unit cost (cycles) | Model (M cycles) | Measured (M cycles) |
| --- | --- | --- | --- | --- |
| `assemble` | 1,024 samples | 195.27 (`i2f` 52.75, 2 × `fmul` 71.26) | 0.200 | 0.231 (+15.6%) |
| `rfft` | about 23,040 operations | 78.04 (mean of `fmul` and `fadd`) | 1.798 | 2.043 (+13.6%) |
| `power` | 511 bins | 227.34 | 0.116 | 0.121 (+4.0%) |
| `mel_dot` | 997 MACs | 167.5 (RAM) to 222.0 (cold) | 0.167 to 0.221 | 0.185 (185 cycles per MAC) |
| `mel_log` | 64 calls | not in finding 571 | not modeled | 0.059 (922 cycles per call) |

- **The FFT operation count.** A radix-2 complex FFT of 512 points has about
  $5 \cdot 512 \cdot \log_2 512 = 23{,}040$ real operations. CMSIS-DSP uses a
  mixed-radix algorithm and adds a real-input split, so the true count differs.
  The match of 3.8% and 13.6% shows that the FFT cost is consistent with about
  this many operations at the mean cost of an `fmul` and an `fadd`. It is not
  an exact model.
- **`assemble` and `power`.** The residual above the model is the loads, the
  stores, and the loop overhead. On Mbed it is 42 cycles per sample in
  `assemble`. The scan kernel in finding 571 shows a similar 45 cycles.
- **`mel_dot` on Mbed matches the cold-flash MAC row** (343.72 cycles, 0.3%
  from the measurement). I predicted the warm case. The `mel_values` table (997
  floats, 3,988 bytes) is read once per hop. The FFT and the window table
  evict it from the 16 KB XIP cache between hops. On `arduino-pico` the
  measurement is between the RAM and cold rows.
- **The `logf` cost.** Subtract the `fadd` and the loop overhead from the
  measured stage. This gives about 4,350 cycles on Mbed and about 840 on
  `arduino-pico`. On Mbed this is close to the `log1pf` row (4,711).

## Numerical comparison

The board sends back the log-mel frames. `logmel_compare.py` compares them
with two host references built from the same tables as the firmware:
`ref64` (float64 everywhere) and `ref32` (float32 samples, window, power, mel
sum, and log, with a float64 FFT). The values below are means over the five
clips. Log-mel values are between about −4 and −9.3. The float32 spacing is
4.77e-7 for magnitudes from 4 to 8 and 9.54e-7 from 8 to 16.

| Board against | Max abs | Mean abs | RMS | 99th percentile | Frame 0 max |
| --- | --- | --- | --- | --- | --- |
| `ref64` | 2.34e-6 | 1.75e-7 | 2.26e-7 | 5.88e-7 | 5.59e-7 |
| `ref32` | 2.48e-6 | 9.32e-8 | 2.20e-7 | 9.54e-7 | 9.54e-7 |

| Clip | Identical values, mbed against pico | Max abs difference | Mean abs difference |
| --- | --- | --- | --- |
| 0004 | 92.3% | 9.54e-7 | 3.11e-8 |
| 0007 | 91.8% | 9.54e-7 | 3.23e-8 |
| 0010 | 91.8% | 9.54e-7 | 3.37e-8 |
| 0019 | 92.2% | 9.54e-7 | 3.26e-8 |
| 0021 | 92.1% | 9.54e-7 | 3.51e-8 |

- Each build matches the ideal result to within 2.5 to 5 float32 steps at the
  maximum and 0.4 step on average. The float32 output precision sets the
  error. The FFT is not a visible error source.
- The two builds differ by 1 to 2 float32 steps in about 8% of the values.
  This explains the different fingerprints. Two things changed between the
  builds: the compiler (GCC 7.2.1 against 16.1.0) and the float library
  (libgcc against the bootrom routines). This comparison cannot separate them.
- The overlap was reset at the start of each clip. Frame 0 has the same error
  as the other frames.
- After division by `ssm_norm_std` (1.93 to 4.89), the largest error is
  about 1.2e-6 in normalized units.

## Effect on the real-time budget

The hop period is 32 ms. The deployed model is `true_int8_h8`. The `pico`
backbone figures are model predictions (finding 571), not measurements. The
normalize stage (0.4 ms on Mbed) is not in the table.

| Build | Feature | Backbone | Total per hop | Factor |
| --- | --- | --- | --- | --- |
| Mbed, 125 MHz (sweep, finding 572) | 38.2 ms | 74.5 ms | 113 ms | 3.5 |
| `pico`, 125 MHz | 21.1 ms | 38.5 ms (model) | about 60 ms | 1.9 |
| `pico`, 200 MHz (from the cycle counts) | 13.2 ms | 24.0 ms (model) | about 37 ms | 1.2 |

The 200 MHz row scales the cycle counts by 125 / 199.42. It is an overclock.
The RP2040 is rated for 133 MHz.

## Open questions

- **The 1% to 4% gap** between the profile total and the sweep's feature stage.
  A different code layout in flash is a possible cause. It is not tested.
- **The compiler versions** are not recorded in the output files. The script
  metadata should include them.
- **Interrupt outliers.** The cause of the +0.35 ms outliers is not
  identified.
- **A possible stale overlap.** `overlap` in `feature_pipeline.c` starts as
  zero, and nothing resets it between clips.

  **Claude comment:** `SSMBackbone_Reset` runs for each clip, and the feature
  pipeline has no reset function. If this is right, the first frame of every
  clip after the first uses the last hop of the previous clip instead of
  zeros. This affects 1 frame of 344, so the effect on the embedding is small.
  It could explain part of the 0.3% score difference between the RP2040 and
  the Nucleo. This is not verified. Check whether the Nucleo firmware resets
  the overlap. The profiler has the same behavior, and it does not matter for
  the timing.

## Fixed-point FFT: feasibility and Level 2 decision

The FFT is 71.7%-77.4% of the feature stage (see above), so it is the only
optimization target worth a fixed-point port. Tested with a self-contained
host script (`test_harness/fixed_point_feasibility.py`) against CMSIS-DSP's
own q15/q31 RFFT, calibrated against NumPy's `rfft` before trusting any
error number (layout and output scale both confirmed, not assumed).

Two-level rule (finding 250's tolerance methodology): Level 1 is log-mel
error against a float64 reference, screening threshold 0.01 max absolute
error. Level 2 is whether the deployed head's AUC moves, using the actual
baked centroid/threshold from `ssm_head_ref.c`, not a refit head.

### Level 1 — log-mel error (5 clips, all hops)

| Variant | max_abs | mean_abs | rms | p99_abs | Verdict |
| --- | --- | --- | --- | --- | --- |
| q15_plain | 6.565 | 1.073 | 2.125 | 5.685 | FAIL |
| q15_gain | 1.101 | 0.00586 | 0.0215 | 0.0625 | FAIL |
| q31_plain | 3.49e-4 | 3.76e-6 | 1.40e-5 | 6.08e-5 | PASS |
| q31_gain | 1.80e-5 | 1.23e-7 | 3.44e-7 | 1.10e-6 | PASS |

Per-frame gain scaling (power-of-two shift, 2-12 bits, median 3) is
necessary for q15: without it (`q15_plain`), the median FFT output peak is
76 out of a possible 32767 — most frames use under 1% of the format's
range. `q15_gain` fails the Level 1 bound on worst-case error (max_abs
1.10, vs. the 0.01 threshold) but its typical-case error is much smaller
(mean_abs 0.00586, under the threshold; rms 0.0215).

### Level 2 — AUC impact (40 clips, class-balanced, both backbones)

| Backbone | Set | auc_drop | max_rel_score | mean_rel_score | Verdict |
| --- | --- | --- | --- | --- | --- |
| fp32 | q31_plain | 0 | 1.0e-6 | 4.0e-7 | PASS |
| fp32 | q31_gain | 0 | 1.4e-7 | 4.3e-8 | PASS |
| fp32 | q15_plain | -0.0625 | 0.664 | 0.470 | PASS* |
| fp32 | q15_gain | -0.010 | 0.0241 | 0.00808 | PASS |
| int8_h8 | q31_plain | -0.0125 | 0.172 | 0.00751 | PASS |
| int8_h8 | q31_gain | 0 | 0.0231 | 0.000989 | PASS |
| int8_h8 | q15_plain | -0.145 | 0.787 | 0.383 | PASS* |
| int8_h8 | q15_gain | -0.0275 | 0.151 | 0.0510 | PASS |

`auc_drop` is `ref64_auc - set_auc`; negative means the set's AUC is higher
than the float64 reference's. `*`: see the Claude comment below —
`q15_plain`'s PASS is treated as an artifact of the verdict rule, not a
real pass.

**Decision:** `q15_gain` is requalified. It fails the conservative Level 1
bound but is AUC-neutral to AUC-improving on both backbones at Level 2,
with a moderate per-clip score perturbation (2%-15% relative to the
float64 reference, not the 25%-83% seen on `q15_plain`). Feature-pipeline
profiling on hardware moves forward on the q15 fixed-point FFT path
(gain-scaled), not q31.

**Claude comment:** worth flagging why I'd treat `q15_plain` differently
from `q15_gain` even though the verdict table marks both PASS. The verdict
rule (`fixed_point_level2.py`) marks any negative `auc_drop` as an
automatic PASS, with no separate check for whether the "improvement" itself
is noise. `q15_plain` has the worst log-mel error of anything tested and a
per-clip score perturbation of 47%-83% (mean) against the float64
reference — yet its AUC goes *up* on both backbones. On a 20-normal/
20-anomaly, 40-clip subset, a couple of clips crossing the ROC in the right
place can swing AUC by 0.05-0.15, so I read `q15_plain`'s result as noise
landing favorably on this split, not a genuine improvement. `q15_gain`'s
much smaller score perturbation makes its neutral-to-positive AUC result
more credible, which is the basis for requalifying it specifically (not
`q15_plain`). One more caveat: 40 clips is a thin sample for an AUC-equal
call either way — if the on-hardware q15 port becomes the committed path,
it's worth rerunning Level 2 on a larger clip subset before relying on this
number in the dissertation.

## What this finding does not show

- It does not show a measured latency of the whole hop on the `arduino-pico`
  core. The backbone figures in the last table are predictions.
- It does not show the effect of clip-to-clip loudness variation beyond the
  five profiled clips. Within those five, hops are not uniformly quiet: each
  clip has a run of near-silent hops (RMS 1.6-1.8 LSB) alongside a longer run
  of active hops (RMS 30-1070 LSB). An earlier version of this finding
  described the audio as uniformly quiet; that was wrong.

**Claude comment:** Three candidates for a later optimization, with their
size. None is tested.

1. The FFT is the only large target: 3.30M cycles on Mbed and 2.04M on
   `arduino-pico`. The other four stages together are 1.31M and 0.60M.
2. Each sample is converted from int16 to float twice: once as the new hop
   and once as the overlap in the next hop. Caching the converted float
   removes 512 `i2f` and 512 scale multiplies per hop. This is bit-exact. It
   saves about 0.12M cycles on Mbed (0.95 ms, 2.6% of the stage) and 0.06M on
   `arduino-pico` (0.51 ms, 2.4%). It needs 2 KB of RAM for the float copy.
3. A fixed-point FFT is faster, but it has a parity risk. Resolved below —
   see "Fixed-point FFT: feasibility and Level 2 decision".