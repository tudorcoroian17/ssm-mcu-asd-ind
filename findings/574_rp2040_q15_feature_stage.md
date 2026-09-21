# RP2040 q15 feature stage: latency and parity

Records the change of the feature stage on the Arduino Nano RP2040 Connect
from float32 to a q15 fixed-point FFT with a per-frame gain. It describes the
firmware change and the scripts, gives the timing on the Mbed core and on the
`arduino-pico` core, and checks the output against the host `q15_gain` variant
from finding 573. It closes the gain-shift coverage gap that the first parity
run left open. Finding 573 has the float32 baseline and the fixed-point
feasibility test.

## Summary

- The feature stage takes 8.90 ms per hop on the Mbed core and 5.39 ms per hop
  on the `arduino-pico` core, both at 125 MHz. In finding 573 the float32
  stage took 36.84 ms and 21.12 ms. The stage is 4.14 times faster on Mbed and
  3.92 times faster on `arduino-pico`.
- The 1024-point real FFT takes 0.338 M cycles on Mbed and 0.327 M on
  `arduino-pico`. The float32 FFT took 3.300 M and 2.043 M. The integer FFT
  costs about the same on both cores.
- On Mbed, the feature stage is now 28% of the 32 ms hop period. It was 115%.
  On `arduino-pico` it is 17%. It was 66%.
- The FFT is no longer the largest stage on Mbed. The three large stages are
  `mel_log` (30.8%), `rfft` (30.4%), and `mel_dot` (28.7%). On `arduino-pico`
  the FFT is still 48.6% of the stage.
- The board output matches the host `q15_gain` variant to 7.7e-07 on the five
  profiled clips (1,720 hops) and to 6.8e-07 on 214 synthetic hops. This is
  1 to 2 float32 steps.
- The synthetic clips cover all 16 gain shifts (0 to 15). The five profiled
  clips used shifts 2 to 12 only.
- The error of the board against the float64 reference equals the error of the
  host `q15_gain` variant in every printed digit. On the five clips the largest
  error is 1.101 and the mean is 5.86e-03, which reproduces the `q15_gain` row
  of finding 573.
- A full-scale synthetic clip (DC, Nyquist tone, sine, impulses, square wave,
  noise) has a mean error of 2.20 against the float64 reference. The host
  variant has the same error, so this comes from the q15 algorithm and not
  from the firmware.
- The stage time does not depend on the audio. The mean total per hop differs
  by 0.6% between the profiled clips and the synthetic clips.

## What ran

| Item | Value |
| --- | --- |
| Sketch | `ssm-mcu-asd-deploy/rp2040-feature-profile-q15/` |
| Window generator | `ssm-mcu-asd-deploy/test_harness/gen_q15_window.py` |
| Coverage clips | `ssm-mcu-asd-deploy/test_harness/make_q15_coverage_clips.py` |
| Parity script | `ssm-mcu-asd-deploy/test_harness/logmel_compare_q15.py` |
| Profiler | `ssm-mcu-asd-deploy/test_harness/feature_profile.py` (unchanged) |
| Results | `test_harness/output_feature_profile/<label>/`, with the labels `mbed_q15`, `mbed_q15_coverage`, and `pico_q15` |
| Clips (timing and parity) | The 5 clips of finding 573: 0004, 0007, 0010, 0019, 0021 of `case1`, model `16662b29beb3`. 344 hops per clip, 1,720 hops per core |
| Clips (coverage) | `coverage_gain_sweep.wav` (186 hops) and `coverage_full_scale.wav` (28 hops), Mbed only |
| Clock | 125 MHz on both cores |
| Overlap mode | `reset` (`RESET_OVERLAP_PER_CLIP 1`) |
| Timer | `TIMERAWL`, 1 µs resolution |
| CMSIS-DSP | Device: the library `CMSIS_DSP_RFFT`, vendored from the project checkout (`arm_rfft_q15.c` revision V1.9.0). Host: the `cmsisdsp` pip package, version 1.10.1 |

The cores and the toolchains are the ones from finding 573. The compiler
versions are still not recorded in the output files.

## How the q15 pipeline works

The sketch is a copy of `rp2040-feature-profile` with a new
`feature_pipeline.c` and a new header, `feature_window_q15.h`. The five stages
and their timer marks are the same as in finding 573, so the tables compare
one to one. `feature_profile.py` and `feature_profile.h` are unchanged.

### Recipe

The recipe is the `q15_gain` variant of `fixed_point_feasibility.py`. For each
1,024-sample frame (the previous hop and the new hop):

1. Find the peak $p$ of the frame. Pick the largest shift $k \le 15$ with
   $p \cdot 2^{k} \le 32767$. A peak of 0 gives $k = 15$.
2. Multiply each sample by $2^{k}$ and by the q15 window $w_i$, with rounding:
   $(x_i 2^{k} w_i + 2^{14}) \gg 15$. The result is saturated to `int16`.
3. Run `arm_rfft_q15`. The output is interleaved (re, im) pairs, scaled by
   $2^{-10}$ for 1,024 points. The host calibration gives layout `interleaved`,
   scale $2^{-10}$, and a calibration residual of 0.517%.
4. Compute the power of each bin in `uint32`: $\mathrm{re}^2 + \mathrm{im}^2$.
   Convert it to float32.
5. Sum the sparse mel weights in float32, then undo the FFT scale and the gain
   with one power-of-two factor, before the epsilon and the logarithm:

$$E_m = 2^{-10-2k} \sum_{b} w_{m,b} \left( \mathrm{re}_b^2 + \mathrm{im}_b^2 \right)$$

$$\mathrm{logmel}_m = \ln \left( E_m + \varepsilon \right), \quad \varepsilon = 10^{-6}$$

The factor $2^{-10-2k}$ is a power of two, so the multiplication is exact in
float32. The firmware reads it from a 16-entry table (`0x1p-10f` to
`0x1p-40f`).

The peak of the overlap is the peak of the previous hop. The firmware stores
it, so only the 512 new samples need a scan.

**Claude comment:** The theoretical bound of $\mathrm{re}^2 + \mathrm{im}^2$ is
$2^{31}$, which fits in `uint32`. A tighter bound is $2^{28}$. The magnitude of
each raw bin is at most $32768 \cdot 512 / 1024 = 2^{14}$, because the sum of
the window is 512. I derived this and did not measure it.

### Window table

`gen_q15_window.py` reads the float32 `hann_window[]` from
`mcu_feature_contract.h` and writes `hann_window_q15[1024]` to
`feature_window_q15.h`. The contract header stays unchanged. The formula is
$\mathrm{clip}(\mathrm{round}(h_i \cdot 32768), -32768, 32767)$, with NumPy
rounding (half to even), as in the host test.

| Property | Value |
| --- | --- |
| Entries | 1,024 |
| Clipped to 32767 | 3 (indices 511, 512, and 513) |
| Exact .5 rounding ties | 8 (4 values, each twice, because the window is symmetric) |
| Largest error against the float window | 3.052e-05 (1 step, at the clipped peak) |
| Symmetric around index 512 | Yes |

Indices 511 and 513 clip because $h = 0.999990588$ and
$0.999990588 \cdot 32768 = 32767.69$ rounds to 32768. The table is generated
and not computed at start-up, because C `roundf` rounds half away from zero.
It would differ from the host at up to 8 entries.

### Library and memory

The device library gained five files from the CMSIS-DSP checkout:
`arm_rfft_q15.c`, `arm_rfft_init_q15.c`, `arm_cfft_q15.c`,
`arm_cfft_radix4_q15.c`, and `arm_shift_q15.c`. The library already had the
full common tables and the constant structures. The firmware calls
`arm_rfft_init_1024_q15`, which does not keep the code for other FFT sizes.

The buffers are 2,048 bytes (`frame_q15`), 4,096 bytes (`fft_out_q15`), and
2,052 bytes (`power_spectrum`). The total is 8.2 KB, the same as the two
float32 buffers of finding 573 (8.0 KB).

## Results

Mean per hop over 1,720 hops. Cycles are milliseconds × 125 MHz.

### Mbed core

| Stage | fp32 (ms) | q15 (ms) | q15 (M cycles) | q15 share | Change |
| --- | --- | --- | --- | --- | --- |
| `assemble` | 3.58 | 0.557 | 0.070 | 6.3% | 6.4 times faster |
| `rfft` | 26.40 | 2.702 | 0.338 | 30.4% | 9.8 times faster |
| `power` | 1.84 | 0.340 | 0.042 | 3.8% | 5.4 times faster |
| `mel_dot` | 2.73 | 2.556 | 0.320 | 28.7% | 6% faster |
| `mel_log` | 2.28 | 2.740 | 0.343 | 30.8% | 20% slower |
| `other` | 0.007 | 0.007 | 0.001 | 0.08% | |
| **Total** | **36.84** | **8.902** | **1.113** | | **4.14 times faster** |

The total per hop ranges from 8.725 ms to 9.326 ms (standard deviation
0.083 ms).

### `arduino-pico` core

| Stage | fp32 (ms) | q15 (ms) | q15 (M cycles) | q15 share | Change |
| --- | --- | --- | --- | --- | --- |
| `assemble` | 1.85 | 0.471 | 0.059 | 8.7% | 3.9 times faster |
| `rfft` | 16.35 | 2.618 | 0.327 | 48.6% | 6.2 times faster |
| `power` | 0.97 | 0.288 | 0.036 | 5.3% | 3.4 times faster |
| `mel_dot` | 1.48 | 1.502 | 0.188 | 27.9% | 1.5% slower |
| `mel_log` | 0.47 | 0.508 | 0.064 | 9.4% | 8% slower |
| `other` | 0.005 | 0.002 | 0.000 | 0.05% | |
| **Total** | **21.12** | **5.389** | **0.674** | | **3.92 times faster** |

The total per hop ranges from 5.340 ms to 5.902 ms (standard deviation
0.031 ms).

### Predictions before the run

The predictions are for the Mbed core. I wrote them before the measurement.

| Stage | fp32 (M cycles) | Prediction (M cycles) | Measured (M cycles) | Result |
| --- | --- | --- | --- | --- |
| `assemble` | 0.448 | 0.03 to 0.05 | 0.070 | Above the range |
| `rfft` | 3.300 | 0.2 to 0.5 | 0.338 | Inside |
| `power` | 0.230 | 0.05 to 0.08 | 0.042 | Below the range |
| `mel_dot` | 0.342 | about 0.34 | 0.320 | 6% below |
| `mel_log` | 0.285 | about 0.29 | 0.343 | 18% above |
| **Total** | 4.605 | 0.95 to 1.25 | 1.113 | Inside |

The `arduino-pico` core had no written predictions. I did not extend the cost
model of finding 571 to the q15 stages.

- `assemble` costs about 68 cycles per windowed sample (0.070 M / 1,024
  samples), including the peak scan and the copy of the overlap.
- `power` costs about 82 cycles per bin (0.042 M / 513 bins).
- `mel_dot` sums the same 997 values as before. The operands are now unscaled
  integers converted to float, which can change the cost of each operation
  (finding 571). The stage is 6% faster on Mbed and 1.5% slower on
  `arduino-pico`.

**Claude comment:** `mel_log` is the only stage that got slower on both cores,
and the time now correlates with the hop energy on Mbed (+0.449, against +0.05
in finding 573). The argument of `logf` is the same as before, up to
rounding, so the operand size does not explain the change. The 64 extra
multiplications cost about 0.005 M cycles, which explains less than a tenth of
the 0.058 M increase. A different code layout in flash is a candidate, as in
finding 571. I did not test it.

### Data dependence

| Stage | Mbed: corr with RMS | pico: corr with RMS |
| --- | --- | --- |
| `assemble` | +0.164 | +0.039 |
| `rfft` | -0.016 | -0.011 |
| `power` | +0.066 | -0.086 |
| `mel_dot` | +0.017 | -0.033 |
| `mel_log` | +0.449 | +0.227 |
| **Total** | +0.243 | +0.044 |

The total per hop varies by 0.9% (Mbed) and 0.6% (`arduino-pico`) between hops.
On the 214 synthetic hops the Mbed mean total is 8.850 ms, against 8.902 ms on
the profiled clips. The lowest values are in the synthetic clips: `mel_dot`
reaches 1.893 ms and `power` reaches 0.200 ms.

**Claude comment:** I think these minimums are the digital-silence hops, where
all energies are zero. I did not check this.

## Parity with the host `q15_gain` variant

`logmel_compare_q15.py` builds the host `q15_gain` log-mel frames with the
CMSIS q15 code of the `cmsisdsp` package. It compares them with the frames of
the board. The host uses float64 for the power, the mel sum, and the
logarithm. The board uses `uint32` power and float32. The differences should be
at the level of float32 rounding. The threshold is 1e-05. Log-mel values are
between about -13.8 and 0 in the synthetic clips and between about -9.3 and -4
in the profiled clips. The float32 spacing is 4.77e-07 for magnitudes from 4
to 8 and 9.54e-07 from 8 to 16.

### Five profiled clips (Mbed, 1,720 hops)

| Clip | Max abs diff | Mean abs diff | 99th percentile | Worst hop (k) |
| --- | --- | --- | --- | --- |
| 0004 | 7.005e-07 | 1.438e-07 | 4.749e-07 | 55 (4) |
| 0007 | 6.941e-07 | 1.418e-07 | 4.757e-07 | 285 (3) |
| 0010 | 6.762e-07 | 1.430e-07 | 4.759e-07 | 289 (4) |
| 0019 | 6.866e-07 | 1.462e-07 | 4.776e-07 | 61 (4) |
| 0021 | 7.657e-07 | 1.486e-07 | 4.791e-07 | 230 (3) |

The gain shifts in these clips are 2 to 12. Hops per shift: 2: 104, 3: 1,039,
4: 164, 5: 37, 6: 4, 7: 11, 8: 7, 9: 3, 10: 5, 11: 5, 12: 341.

Error against the float64 reference (`ref64`):

| Clip | Board max | Board mean | Host `q15_gain` max | Host `q15_gain` mean |
| --- | --- | --- | --- | --- |
| 0004 | 0.9236 | 5.400e-03 | 0.9236 | 5.400e-03 |
| 0007 | 0.9398 | 5.488e-03 | 0.9398 | 5.488e-03 |
| 0010 | 1.101 | 6.683e-03 | 1.101 | 6.683e-03 |
| 0019 | 0.9635 | 5.401e-03 | 0.9635 | 5.401e-03 |
| 0021 | 1.057 | 6.339e-03 | 1.057 | 6.339e-03 |

The mean over the five clips is 5.86e-03 and the largest value is 1.101. Both
equal the `q15_gain` row of the Level 1 table in finding 573. This shows that
the parity script uses the same clips as finding 573.

### Synthetic clips (Mbed, 214 hops)

`make_q15_coverage_clips.py` writes two 16 kHz PCM_16 clips. The script reads
them back with the profiler's loader and checks that the `int16` samples are
identical.

- `coverage_gain_sweep.wav` has 3 hops for each of 31 peak levels: 0, every
  $2^m$ (m = 0 to 15), and every $2^m - 1$ (m = 1 to 15). The levels run
  upward, then downward. Each hop is uniform random `int16` values with one
  sample forced to the exact peak. A peak of 32768 is the value -32768.
- `coverage_full_scale.wav` has 7 signals of 4 hops each: DC at +32767 and
  at -32768, a Nyquist tone that alternates between 32767 and -32768, a sine
  at bin 100.5, an impulse train, a square wave, and white noise over the
  full `int16` range.

| Clip | Max abs diff | Mean abs diff | 99th percentile | Worst hop (k) |
| --- | --- | --- | --- | --- |
| `coverage_gain_sweep` | 6.770e-07 | 1.637e-07 | 4.980e-07 | 146 (8) |
| `coverage_full_scale` | 6.282e-07 | 1.440e-07 | 5.026e-07 | 14 (0) |

| Gain shift k | Hops | Max abs diff |
| --- | --- | --- |
| 0 | 47 | 6.282e-07 |
| 1 | 12 | 3.420e-07 |
| 2 | 12 | 2.846e-07 |
| 3 | 12 | 3.491e-07 |
| 4 | 12 | 3.652e-07 |
| 5 | 12 | 4.296e-07 |
| 6 | 12 | 5.164e-07 |
| 7 | 12 | 6.585e-07 |
| 8 | 12 | 6.770e-07 |
| 9 | 12 | 6.672e-07 |
| 10 | 12 | 5.628e-07 |
| 11 | 12 | 5.706e-07 |
| 12 | 12 | 5.347e-07 |
| 13 | 12 | 5.281e-07 |
| 14 | 6 | 5.124e-07 |
| 15 | 5 | 1.893e-07 |

Only 5 hops have k = 15, not 6. The first hop after the downward pass has an
overlap peak of 1, so its shift is 14.

- The largest difference is 6.770e-07, which is 1 to 2 float32 steps. All 16
  gain shifts agree with the host. The shifts that the profiled clips never
  used (0, 1, 13, 14, and 15) show no divergence.
- The extreme samples cause no divergence: -32768, DC at both signs, the
  Nyquist tone, and impulses at full scale. So the `int32` products in the
  window multiplication and the saturation are consistent with the host.
- The device library (revision V1.9.0 in the file header) and the host package
  (1.10.1) give the same result on all 1,934 hops that this finding compares.

### Error against the float64 reference on the synthetic clips

| Clip | Board max | Board mean | Host `q15_gain` max | Host `q15_gain` mean |
| --- | --- | --- | --- | --- |
| `coverage_gain_sweep` | 3.441e-02 | 1.450e-03 | 3.441e-02 | 1.450e-03 |
| `coverage_full_scale` | 7.007 | 2.202 | 7.007 | 2.202 |

The board and the host have the same error, so the large error on
`coverage_full_scale` is a property of the q15 algorithm on these signals.

**Claude comment:** My hypothesis is a noise floor in the q15 FFT. A pure tone,
DC, or a square wave leaves most mel bins with almost no true energy. The
float64 reference then gives $\ln(\varepsilon) \approx -13.8$. The rounding
noise of the q15 FFT lifts these bins above $\varepsilon$. The broadband noise
clip has a much smaller error (max 0.034), which fits this. It also fits the
Level 1 result of finding 573 (max 1.10, mean 0.006). I did not test the
hypothesis.

## Effect on the real-time budget

The hop period is 32 ms. The feature figures are measured. The Mbed backbone
figure is from the sweep of finding 572. The `arduino-pico` backbone figures
are model predictions (finding 571). The Mbed total includes the normalize
stage (0.4 ms).

| Build | Feature | Backbone | Total per hop | Factor | Factor in finding 573 |
| --- | --- | --- | --- | --- | --- |
| Mbed, 125 MHz | 8.9 ms | 74.5 ms | about 84 ms | 2.6 | 3.5 |
| `pico`, 125 MHz | 5.4 ms | 38.5 ms (model) | about 44 ms | 1.4 | 1.9 |
| `pico`, 200 MHz (from the cycle counts) | 3.4 ms | 24.0 ms (model) | about 27 ms | 0.86 | 1.2 |

The 200 MHz row scales the cycle counts by 125 / 199.42. It is an overclock.
The RP2040 is rated for 133 MHz.

The feature stage is now not the limit. The backbone is: 74.5 ms on Mbed
against a hop of 32 ms. The sweep still measures the float32 feature stage,
because the deployed sketch does not use the q15 code yet. The 1% to 4% gap
between the profile and the sweep in finding 573 may also apply here. I did
not measure it.

**Claude comment:** `mel_dot` and `mel_log` are 59% of the Mbed stage (5.3 ms).
A fixed-point logarithm and an integer mel sum could save part of it. Each
change needs its own Level 1 and Level 2 check. The best case saves about 5 ms
of an 84 ms hop, so I would not do it before the backbone.

## How to reproduce

1. Generate the window table. From `test_harness`:

```text
python gen_q15_window.py
```

2. Copy the five q15 files to the library `CMSIS_DSP_RFFT/src`. Copy
   `rp2040-feature-profile` to `rp2040-feature-profile-q15` and rename the
   `.ino`. Replace `feature_pipeline.c`, and set `RESET_OVERLAP_PER_CLIP` to 1.
3. Build and flash. For the Mbed core, use a fixed build path (finding 570):

```text
arduino-cli compile --fqbn arduino:mbed_nano:nanorp2040connect --build-path <build-dir> <sketch-dir>
arduino-cli upload -p <upload-port> --fqbn arduino:mbed_nano:nanorp2040connect --input-dir <build-dir> <sketch-dir>
```

   For the `arduino-pico` core, select the board and the CPU speed in the
   Arduino IDE.

4. Wait about 5 seconds for the board to boot. Run from `test_harness`:

```text
python feature_profile.py --com-port <D-SUN port> --label mbed_q15 --n-clips 5 --overlap-mode reset --sketch-dir <path of rp2040-feature-profile-q15>
python logmel_compare_q15.py --label mbed_q15
```

   The profiler prints a drift warning for `feature_pipeline.c`. The warning is
   expected, because the file differs on purpose.

5. For the coverage run:

```text
python make_q15_coverage_clips.py
python feature_profile.py --com-port <D-SUN port> --label mbed_q15_coverage --overlap-mode reset --sketch-dir <path of rp2040-feature-profile-q15> --wav <coverage_gain_sweep.wav> --wav <coverage_full_scale.wav>
python logmel_compare_q15.py --label mbed_q15_coverage
```

   Do not run the profiler during a cycle sweep. Both use the board and the
   ports.

## Open questions

- **The cause of the slower `mel_log`.** The stage is 20% slower on Mbed and 8%
  slower on `arduino-pico`. A different code layout is a candidate. It is not
  tested.
- **The cause of `assemble` above the prediction.** It costs 0.070 M cycles
  against a predicted 0.03 to 0.05 M. The difference is about 0.3 ms. It is
  not investigated.
- **The error on full-scale tonal signals.** The noise-floor hypothesis is not
  tested. A host-only test is possible: compute the error per signal and per
  mel bin for `coverage_full_scale`.
- **Parity on `arduino-pico`.** The parity check and the coverage run were
  done on the Mbed build only.
- **The deployed sketch.** It still uses the float32 features. The port needs
  the new `feature_pipeline.c`, `feature_window_q15.h`, and the library files.
- **The overlap mode of the baseline.** When this finding was written, the
  `.ino` in `rp2040-feature-profile` had `RESET_OVERLAP_PER_CLIP 0`, and finding
  573 describes a default of 1. The mode does not change the timing. It changes
  the first frame of each clip after the first. The mode of the finding 573
  runs is not verified.
- **The compiler versions.** They are not recorded in the output files.

## What this finding does not show

- It does not show the AUC of the model with the q15 features on the board.
  Finding 573 shows AUC-neutral results on 40 clips with a host simulation.
  The planned confirmation is a full test with the q15 feature in the deployed
  pipeline, after the `arduino-pico` float32 run finishes.
- It does not show a measured latency of the whole hop with the q15 features
  on either core. The backbone figures in the budget table are from finding 572
  (Mbed) and finding 571 (model, `arduino-pico`).
- It does not show the effect of louder or tonal audio from the real dataset.
  The synthetic clips cover the arithmetic. They do not cover the spectral
  content of real machine sounds.