# Parity failures concentrate in onset/tail silence-bracket frames

**Feeds:** `plans/highlevel/04_phase_3_mcu_feature_pipeline.md` §3.4 (parity acceptance test).

## Finding

Across a 5-clip batch (case2/case3, normal and anomalous), every failing or
near-failing (frame, bin) pair falls in frames 0-~24 or ~330-343 -- exactly
the onset (~0.8s) and tail (~0.45s) near-silence brackets `findings/115`
documented for IND clips (0.8s / 0.032s per frame = frame 25, an exact match
to where energy visibly jumps in this batch's worst clip). No mid-clip frame
has ever failed in any clip tested.

## Mechanism, directly verified

Checked the actual int16 samples sent to the MCU for the worst-failing clip
(`1252010002_ToyCar_case2_ab52_IND_ch1_0002.wav`). Frame 0's real audio:
range -3 to +3 (int16), RMS 1.46 LSB -- against 791 LSB mid-clip (543x
quieter). In normalized units, that RMS (4.4e-5) is smaller than one
quantization step (3.1e-5): at this level, int16 rounding error is
comparable to the signal itself, not a small perturbation of it.

This is a real, additional error source specific to the MCU path: the
Python reference (`extract_logmel`) never quantizes to int16 -- it stays in
float32/float64 from `librosa.load` through the STFT. The MCU path's int16
transport (matches real ADC/mic output, halves transfer size, fits RAM)
introduces this quantization once, and it only matters where the signal is
already this quiet.

`log(x + 1e-6)`'s slope is steepest exactly where x is tiny, so a linear-domain
quantization error that would be negligible at mid-clip energy levels becomes
a large log-domain gap specifically in these frames.

## Not a pipeline bug

Windowing, FFT, mel matmul, and log are all doing the correct computation.
The amplification is inherent to evaluating log-energy at real,
by-design near-silence, combined with a transport choice (int16) made
deliberately to match real microphone/ADC behavior. It is not fixable by
changing the DSP chain, and changing the log epsilon or transport format
would diverge from the trained model's own preprocessing or from realistic
deployment, respectively.

## Decision: acceptance criterion unchanged

Keeping the full 344-frame, all-bins tolerance check as-is, rather than
excluding onset/tail frames. Rationale: minimizing the number of things that
differ between the host-computed metrics (AUC, etc.) and the device parity
check matters more than tightening the pass rate -- boundary frames'
elevated error is real (a genuine consequence of int16 transport meeting
near-silence) and should count, not be defined away. A clip that fails
because of clean, understood boundary-frame amplification is not being
treated differently from a clip that fails for any other reason.

## Results, this batch

Tolerance: 0.2627 (`findings/250`). 344 frames x 64 bins = 22016 points per clip.

| clip | case | label | max abs diff | mean abs diff | signed mean | failing points | result |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `1200010799_..._0799` | 2 | normal | 0.061050 | 0.000775 | +0.000222 | 0 / 22016 | PASS |
| `1200011115_..._1115` | 2 | normal | 0.288310 | 0.000816 | +0.000250 | 1 / 22016 | FAIL |
| `1252010002_..._0002` | 2 | ab52 (anomaly) | 0.562432 | 0.001087 | +0.000405 | 2 / 22016 | FAIL |
| `1300010486_..._0486` | 3 | normal | 0.212889 | 0.001306 | +0.000417 | 0 / 22016 | PASS |
| `1300010491_..._0491` | 3 | normal | 0.195223 | 0.001284 | +0.000421 | 0 / 22016 | PASS |

3 of 5 PASS. Both failures are single- or double-point overshoots, both
localized to frame 0, both explained by the mechanism above -- not evidence
of a broader or growing problem. See `findings/265` for the case1 clip
tested earlier (PASS, max 0.0617), not repeated in this table.

## Open item -- not decided here

Whether this matters for the actual anomaly-detection task at all, given the
model's `pooling: mean` (`configs/default.yaml`) likely dilutes the influence
of ~39 boundary frames out of ~344 regardless of their individual accuracy --
untested here.