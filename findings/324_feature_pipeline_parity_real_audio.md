# Feature pipeline parity on real audio: Nucleo-H7S3L8 PASS

**Feeds:** `plans/highlevel/04_phase_3_mcu_feature_pipeline.md` §3.4 (parity acceptance test).

## Result

Full on-device feature pipeline (periodic Hann -> CMSIS-DSP `arm_rfft_fast_f32`
-> power -> sparse mel matmul -> natural log) run on a real IND clip
(`1100010001_ToyCar_case1_normal_IND_ch1_0001.wav`, 344 frames x 64 mels),
compared per-bin against `src/features/logmel.py`.

| metric | value |
| :--- | ---: |
| max abs diff | 0.061728 |
| tolerance (`findings/250`) | 0.2627 |
| failing points | 0 of 22016 |
| mean abs diff | 0.001001 |
| signed mean diff | +0.000326 |

Residual error concentrates in mel bins 0-5 at values near the `log(1e-6)`
floor, consistent with int16 transport quantization and float32-vs-float64
arithmetic amplified by the log at low energies. Signed mean near zero
confirms symmetric noise rather than a systematic offset.

## Methodological requirement: identical input samples

The first real-audio run FAILED (max diff 1.705, 1.64% of points), with errors
confined to mel bins 0, 2, 3, 62, 63 -- the two edges of the filterbank, with
opposite signs (MCU high near Nyquist, low at the bottom). Cause: the source
clips are natively **48 kHz**, and `librosa.load(sr=16000)` resamples them.
The sender was independently loading the same file with
`soundfile` + `scipy.signal.resample_poly` -- a different anti-aliasing filter,
so the two paths were comparing different audio.

**Requirement:** the samples streamed to the board must come from the same
`librosa.load(wav_path, sr=SR)` call the reference uses. `mcu/prepare_clip.py`
does this and saves int16 samples to `.npy`; the sender streams that file and
performs no audio loading of its own. This applies to all three boards.

**Claude comment:** worth internalizing the general shape of this. A parity
harness that loads its own copy of the input measures the loader and the
pipeline together, and the loader difference is invisible until it shows up
as a pipeline failure at the band edges. Feed the device the reference's own
samples, not the reference's own file.

Also note `np.round` before `.astype('<i2')` -- `.astype` truncates toward
zero, biasing every sample by up to 1 LSB.

## Scope

Covers: §3.2 CMSIS-DSP chain and §3.3 log on the H7S3L8 (hardware FPU), one
clip, streaming framing with `center=True`-equivalent zero padding at both ends.

Does not cover: multiple clips or cases, anomalous clips, the ESP32 (esp-dsp)
or RP2040 (log LUT, no FPU) implementations, or §3.5/§3.6.