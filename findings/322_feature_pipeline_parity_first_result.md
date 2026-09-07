# Feature pipeline parity: first result, single synthetic frame

**Feeds:** `plans/highlevel/04_phase_3_mcu_feature_pipeline.md` §3.4 (parity acceptance test).

Ran the full on-device feature pipeline (window -> CMSIS-DSP real FFT -> power ->
sparse mel matmul -> log) on the Nucleo-H7S3L8 against a synthetic two-tone test
frame (440 Hz + 2000 Hz, 16 kHz, 1024 samples), compared per-bin against a Python
float64 reference computed with the same window/FFT/mel/log chain as
`src/features/logmel.py` (see `parity_test_vector.h`).

**Result:** max absolute per-bin difference = 0.000078, at mel bin 14, against
the `findings/250` tolerance of 0.2627 -- roughly 3,400x inside budget. This
magnitude matches expected float32-hardware-vs-float64-reference rounding, not
a coincidence: it is far too small to be AUC-relevant noise and far too
systematic (single deterministic frame) to be a fluke pass.

**Scope of this result:** validates §3.2 (CMSIS-DSP chain) and §3.3 (log,
H7S3L8 path only, hardware FPU) for one already-extracted, unpadded 1024-sample
frame. Does not yet cover: multiple/varied test frames, real dataset audio, the
other two boards' log implementations (esp-dsp for ESP32, LUT for RP2040,
neither built yet), or continuous-stream framing/hop/centering (`center=True`
reflect-padding at stream boundaries remains unimplemented and untested).