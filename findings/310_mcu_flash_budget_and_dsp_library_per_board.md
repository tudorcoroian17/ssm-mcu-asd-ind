# Flash budget and DSP library, per board

**Feeds:** `plans/highlevel/04_phase_3_mcu_feature_pipeline.md` §3.1 (filterbank format decision) and §3.2 (CMSIS-DSP chain).

## Filterbank storage, exact (from `configs/mel_filterbank.npy`, n_mels=64, n_fft=1024)

- Dense float32: 131,328 bytes (128.25 KiB).
- Matrix is 3.04% dense (997 of 32,832 entries nonzero).
- Every mel filter's nonzero entries form one contiguous run (5–46 bins, mean 15.6) — the
  standard triangular-filter shape.
- Contiguous-run sparse storage (per row: start bin, length, float32 values): 4,244 bytes
  (4.14 KiB) — a 31x reduction over dense.

## Per-board flash budget

| Board | Flash | Dense filterbank | Sparse filterbank |
|---|---|---|---|
| Arduino Nano RP2040 Connect | 2 MB | 6.3% | 0.2% |
| ESP32-WROOM-32 | 4 MB (external SPI, memory-mapped/XIP) | 3.1% | <0.1% |
| Nucleo-H7S3L8 | **64 KB internal user flash** (STM32H7S3L8H6, ST's bootflash H7S line; 128 KB system flash is not usable for application code) | 200% — does not fit | 6.5% |

**Correction to `00_master_file.md` §4:** "generous flash/RAM" for the H7S3L8 is only true of RAM
(up to 620 KB). Flash is the smallest of the three boards by far. The Nucleo board carries an
external XSPI NOR flash for exactly this reason; decide explicitly whether the feature pipeline
(and later, the ported backbone) targets internal flash (use the sparse filterbank, budget
tightly) or external flash (internal-flash size stops being the binding constraint, but confirm
execute-in-place latency doesn't eat into the real-time margin in §3.6).

**Decision for the correction:** use the external flash and document latency issues generated because of that.

**Decision for §3.1:** use the contiguous-run sparse format on all three boards. It's cheap
everywhere and load-bearing on the H7S3L8 specifically.

## DSP library per board (§3.2)

CMSIS-DSP is ARM Cortex-M only — covers the H7S3L8 and RP2040, not the ESP32 (Xtensa LX6).
ESP32 needs Espressif's **esp-dsp** library instead (`dsps_fft2r_fc32` or similar for the real
FFT). Same pipeline shape (window multiply → FFT → magnitude → mel matmul → log), two different
APIs to implement against.