## Results

Steps 1 and 2 landed. Step 3 was implemented, then reverted after it was shown
to cause a silent device reset. Options 4′ and 5′ were not attempted.

Shipping configuration: `-Os`, steps 1 and 2, single-buffered hop receive
reverted to double-buffered.

### Footprint

| State | Frontend | Backbone | Frontend share | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 16,412 | 5,380 | 66.8% | 3.05× |
| After step 1 | 14,360 | 5,380 | — | 2.67× |
| After step 2 | 11,288 | 5,380 | 58.1% | 2.10× |

Backbone static RAM is 5,380 bytes for the selective variant and 4,996 bytes
for the classic variant. The selective variant is deployed, so it sets the
target.

The frontend figures after steps 1 and 2 are predicted from the symbol sizes
removed, not read from `footprint_symbols_stm32-nucleo-h7s3l8.csv`. Regenerate
the footprint report against the `-Os` build and replace this table with
measured values before citing it.

The goal of frontend-backbone parity was not reached. Reaching it requires
options 4′ or 5′, described under "Parked options" below.

### Step 1: verified buffer roles

The options in the original plan rested on inferred buffer roles. Reading
`feature_pipeline.c`, `audio_ingest.c`, and `main.c` established the following.

| Question | Finding |
| --- | --- |
| Does the RFFT use distinct input and output arrays? | Yes — `arm_rfft_fast_f32(&rfft_instance, windowed_frame, fft_output, 0)` |
| Is `frame_history` a ring or a shifted linear buffer? | Shifted linear: `memmove` down by one hop, then `memcpy` the new hop into the tail |
| What does `hop_buf` hold? | `static int16_t hop_buf[2][512]` — already int16, already double-buffered |
| Is `power_spectrum` dead after the mel projection? | Yes |

Three corrections to the original options follow.

**Option E does not apply.** The host already sends int16 PCM and `hop_buf` is
already int16. Its 2,048 bytes are `2 × 512 × 2`, not `512 × 4`. The predicted
1 KB saving does not exist there. An equivalent 1 KB was recovered instead by
keeping the retained overlap in int16 (step 2).

**The symbol table understated frontend RAM by 2,048 bytes.** `main.c` declared
`float32_t hop_f32[512]` inside the hop loop — a stack array, invisible to a
symbol-based footprint report. Step 2 removed it.

**Option B is safer aimed at the FFT input than at the FFT output.** Overlaying
the power spectrum onto `fft_output` requires saving the DC and Nyquist slots
before the magnitude pass, because `fft_output[512]` is still live input at the
moment `power_spectrum[512]` would be written. Overlaying onto `windowed_frame`
has no such hazard: the RFFT consumes and overwrites its input buffer, so those
1,024 floats are provably dead, and the two arrays remain distinct objects at
every CMSIS call site.

### Step 1: power spectrum over the windowed frame

Removed the standalone `power_spectrum[513]` array and pointed a
`float32_t *const` at `windowed_frame`. Saves 2,052 bytes. No CMSIS call
receives aliased arguments.

### Step 2: int16 overlap with fused scaling

Replaced `frame_history[1024]` (float) with `overlap[512]` (int16), and fused
the `/32768.0f` scaling into the Hann multiply. Split the pipeline API into
`FeaturePipeline_BeginFrame` and `FeaturePipeline_FinishFrame`. Saves 3,072
bytes of static RAM plus the 2,048-byte stack array.

At 50% overlap the first half of frame N is exactly the hop received for frame
N-1, so a full-frame float history stored 4,096 bytes to hold 1,024 bytes of
information.

The fusion is bit-exact. Dividing by 32,768 is exact power-of-two scaling, so
`(s / 32768.0f) * hann[i]` performs the same two operations in the same order
as the old three-pass path, with the window multiply as the only rounding.

### Step 3: single hop buffer — implemented, then reverted

Collapsing `hop_buf[2][512]` to `hop_buf[512]` saves a further 1,024 bytes and
is sound on paper: `FeaturePipeline_BeginFrame` consumes the hop entirely, so
the buffer is dead when it returns and the next DMA can reuse it.

It causes a silent device reset. Evidence:

| Build | Step 3 present | Result |
| --- | --- | --- |
| `-O2` | Yes | Reset at clips 5, 7, 11, 25 |
| `-O2` | No | Reset at clip 5 |
| `-Os` | Yes | Reset at clip 5 |
| `-Os` | No | 528 clips clean |

At `-O2` the reversion appeared to change nothing, which briefly suggested step
3 was innocent. It was not — a second, independent trigger was masking it. At
`-Os`, step 3 alone reproduces the fault.

The likely mechanism is the widened window during which no receive DMA is
armed. Step 3 moved the re-arm from immediately after `WaitHopComplete` to
after frame assembly, roughly 30 µs later. Against an 86.8 µs byte time at
115200 baud plus the USART3 receive FIFO, the margin should be adequate — but
ORE overrun latching is a recorded failure mode on this hardware, and a fault
firing once per few thousand hops is what an insufficient margin looks like.
This was not confirmed; `RCC->RSR` was never captured.

Claude comment: I'd leave step 3 parked rather than fixed. It buys 1,024 bytes
and moves the ratio from 2.10× to 1.91× — a difference no reader will notice —
against a regression in the data path. If someone does revisit it, capture
`RCC->RSR` first; the reset cause decides whether this is a UART race at all.

### Falsified: in-place RFFT

**Hypothesis.** `arm_rfft_fast_f32` tolerates `pSrc == pDst`, allowing
`windowed_frame` and `fft_output` to merge and saving 4,096 bytes.

**Result.** It does not. Measured on-device with a 1024-point transform over
deterministic broadband noise:

```
INPLACE: mismatches=510/1024 max_abs_diff=6.579071e+01 max_abs_ref=4.746871e+01
```

The maximum deviation exceeds the maximum reference magnitude, so this is not a
rounding difference.

**Mechanism.** `arm_rfft_fast_f32` runs `arm_cfft_f32` in place on the source
buffer, then calls `stage_rfft_f32`, which advances one read pointer forward
from index 2 while a second read pointer walks backward from the end, with the
write pointer advancing forward from index 0. Past the midpoint the backward
reader reads slots the writer has already overwritten. For a 1024-point
transform this predicts 512 correct and 512 corrupt outputs; 514 and 510 were
measured.

The corruption is structural. No CMSIS version, compiler flag, or alignment
change will fix it. Do not retry this.

### Parked options

Neither was attempted. Both remain viable if frontend-backbone parity becomes
a requirement.

**Option 4′ — custom recombination stage.** Call `arm_cfft_f32` directly, which
*is* documented as in place, and write a replacement for `stage_rfft_f32` that
computes `|X[k]|²` directly instead of materialising 1,024 floats of complex
spectrum. Estimated saving 2,044 bytes, reaching 8,220.

**Option 5′ — fuse the mel projection into the recombination.** Scatter each
bin into the 64 mel accumulators as it is computed, so no power spectrum array
exists. Estimated saving a further 1,796 bytes, reaching 6,424 — a ratio of
1.19×. Requires a transposed filterbank export.

**Option F — q15 front end.** Halves both FFT buffers outright. Changes the
features and needs an AUC re-check, so it is an experiment rather than a
footprint fix — but it is on-topic for a dissertation about quantized
deployment rather than off it.

Claude comment: before spending effort on 4′ and 5′, consider whether parity is
the right target. A 1024-point float STFT costs roughly 4 KB of scratch however
the surrounding code is arranged, and at `d_model=64` across two layers the
backbone is small. "The feature front end is comparable in RAM to the SSM
itself at this model scale" is a real result about SSMs on microcontrollers.
Driving the front end below the backbone by aliasing tricks optimizes the
narrative rather than the engineering.

## Toolchain findings

Three toolchain issues surfaced during this work. All three affect previously
recorded measurements.

### The launch configuration flashed a different build than it compiled

`Appli/nucleo-h7s3l8-ssm-mamba-asd_Appli.launch` hardcoded
`Debug/nucleo-h7s3l8-ssm-mamba-asd_Appli.elf` in both `PROGRAM_NAME` and the
Startup tab load list, while `PROJECT_BUILD_CONFIG_AUTO_ATTR` was `false`. With
no configuration pinned, the launch built the active configuration — Release —
and then flashed a hardcoded path into `Debug/`, which had not been rebuilt for
days.

Every flash therefore wrote a stale image and verified it successfully.
Verification passing means the transfer was clean, not that the current code is
on the board.

To avoid a repeat:

- Set **Build Configuration** to **Use Active** on the Main tab.
- Update the path on **both** the Main tab and the Startup tab load list. The
  Main tab controls which symbols the debugger loads; the Startup entry
  controls which bytes reach flash.
- Keep a `BUILD: __DATE__ __TIME__` line in the startup banner. Note that
  `__TIME__` reflects the compile time of `main.c` only, so it catches a stale
  flash rather than a stale object file.

### All device measurements before 2026-09-13 are `-O0` builds

Debug builds with `-g3 -O0`; Release built with `-O2` and no `-g`. Because of
the launch configuration bug above, every device measurement recorded before
the fix came from an `-O0` binary.

Measured on identical source for `normalize`, `backbone`, and `head`:

| Counter | `-O0` | `-O2` | Speedup |
| --- | ---: | ---: | ---: |
| `feature` | 176,310,368 | 107,202,123 | 1.64× |
| `normalize` | 12,993,984 | 6,399,033 | 2.03× |
| `backbone` | 3,297,079,526 | 1,340,372,406 | 2.46× |
| `head` | 4,339 | 1,833 | 2.37× |

Consequences:

- Any recorded cycle count, millisecond figure, or real-time headroom claim
  from before this date understates the hardware by roughly 2.4×. Audit
  `findings/` before the write-up.
- Static RAM figures are unaffected — `.bss` array sizes do not move with
  optimization level.
- Flash figures and percentages are affected. Regenerate any that came from a
  Debug map.
- Timings must be re-measured under `-Os`, the shipping configuration. No `-Os`
  timing data exists yet.

Release now carries `-g3`. Debug information does not change code generation
and the DWARF sections are not loadable, so this costs nothing on the device
and makes Release debuggable.

### `-O2` and `-Ofast` cause a silent device reset

Independent of step 3.

| Flags | Result |
| --- | --- |
| `-O0` | Clean, sustained |
| `-Os` | 528 clips clean |
| `-O2` | Reset at clips 5, 7, 11, 25 |
| `-Ofast` | Reset at clip 9 |

**Diagnosis.** The device resets rather than hangs. Under a debug session it
halts spontaneously with `pc = 0x70000a90` (byte zero of `main`) and
`lr = 0x70001827` (`LoopForever` plus the Thumb bit — the return address pushed
by `bl main` in `Reset_Handler`). The green LED stays solid and neither
`Error_Handler` nor `HardFault_Handler` is reached. With the debugger attached
the core halts at the `main` breakpoint and goes silent, which is why the host
observes zero bytes and why reopening the serial port does not recover it.

**Not yet determined.** `RCC->RSR` was never captured, so the reset cause —
brown-out, lockup escalation, or watchdog — is unknown. Capture it first if
this is revisited:

```c
/* In USER CODE BEGIN 1 */
volatile uint32_t reset_flags = RCC->RSR;
__HAL_RCC_CLEAR_RESET_FLAGS();

/* After BSP_COM_Init succeeds */
printf("RESET: RSR=0x%08lX\r\n", (unsigned long) reset_flags);
```

**Untested probes.** External power rather than an ST-Link USB port, which
tests the brown-out hypothesis. `-O2 -fno-tree-loop-vectorize
-fno-tree-slp-vectorize`, since GCC 13 enables vectorization at `-O2` but not
at `-Os`, and merged `LDRD`/`STRD` accesses fault unconditionally on misaligned
addresses.

`-Ofast` is unusable regardless of this fault: it implies `-ffast-math`, which
permits reassociation and flush-to-zero, invalidating bit-exactness checks
against the Python reference.

**Status.** Open, non-blocking. `-Os` is the shipping configuration.

### Log-mel drift against the `-O0` baseline is explained

Comparing step 2 output against a saved `-O0` capture showed a maximum absolute
delta of 2.4×10⁻⁷ to 5.1×10⁻⁷ and a mean of 7×10⁻⁸ to 1.4×10⁻⁷ — one to two
ULP, spread across most bins.

This is the `-O0` to `-O2` transition, not a regression in steps 1 or 2.
`arm_cmplx_mag_squared_f32` computes `real*real + imag*imag`. GCC defaults to
`-ffp-contract=fast`, which fuses that into a single VFMA at `-O2` — one
rounding instead of two — and does not contract at `-O0`. That produces exactly
a pervasive one-ULP difference in every power spectrum bin.

Steps 1 and 2 are bit-exact. The drift entered through the build configuration
changing underneath the comparison.

Claude comment: this cost several hours to chase. The lesson worth keeping is
procedural rather than numerical — when a bit-exactness check fails, pin the
baseline's provenance before theorizing about the code. Which binary, which
configuration, which date.

## Open issues

1. Regenerate `footprint_symbols_stm32-nucleo-h7s3l8.csv` against the `-Os`
   build and replace the predicted frontend figures above with measured ones.
2. Re-measure all timings under `-Os`. No valid deployment latency data
   currently exists.
3. Audit `findings/` for `-O0` timing figures and mark or regenerate them.
4. `-O2` silent reset: capture `RCC->RSR` before further investigation.
5. Step 3 remains available if the 1,024 bytes are ever needed, contingent on
   issue 4.