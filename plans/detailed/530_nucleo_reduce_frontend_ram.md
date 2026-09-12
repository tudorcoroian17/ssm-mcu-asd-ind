# Plan: reducing frontend RAM

The feature extraction front end holds 66.9% of static RAM on current builds,
against the backbone's 21.9%. This plan works through the reductions available,
ordered by ratio of bytes saved to risk incurred.

Every buffer role stated here is inferred from symbol names and sizes in
`footprint_symbols_stm32-nucleo-h7s3l8.csv`. Step 1 verifies them against the
source before anything is changed.

## Current state

| Symbol | Bytes | Inferred role |
| --- | ---: | --- |
| `fft_output` | 4,096 | 1,024 floats, real FFT output |
| `windowed_frame` | 4,096 | 1,024 floats, windowed input to the FFT |
| `frame_history` | 4,096 | 1,024 floats, overlap buffer across hops |
| `mel_values` | 3,988 | 997 floats, mel filterbank coefficients |
| `hop_buf` | 2,048 | 512 int16 or 512 floats, incoming hop |
| `power_spectrum` | 2,052 | 513 floats, magnitude of the rfft output |
| `hann_window` | 4,096 | 1,024 floats, window table |
| `mel_starts` / `mel_lengths` / `mel_offsets` | 386 | Filterbank index tables |
| `rfft_instance` | 24 | CMSIS FFT instance |

Roughly 16.4 KB of RAM plus 8.5 KB of flash tables. `hann_window` and the mel
tables are `const` and live in flash, so they are not RAM targets.

The configuration implied by these sizes: 1,024-point real FFT, 513-bin spectrum,
512-sample hop, 50% overlap.

## Step 1: verify the buffer roles

Nothing below is safe to attempt on inferred roles. Confirm what each buffer
holds and where it is written:

```bash
cd ~/projects/ssm-mcu-asd-ind
grep -n 'static.*\(fft_output\|windowed_frame\|frame_history\|hop_buf\|power_spectrum\)' \
  mcu/deploy/case1/*/setups/true_int8_h16_euclidean_fp32/*.c \
  /mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/nucleo-h7s3l8-ssm-mamba-asd/Appli/Core/Src/*.c
```

Then read `FeaturePipeline_ComputeLogMelFrame` end to end. The specific questions:

- Does `arm_rfft_fast_f32` write into `fft_output` with `windowed_frame` as
  input, and are they distinct arrays at the call site?
- Is `frame_history` a ring buffer or a shifted linear buffer?
- What element type does `hop_buf` hold — int16 samples from the UART, or floats?
- Is `power_spectrum` consumed by the mel projection and then dead?

Record the answers before proceeding. Each option below states which answer it
depends on.

## Option A: overlap the FFT input and output

**Saves:** 4,096 bytes. **Risk:** low. **Depends on:** `arm_rfft_fast_f32`
supporting in-place operation for this configuration.

CMSIS-DSP's `arm_rfft_fast_f32` takes separate input and output pointers.
Whether passing the same pointer is safe depends on the internal butterfly
implementation, and the CMSIS documentation does not guarantee it for the fast
real FFT.

Rather than assume, test it:

1. Capture a known frame through the current two-buffer path and record
   `fft_output` to the host.
2. Change the call to pass `windowed_frame` for both arguments.
3. Compare the output bit-for-bit against the recorded reference.

If the outputs match exactly, the saving is free. If they differ, stop — a
subtly wrong spectrum would degrade accuracy in a way that looks like a model
problem.

**Fallback if in-place fails:** the same 4,096 bytes are recoverable by using
`fft_output` as the windowing destination instead of `windowed_frame`, since the
windowed frame is dead the moment the FFT completes. That requires only that the
FFT reads its input before writing its output for each element, which is a
weaker condition than full in-place safety, but still needs the same
verification.

## Option B: overlay power_spectrum onto the FFT output

**Saves:** 2,052 bytes. **Risk:** low. **Depends on:** `power_spectrum` being
consumed only by the mel projection.

`arm_cmplx_mag_squared_f32` reads 1,024 floats of complex output and writes 513
floats of magnitude. The output is strictly smaller than the input and the
complex data is dead afterwards, so the magnitudes can be written into the first
2,052 bytes of `fft_output`.

CMSIS magnitude functions process element by element with the output index
trailing the input index, so writing into the same array at offset zero is safe.
Verify with the same reference-comparison method as Option A.

## Option C: use a union for the whole scratch chain

**Saves:** up to 6,148 bytes. **Risk:** low. **Depends on:** the buffers having
disjoint lifetimes within one call to `ComputeLogMelFrame`.

Options A and B are special cases of a general observation: `windowed_frame`,
`fft_output`, and `power_spectrum` are all scratch within a single frame
computation. None survives the function's return. A union makes that explicit
and lets the compiler and a future reader see the intent:

```c
/* Scratch for one frame of feature extraction. The three views are used in
 * sequence within FeaturePipeline_ComputeLogMelFrame and none is live across
 * calls, so they share storage.
 *
 * Order of use:
 *   1. windowed  - hann-windowed samples, written from frame_history
 *   2. spectrum  - complex FFT output, written in place over windowed
 *   3. magnitude - power spectrum, written over the head of spectrum
 */
typedef union {
    float windowed[FFT_SIZE];
    float spectrum[FFT_SIZE];
    float magnitude[FFT_SIZE / 2 + 1];
} FeatureScratch;

static FeatureScratch feature_scratch;
```

This is preferable to Options A and B applied separately, because the aliasing
becomes documented rather than implicit. But it still rests on the same in-place
FFT question, so run the Option A test first.

**Caution:** type-punning through a union is well-defined in C for reading the
member last written, and all three members here are float arrays, so no strict
aliasing issue arises. It would be different if the members had different element
types.

## Option D: reduce frame_history to the overlap

**Saves:** 2,048 bytes. **Risk:** medium. **Depends on:** `frame_history` being a
shifted linear buffer rather than a ring.

With a 1,024-point FFT and a 512-sample hop, only 512 samples carry over between
frames. A 1,024-float history is holding a full frame when half of it is
reconstructible from the incoming hop.

The restructure: keep a 512-float overlap buffer, and assemble each frame by
writing the overlap into the first half of the scratch buffer and the new hop
into the second half. The assembly target is the scratch from Option C, so this
costs no additional memory.

This is medium risk because it changes the frame assembly logic rather than just
where bytes live. An off-by-one in the overlap boundary produces a spectrum that
is subtly wrong rather than obviously broken. Validate by comparing computed
log-mel frames against the offline Python pipeline for a known clip, not by
listening to the output.

## Option E: narrow hop_buf to the wire format

**Saves:** up to 1,024 bytes. **Risk:** low. **Depends on:** what the host
actually sends.

If `hop_buf` is 512 floats but the host transmits int16 PCM, the buffer is twice
the width it needs. Receive int16 and convert during the frame assembly in
Option D, which touches every sample anyway.

If the host already sends floats, this option does not apply — and it is worth
asking whether it should, since halving the wire format also halves the transfer
time per hop.

## Option F: reconsider the FFT size

**Saves:** scales everything above by half. **Risk:** high — this changes the
features.

A 512-point FFT would halve every buffer in the chain. But it also changes the
frequency resolution of the mel filterbank, which changes the features, which
changes the model. This is not a memory optimization; it is a different
experiment.

Do not treat it as a footprint fix. If you want it, it belongs in the offline
pipeline as a configuration variant, retrained and evaluated end to end, with its
accuracy reported alongside the memory saving.

## Recommended sequence

| Order | Option | Cumulative saving | Gate |
| --- | --- | ---: | --- |
| 1 | Verify roles | 0 | Source review complete |
| 2 | A — in-place FFT test | 0 | Bit-exact match against reference |
| 3 | C — scratch union (subsumes A and B) | 6,148 | Log-mel frames match offline pipeline |
| 4 | D — overlap-only history | 8,196 | Same validation, plus boundary check |
| 5 | E — narrow hop_buf | 9,220 | Host wire format confirmed |

That takes the frontend from 16,412 bytes to roughly 7,200 — from 66.9% of RAM
to about 32%, making the backbone the largest RAM consumer, which is the shape a
reader of the paper would expect.

## Validation protocol

Memory changes that alter feature values are the dangerous kind, because the
failure mode is degraded accuracy rather than a crash. For each step:

1. **Before changing anything**, capture log-mel frames for a fixed test clip
   through the current firmware and save them to the host.
2. Apply one option. Never two at once.
3. Recompute frames for the same clip and compare against the reference. For
   Options A, B, C, and E the result should be bit-identical. For Option D, a
   difference means the overlap boundary is wrong.
4. Re-run the sweep for one setup and confirm the RAM reduction appears in
   `footprint_highlevel`.
5. Only then move to the next option.

Bit-exactness is the right standard for everything except Option D, and even
there the frames should match to within float rounding, not merely look similar.

## What this does not address

**Stack.** The per-timestep SSM activations live on the stack and appear in no
symbol, so none of the above touches peak RAM in the way it touches static RAM.
Collect stack data before and after, so the reported reduction is honest about
which figure moved.

**The mel filterbank in flash.** `mel_values` at 3,988 bytes is `const` and
therefore flash, not RAM. A sparse representation would shrink it — the
`mel_starts`, `mel_lengths`, and `mel_offsets` tables suggest one is already
partly in use — but that is a flash optimization and belongs in a different
plan.

**Claude comment:** the honest framing for the paper is that this was found, not
designed. A first working implementation allocating a separate buffer per
pipeline stage is normal and correct, and the measurement is what revealed the
front end outweighing the model three to one. Presenting the before-and-after,
with the tooling that surfaced it, is a more useful contribution than presenting
only the optimized figure — it shows the measurement method earning its keep.