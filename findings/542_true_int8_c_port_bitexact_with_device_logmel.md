# True-int8 C port matches the device bit-for-bit once fed the device's own log-mel

**Feeds:** `findings/541_true_int8_survives_primary_metrics.md`, whose
"Open items" listed *no C implementation yet* as the remaining gap before
this scheme could be trusted on hardware -- this finding closes that gap
for `h=int8`. `plans/detailed/07_phase5_two_head_deployment.md` steps 4-6
(backbone port, weight export). `findings/322`-`325` (the feature-pipeline
parity work and debug-capture mechanism this reused directly, unmodified).

## Result

`mcu/ssm_true_int8_src/ssm_backbone.c`, running on the Nucleo-H7S3L8 at
`h=int8`, produces a pooled embedding that is **bit-for-bit identical** to
`checks/smoke/quant/true_int8_sim.py`'s `quantized_forward` Python
reference, once both are given the exact same input log-mel. Not "within
tolerance" -- `max_abs_err = 0.0`, `mean_abs_err = 0.0`.

The first parity attempt, using the same cached (librosa-derived) log-mel
every other scheme's parity check has always used, showed
`max_abs_err = 1.71e-1`. That number was never a logic bug in the C port.
It was entirely attributable to feeding Python a different -- if small and
already-accepted -- log-mel than the one the device's own CMSIS-DSP feature
pipeline actually produced for this clip.

## Method: elimination, not guessing

Three hypotheses were tested in order, each one a cheap, decisive,
no-reflash-required check against the same on-device capture, before
touching hardware again:

1. **Rounding convention** (C's `roundf`, round-half-away-from-zero, vs
   NumPy's `np.round`, round-half-to-even). Recomputed the Python reference
   with round-half-away-from-zero substituted throughout. Result: bit-
   identical to the original reference -- zero `.5` ties occurred anywhere
   in this clip. Ruled out completely, not just discounted.
2. **Float32 (C) vs float64 (Python) precision** in the rescale steps.
   Re-implemented `quantized_block_step`/`quantized_forward`'s exact logic
   with every real-valued operation forced to `float32`, matching the C
   port's precision rather than just its logic. Result: bit-identical to
   the float64 reference. Ruled out completely.
3. **Input mismatch.** The Python reference was loading the *cached*
   librosa log-mel (`cache_path`); the device never sees that -- it
   computes log-mel itself, on-device, via CMSIS-DSP. Findings 322-325 had
   already quantified a real, accepted discrepancy there for the fp32
   backbone (max abs diff `0.0617` against a derived tolerance of `0.2627`)
   and built a debug-capture path to record the device's own log-mel per
   clip. That capture already existed on disk from earlier phase work --
   no reflash needed. Feeding it into the same, unmodified
   `quantized_forward` in place of the cached file: **exact match.**

| log-mel diff, cached vs device (this clip) | max_abs | mean_abs |
| :--- | ---: | ---: |
| this check | 5.34e-2 | 7.97e-4 |
| Phase 3's original figure (`findings/324`) | 6.17e-2 | -- |

| embedding parity, reference input source | max_abs_err | mean_abs_err |
| :--- | ---: | ---: |
| cached (librosa) log-mel | 1.71e-1 | 4.61e-2 |
| device's own (CMSIS-DSP) log-mel | **0.0** | **0.0** |

## The precisely attributable finding

Holding the C port, the Python logic, and every weight/scale fixed, and
changing only which log-mel the reference computation was fed, took the
error from `1.71e-1` to exactly `0.0`. That isolates the cause to the input
alone -- not a struct field, not a scale, not a LUT boundary, not an
accumulator width. The magnitude of the underlying log-mel discrepancy
(`5.34e-2`) is itself unremarkable and well inside the tolerance every fp32
and fake-quant scheme in this project has always absorbed without
comment. What's new here is that true-int8's arithmetic isn't smooth the
way theirs is: it makes dozens of hard rounding decisions per frame, and
those decisions feed a recurrent `h` that remembers them across the whole
clip. A small, ordinary input wobble that fp32 just absorbs continuously
can, in this scheme, occasionally land a value on the other side of a
rounding boundary -- and because `h` carries state forward, that flip
doesn't average out, it persists.

## Consequence

**True-int8 parity checks must be run against the device's own captured
log-mel, never the cached librosa file** -- this is a real, permanent
difference in this scheme's verification procedure, not a one-off
debugging fact to forget. Every other scheme in this project could use the
cached file interchangeably with the device's own capture because their
arithmetic doesn't amplify the difference; this one can't.

The `h=int8` C port's logic is now validated with the strongest form this
kind of check can give -- exact equality, not a tolerance pass. `h=int16`
still needs its own confirmation run, but the same device log-mel capture
can be reused directly: the feature pipeline (`feature_pipeline.c`) doesn't
depend on which backbone scheme is linked, only the SSM backbone and its
`h` storage do.

## Open items

- **`h=int16` not yet confirmed by this method** -- only `h=int8` has a
  bit-exact result. Same procedure, same captured log-mel, next.
- **Single clip, single fold (case 1)**, as always in this project.
- This methodology point (device log-mel required, not the cache file)
  needs to be folded into whatever instructions eventually cover re-running
  or extending this scheme's parity checks, so it isn't silently forgotten
  and rediscovered as a fresh "bug" later.

<!-- Claude comment: the thing worth calling out about this debugging
sequence for the dissertation write-up isn't just the conclusion, it's the
shape of the elimination: two hypotheses (rounding, precision) were each
tested in a way that could have come back either "confirmed" or "ruled
out" with equal clarity, and both came back cleanly ruled out (exactly
zero difference, not "small difference") before the actual cause was
found. That's a stronger methodological story than "I guessed the input
mismatch and I was right" -- it shows the search was systematic and the
negative results were as informative as the positive one. Bit-exact
equality as the target, rather than a tolerance band, is what made the
eliminations trustworthy in the first place: a tolerance-band pass
couldn't have distinguished "rounding convention doesn't matter" from
"rounding convention matters a little, coincidentally under the
tolerance." -->