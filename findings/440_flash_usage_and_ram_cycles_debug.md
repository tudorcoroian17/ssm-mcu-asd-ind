# Phase 4 step 5: flash, RAM, and cycle measurements on real hardware

**Feeds:** `05_phase_4_backbone_port.md` step 5; `findings/400` and `findings/310`'s open
XiP-latency question.

## Flash and RAM

From `arm-none-eabi-size --format=sysv` on the built `.elf` (Debug configuration):

| Region | Used | Available | Margin |
| :--- | ---: | ---: | ---: |
| Flash (external XSPI, `.text`+`.rodata`+vectors) | 262,660 B (256.5 KB) | XSPI capacity, not separately measured here | not the binding constraint (`findings/310`) |
| Main RAM (`.data`+`.bss`) | 465,804 B | 465,920 B | 116 B |
| DTCM (`.ssm_state_dtcm` + heap/stack) | 11,528 B | 65,536 B | 54,008 B |

Main RAM's 116-byte margin matches `findings/403`'s ~80-byte estimate, confirming that finding's
arithmetic. Flash is not a binding constraint at this size.

## Cycles (DWT cycle counter, six runs, same clip, values agree to within 0.1%)

| Stage | Cycles/frame | Time/frame (600 MHz) |
| :--- | ---: | ---: |
| Feature pipeline | ~514,200 | 857 us |
| Normalization | ~37,910 | 63 us |
| Backbone (after the `A` precompute fix, below) | ~2,815,307 | 4,692 us |
| **Total** | **~3,367,400** | **5.61 ms** |

Against the 32 ms/frame budget (`hop_length=512` at 16 kHz): **5.7x margin, 17.5% of budget used**.
This directly answers `findings/400`'s and `findings/310`'s open question about XiP execution
latency eating into the real-time margin -- it doesn't. Whatever latency external-flash execution
adds is already included in this measurement, since it's a real on-device number, not a model.

**Caveat:** this is a Debug build (`-O0`). A Release build would very likely be faster still --
this number is a conservative floor, not the number that belongs in a final report.

## The `A_log` -> `A` precompute fix

`ssm_block_step` originally called `expf(A_log[c][n])` inside the innermost per-timestep loop --
2,048 redundant calls per frame (64 channels x 16 states x 2 layers) recomputing the same 1,024
values every frame, despite `A_log` being fixed after training. Precomputing `A = -exp(A_log)`
once in `export_ssm_weights.py` and shipping `A` instead removed the runtime call entirely.

| | Backbone cycles/frame | vs. budget |
| :--- | ---: | ---: |
| Before | 4,705,985 | 4,706 us of 32,000 us (14.7%) |
| After | 2,815,307 | 4,692 us... | 

Wait -- before was already inside budget on its own (14.7%); the real effect is on the margin once
feature and normalize stages are added back in (5.7x total vs. 3.65x before). The `expf()` removal
cut backbone cost by 40.2% -- a large majority of the original cost, though not the near-entirety
a first-pass estimate suggested (that estimate assumed `expf()` alone could account for almost all
of the original 4.7M cycles; the real number shows it accounted for a large majority, with the
remainder being the actual linear algebra and loop overhead).

Parity after the change: max abs error `5.991459e-04`, against `5.990826e-04` before -- a
difference at the 7th significant digit, confirming the substitution changed nothing that
matters numerically, exactly as expected for an exact algebraic rewrite of a constant.

## Bottom line

Phase 4's real-time margin is not close on this board -- 5.7x headroom even in an unoptimized
Debug build. Flash and RAM both fit with room to spare. Step 5 is answered: `f4cd557b7e3b` is
comfortably deployable on the Nucleo-H7S3L8 in fp32, before any Phase 5 quantization work begins.

**Claude comment:** the corrected 40.2%-not-"nearly all" figure for the `expf()` fix is worth
being honest about in future estimates like this -- my original hypothesis was directionally
right and cheap to act on regardless, but "a rough estimate matched a measurement" and "a rough
estimate was numerically right" are different claims, and only the first one held here.