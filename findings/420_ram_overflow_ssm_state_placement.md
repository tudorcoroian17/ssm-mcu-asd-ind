# RAM overflow on Phase 4's ssm_state, and where it ended up

**Feeds:** `05_phase_4_backbone_port.md` step 3; `/areas/mcu-firmware-setup.md`.

## Result

Adding `SSMBackbone_State` to `main.c` overflowed the main RAM region by 9,908 bytes at link
time. The region was already at roughly 99.98% utilization before Phase 4 touched anything --
this was not a Phase 4 sizing mistake, it surfaced a pre-existing near-zero margin.

## The numbers

The active linker script (`STM32H7S3L8HX_ROMxspi2.ld`) gives the main RAM region as
`__RAM_SIZE = 0x71C00` = 465,920 bytes, shared by every `.data` and `.bss` symbol in the
firmware. Before Phase 4:

| Buffer | Size | Source |
| :--- | ---: | :--- |
| `clip_buffer` | 360,000 B | `AUDIO_INGEST_MAX_SAMPLES` (180,000) x 2 bytes |
| `logmel_output` | 90,112 B | `LOGMEL_MAX_FRAMES` (352) x `MEL_N_MELS` (64) x 4 bytes |
| Everything else (HAL, BSP, newlib) | ~15,728 B | derived, not itemized |
| **Total before Phase 4** | **465,840 B** | **~80 bytes of headroom** |

`SSMBackbone_State` added 9,988 bytes (`h`: 8,192 B, `conv_hist`: 1,536 B, `pooled_sum` +
`frame_count`: 260 B) -- more than the entire remaining margin, hence the exact 9,908-byte
overflow reported (9,988 requested against ~80 available).

## What was tried

**First attempt: `SRAMAHB`.** The same linker script defines a completely unused 32,768-byte
region at `0x30000000` (`SRAMAHB (rw) : ORIGIN = 0x30000000, LENGTH = 0x00008000`). Added a
`.ssm_state_ahb` output section mapped there, and tagged `ssm_state` with
`__attribute__((section(".ssm_state_ahb")))`. This built and linked cleanly, but the on-device
embedding came back as all-64-values `~0.0000` for every clip, regardless of input audio --
consistent with `SSMBackbone_State`'s reads and writes not sticking to real values, though this
was never confirmed against the chip's reference manual directly.

**Second attempt: `DTCM`.** Switched to the Cortex-M7's Tightly-Coupled Memory instead --
architecturally guaranteed CPU-accessible with no bus-domain clock-gating question, and mostly
idle: `_Min_Heap_Size` (512 B) + `_Min_Stack_Size` (1,024 B) were the only claims on a 65,536-byte
region. Same section-attribute mechanism, retargeted:

```
.ssm_state_dtcm :
{
  . = ALIGN(4);
  *(.ssm_state_dtcm)
  . = ALIGN(4);
} >DTCM
```

```c
static SSMBackbone_State ssm_state __attribute__((section(".ssm_state_dtcm")));
```

After this switch, the embedding stopped being all-zero and started returning real, substantive
(if at that point still numerically wrong, for the unrelated reason in `findings/402`) values.
That the symptom changed character between the two placements, rather than persisting, is
reasonable evidence the `SRAMAHB` placement itself was the problem, not a coincidence of timing.

**Claude comment:** "reasonable evidence" is doing real work in that sentence -- this was never
confirmed against ST's reference manual for this specific chip, only inferred from the symptom
disappearing. If `SRAMAHB` is ever revisited (its 32,768 bytes are sitting idle again), that's
worth actually resolving rather than re-trusting the same inference a second time.

## Current state

`ssm_state` lives in DTCM: 9,988 of 65,536 bytes used, ~54,012 bytes free (minus the 1,536-byte
heap/stack reservation already accounted for above). `SRAMAHB`'s 32,768 bytes are unused and
available, but distrusted per the above until actually investigated.

## Open alternative: don't buffer the whole clip at all

Both `clip_buffer` and `logmel_output` hold an entire clip's worth of data at once, which is why
they consume 97% of RAM between them. Neither strictly needs to: `AudioIngest_ReceiveClip` could
process audio hop-by-hop as it arrives over UART instead of receiving the whole clip first, and
the backbone already only needs the current frame, not the full `logmel_output` matrix --
`SSMBackbone_ProcessFrame` consumes one frame at a time by design (`findings/401`). Streaming both
stages would free the bulk of the 455 KB region.

This wasn't done now because it changes the receive/process/transmit protocol structure, not just
a buffer's placement, and `main.c`'s own comment on the current design (computing every frame into
RAM first so the transmission is "one uninterrupted burst... instead of many small gapped bursts,
which have not [been reliable]") suggests that reliability property was hard-won and worth not
disturbing while Phase 4 had its own problems to solve. Worth reconsidering if a future board's
RAM budget is tighter than this one's, or if `SRAMAHB`'s status ever gets resolved and more
headroom is wanted without a protocol change.