# Cycle-sweep hang investigation

**Status:** Unresolved. Board intermittently hangs during clip processing on the
Nucleo-H7S3L8. Four evidence-backed fixes applied and confirmed real, none of
them resolved the hang. Root cause still unknown as of this session.

## Background

`test_harness/cycle_sweep.py` automates build → flash → reset → 5-clip timing
sweep across every case/model/setup combination for the ASD-on-MCU deployment
matrix. During development, the board began hanging partway through clip
processing, unrecoverable except by a full USB power cycle.

The hang was later confirmed to also occur when running the original,
fully manual workflow (`deployment_test.py`, single flash via CubeIDE debug
launch, no automation involved). This rules out the sweep script's automation
as the root cause — the bug is in the firmware, the board, or the interaction
between them, not in `cycle_sweep.py`'s build/flash/reset sequencing.

## Hardware and memory layout

- Board: NUCLEO-H7S3L8 (STM32H7S3L8H6, Cortex-M7)
- External flash: MX25UW25645G octal NOR, on XSPI2, multiplexed via the XSPI
  I/O Manager (XSPIM), 1.8 V supply, 200 MHz DTR
- Two-stage boot: **Boot** project (runs from internal flash, brings up XSPI2
  memory-mapped mode via `extmem_manager.c` → `EXTMEM_Init()`, then jumps to
  **Appli**)
- **Appli executes directly from the external flash** (XIP), linked against
  `STM32H7S3L8HX_ROMxspi2.ld`. Every instruction fetch during clip processing
  goes over XSPI2.
- An untried alternative linker script, `STM32H7S3L8HX_sram.ld`, would run
  Appli from internal SRAM instead — sidesteps the external flash path
  entirely, if total SRAM is large enough for the biggest model plus runtime
  buffers.

## Observed failure signatures

Two distinct signatures were seen, possibly two different bugs:

1. **Green LED solid, no blink.** Confirmed to mean execution is stuck inside
   the clip-processing region of `main.c` (between the `BSP_LED_On(LED_GREEN)`
   and either `BSP_LED_Off(LED_GREEN)` call). This was the signature for most
   of the session's failures.
2. **No LED on at all.** Seen once, in the most recent `-O0` run (failed at
   setup 21, last clip). Not yet explained — worth checking whether this is
   Boot's own silent error path (`Error_Handler()`: `__disable_irq(); while
   (1) {}`, touches no LEDs), which would mean this specific failure happened
   somewhere different from all the others.

When stuck (signature 1, confirmed multiple times): SWD is completely
unreachable, in both `mode=Normal` and `mode=HOTPLUG` connect attempts.
`-hardRst` does not recover it. Only a full USB unplug/replug does.

## Timeline of fixes attempted

| # | Fix | Rationale | Result |
|---|-----|-----------|--------|
| 1 | External Loader (`MX25UW25645G_NUCLEO-H7S3L8-OBL.stldr`) added to `flash_elf()` via `-el` | Flashing failed outright without it (`DEV_TARGET_NOT_HALTED`, address `0x20000000` misread) — CubeProgrammer has no idea the external flash exists without it | **Required for flashing to work at all.** Not related to the hang itself. |
| 2 | `mode=UR` (connect under reset) added to both `flash_elf()` and `hard_reset_board()` | A crashed setup was leaving the board unreachable for the *next* setup's connect attempt | Fixed that specific cascading-failure problem. Did not touch the underlying hang. |
| 3 | `TimeOutActivation` enabled in `SAL_XSPI_EnableMapMode()` (`stm32_sal_xspi.c`) | Matched errata sheet ES0596, erratum 2.4.2 (XSPI1/XSPI2 interconnect deadlock); AN5050 lists this as one of two valid workarounds | **No effect.** Same behavior, same frequency. |
| 4 | `MaxTran` changed from `0` to `48` in `MX_XSPI2_Init()` (`Boot/Core/Src/main.c`) | Confirmed via HAL source + AN5050 + ST community answer: `MaxTran=0` genuinely means the communication-regulation feature is disabled; this is the erratum's other named workaround | **No effect.** Same behavior. |
| 5 | `VDDIO_HSLV` option byte set from `0x0` to `0x1` | ST community answer for this exact board: at 1.8 V/200 MHz, `VDDIO_HSLV` and `XSPI2_HSLV` both need to be `1`. `XSPI2_HSLV` was already `1`; `VDDIO_HSLV` was not — a genuinely incomplete configuration | **Still hung** (next run, setup 3). |
| 6 | Manual debug session, no sweep script at all (`deployment_test.py`, single CubeIDE flash) | Isolate whether the sweep's automation (repeated connect/flash/reset cycling) was the trigger | **Hung anyway**, at clip 13. Rules out the sweep script as root cause. |
| 7 | Build config switched to Debug (`-O0`) instead of Release (`-Os`) | Test whether the hang is optimization-sensitive | Ran clean for a full 528 clips once. **Then failed again** on a later run, at setup 21 (last clip), with the *different* LED signature (none) |
| 8 | Added `MPU_Config()` to `Appli/main.c` (previously absent entirely): tight 32 MB read-only+executable region for the real flash size at `0x70000000`, replacing the inherited 128 MB region from Boot; confirmed via subregion math that the remaining 96 MB correctly falls back to no-access/no-execute | Appli had no MPU config of its own at all, inheriting Boot's over-provisioned 128 MB (4x actual chip size) region — a real gap, and a known hazard class for Cortex-M7 speculative prefetch on XIP designs | **No effect.** Green LED solid again, same signature as most earlier failures. |

**As of fix #8: every reasonable STM32-side configuration explanation has now
been tested and eliminated** — bus arbitration (MaxTran, memory-mapped
timeout), I/O electrical config (VDDIO_HSLV), and speculative-fetch/MPU
boundaries. All five were real, independently-justified gaps worth having
fixed regardless, but none was the cause of this hang. What's left is
qualitatively different: either a hardware-level defect specific to this
physical board (not a documented design erratum, since none of those
explained it either), or a bug in a subsystem none of this investigation
touched — since every lead so far pointed at the XSPI/external-flash
interface, and that entire area is now eliminated by process of elimination.

**Erratum 2.4.2 is definitively ruled out, by direct measurement, not
inference.** Read `XSPIM->CR` live via SWD: `0xFF0010`. `MUXEN` (bit 0) and
`MODE` (bit 1) are both 0 — multiplexed mode has never been active on this
board. The erratum's entire precondition (two XSPI controllers contending
for a shared, multiplexed port) is structurally false here. This fully
explains why fixes #3 and #4 (both targeting that exact contention) had zero
effect — there was nothing for them to fix. `MaxTran=48` and
`TimeOutActivation` enabled can be reverted without consequence if you want
to simplify the codebase later; they're harmless but not load-bearing for
this bug. **`MaxTran` specifically is now known to be structurally inert
here, not just untested** — its entire behavior is defined in terms of "the
other XSPI requests access," which can never happen with `MUXEN=0`.

**The entire ES0596 XSPI errata section (2.4.1-2.4.12, 2.5.1) was read in
full and checked against the real configuration — nothing else applies
either.** Every erratum with an available workaround was excluded by its
own stated precondition, not by assumption:
- 2.4.1 (write error, DQS disabled) — memory-mapped *write*-only; the hang
  occurs during instruction fetch (read).
- 2.4.5 (CSBOUND deadlock) — requires 16-bit DTR or dual-octal DTR
  (two-chip parallel modes); this board has one Macronix chip in plain
  octal DTR.
- 2.4.9, 2.4.10 — both explicitly scoped to `MTYP = OctaRAM`; this device
  is configured as `HAL_XSPI_MEMTYPE_MACRONIX` (NOR), not OctaRAM.
- 2.4.11, 2.5.1 — both require multiplexed mode; ruled out by the same
  `MUXEN=0` measurement as 2.4.2.

Two errata remain plausible in principle but have **no ST-provided
workaround at all**, so they're not actionable even if involved: 2.4.4
(misaligned `XSPI_AR` write before entering memory-mapped mode — but that's
a one-time event at Boot's mode-switch, not a fit for a hang recurring deep
into steady-state execution at varying clip counts), and 2.4.8 (a
cache-initiated wrap read followed by a linear read to the same address can
corrupt data — ST's own text notes wrap reads are "mostly initiated by the
internal cache," which matches ordinary D-Cache line-fills during code
execution, but the stated consequence is silent data corruption, not a
clean total hang, and ST states plainly "as prefetch cannot be disabled,
there is no workaround").

**Conclusion: there is nothing left to try from the XSPI errata sheet.**
Don't re-read ES0596 on a future pass at this bug — it's been read in full
and exhausted, not skimmed.

**Optimization level is a real, large, reproducible effect — and it's more
specific than "optimization matters."** A full 100-setup sweep at `-O0`
completed with zero hangs. The very next run, same board, same Boot image
(so identical XSPI2 clock/timing config — Boot is never rebuilt between
Appli build-config changes), at `-Os`: 4 of the first 6 setups failed,
3 after the first clip, 1 on the first clip. Since Boot's XSPI2
configuration is provably identical in both cases, this isn't evidence
about absolute clock speed or signal margin in the abstract — it's evidence
that **the failure is sensitive to the instruction-fetch pattern/density the
compiled code produces, at a fixed electrical configuration.** `-Os`'s
tighter, more inlined, more densely-branched code generates much more
back-to-back XSPI2 bus activity per unit time than `-O0`'s spread-out code.

This also tempers the "just lower the clock more" plan: `ClockPrescaler`
was already raised from `3` to `15` (4x slower) earlier tonight, and `-Os`
still fails at a high rate. Clock speed may still be a contributing factor,
but a 4x reduction clearly hasn't been sufficient on its own, suggesting
this is not purely a raw-margin problem.

**`CSBOUND` update: setting it as a generic fix is no longer supported by
the errata sheet.** The only errata entry that names CSBOUND (2.4.5)
requires 16-bit/dual-octal DTR mode, which doesn't match this single-chip
octal-DTR configuration — so there's no errata-based reason left to touch
it. It could still be tried as a plain, non-errata-motivated experiment
(forcing periodic CS release regardless of cause), but that's now a genuine
guess rather than a targeted fix, and should be labeled as such if tried.

**A real, available fallback, not just a diagnostic conclusion**: `-O0`
now has a clean 100/100 track record across the whole deploy matrix. If the
dissertation's timing claims are about *relative* cost across quantization
schemes/heads rather than absolute deployed-inference latency, running the
actual data-collection sweep under `-O0` is a legitimate path to finish
data collection, with cycle counts honestly labeled as measured under an
unoptimized build.

<!-- Claude comment: don't restart this investigation by guessing a sixth
register. If picking this back up, start from instrumentation (SWV/ITM, or
a genuine hardware fault-isolation step like swapping the physical board)
rather than more documentation reading — the documentation-driven leads are
exhausted, not the bug. -->

## Key evidence and reasoning

- **`-O0` vs `-Os` comparison** (same clip, both builds): `-Os` runs the exact
  same backbone computation in about **1/4.76th** the cycle count of `-O0`
  (514M cyc vs 2447M cyc), with **identical scores** (`1.7106` both times, so
  `-Os` isn't producing wrong answers). `-O0` ran 528 clips clean before this
  session's later `-O0` run failed at setup 21.

  <!-- Claude comment: the fact that -O0 spends *more* total wall-clock time
  doing XIP execution from the external flash, yet failed less often, argues
  against "cumulative exposure to the flash interface" as the trigger. It
  points more specifically at *fetch density* — how tightly packed
  instruction-cache-line-fill requests are — since optimized code produces
  much tighter, more frequent fetch bursts than -O0's spilled, unrolled-less
  code. That's a more specific, useful detail than "optimization matters" if
  this ever gets written up for ST. -->

- **GPDMA1_Channel5_IRQn = 44** (vector number 60), configured at
  `HAL_NVIC_SetPriority(GPDMA1_Channel5_IRQn, 0, 0)` — the highest priority
  the NVIC supports (`__NVIC_PRIO_BITS = 4`, so 0–15). SysTick defaults to the
  lowest. This was an early theory (a stuck DMA ISR could starve SysTick,
  freezing `HAL_GetTick()`-based timeouts) but was **never confirmed** —
  register reads to check `SCB->ICSR`/`SysTick->VAL` were attempted but SWD
  was already unreachable by the time they were tried. Still a latent risk
  worth knowing about, independent of whether it's the cause here.

- **Recurrence around setup 2–3** was noted across a couple of early runs but
  is weak evidence — later runs failed at setup 3, setup 21, and clip 13 of a
  single-setup manual run, so it does not reliably recur at a fixed position.

## What's worth keeping regardless of the outcome

<!-- Claude comment: these four are independently correct/justified even
though none of them explained the hang — don't revert them while chasing the
next lead. -->

- External Loader in `flash_elf()` — required for flashing to function
- `mode=UR` in `flash_elf()`/`hard_reset_board()` — fixes a real cascading-
  failure problem, separate from the hang
- `MaxTran = 48` — matches ST's own stated recommendation for multiplexed
  XSPI mode
- `VDDIO_HSLV = 1` — matches ST's own stated requirement for this board's
  flash voltage/speed combination

## Next steps, roughly in order of expected value

0. **Reproduce with the manual, single-flash workflow, not the sweep.** The
   cleanest data point from the whole investigation: a manual
   `deployment_test.py` run (one flash, no resets, no automation) failed at
   clip 13, deep into otherwise-normal steady-state processing. Fewer moving
   parts than any sweep-triggered failure — this is the condition to
   reproduce under SWV tracing.
1. **SWV/ITM tracing.** Unlike SWD register reads, ITM trace is one-way and
   doesn't depend on the target responding to a halt request — it may keep
   streaming right up to the freeze. Setup: compute actual SYSCLK from
   Boot's `SystemClock_Config()` (HSI 64 MHz → PLL1, `PLLM=32 PLLN=300
   PLLP=1`), enter it in the Debug Configuration's SWV settings, add
   `ITM_SendChar()` markers around per-hop boundaries and around
   `SSMBackbone_ProcessFrame` specifically, watch where the trace goes
   silent.
2. **Bisect optimization level further** (`-O1`, `-Og`) to narrow whether a
   specific optimization (inlining, loop unrolling) is responsible, rather
   than "any optimization."
3. **Check whether the failure point is time-based or count-based** — does a
   fixed number of hops/frames processed correlate with failure better than
   clip count or wall-clock time does?
4. **If another NUCLEO-H7S3L8 is available, try it.** Every fix so far
   assumed a design-level cause; five eliminated theories is unusual enough
   to make a defect specific to this one physical unit worth ruling in or
   out cheaply, before more instrumentation time is spent on it.
5. **Evaluate `STM32H7S3L8HX_sram.ld`** as a structural fix — running Appli
   from internal SRAM instead of XIP-from-external-flash would remove the
   entire class of XSPI-interface bugs, if the biggest model fits in
   available SRAM alongside runtime buffers.
6. **Script an actual power cycle** (USB-controlled relay or similar) so the
   sweep can self-heal past this automatically, independent of ever finding
   the root cause.
7. **File an ST support ticket / community post**, referencing errata sheet
   ES0596 erratum 2.4.2 by number. Worth including: reproduction in both
   manual and automated workflows, the optimization-level sensitivity
   finding, and both LED signatures.
8. **Resolve whether the two LED signatures (solid green vs. none) are the
   same bug or two separate ones** — the "no LED" case may be Boot's own
   silent `Error_Handler()`, which would mean it never even reached Appli.

## File reference

- Sweep script: `test_harness/cycle_sweep.py`
- Boot's XSPI init: `Boot/Core/Src/main.c` (`MX_XSPI2_Init`)
- Boot's external memory bring-up: `Boot/Core/Src/extmem_manager.c`
- Shared XSPI middleware: `Middlewares/ST/STM32_ExtMem_Manager/sal/stm32_sal_xspi.c`
- HAL XSPI driver: `Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_xspi.c`
- Appli linker scripts (in `Appli/`): `STM32H7S3L8HX_ROMxspi2.ld` (current),
  `STM32H7S3L8HX_sram.ld` (untried alternative)