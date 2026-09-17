# How power, current, and energy are measured

Describes the pipeline that produces every current, energy, and power figure
in this project, from the PPK2 hardware setup through the five scripts that
turn a raw capture into a plotted bar chart. Read this before quoting any
number from `pec/`, because several steps here exist specifically to head off
a wrong answer that would otherwise look entirely plausible.

## Hardware setup

**Instrument:** Nordic PPK2, in **Ampere-meter mode**. The board is powered
normally through its own USB/ST-LINK path; PPK2 sits in series and only
measures the current already flowing, rather than sourcing it (that's Source
mode, not used here).

**Current-sense jumper:** JP2, silkscreened `VDD_MCU` on the
NUCLEO-H7S3L8 (confirmed against UM3276, not assumed from a similar board —
comparable Nucleo-144 boards sometimes label this jumper differently). Removed
to break the on-board supply path, with PPK2 wired across the two pins:

| PPK2 | JP2 pin | Why |
| --- | --- | --- |
| VIN | pin 1 | comes from the board's own 3V3 regulator |
| VOUT | pin 2 | feeds VDD_MCU directly — this is where current is actually consumed |
| GND | any GND on the Zio/morpho header | common reference |

**GPIO instrumentation:** three digital lines, chosen to avoid soldering by
using pins already exposed on the Zio (Arduino-compatible) headers, and
confirmed free of every other use on this board — checked against the
project's own `.ioc` (which locks `PB7`/`PC13`/`PD8`/`PD9`/`PD10`/`PD13` for
`LD3`/`B2`/USART3-VCP/`LD1`/`LD2`) and against UM3276's SWD, SWO/trace,
oscillator, Ethernet, and USB pin tables.

| Signal | MCU pin | Zio label | PPK2 digital input |
| --- | --- | --- | --- |
| Feature | PF3 | D4 (CN10) | D0 |
| Backbone | PF4 | D7 (CN10) | D1 |
| Total | PF5 | D8 (CN7) | D2 |

D3–D7 are left unconnected and read a constant 0 throughout every capture —
a useful negative control confirming the connected lines aren't crosstalking
onto the unconnected ones.

**PPK2's logic port needs its own GND *and* VCC connected**, not just GND.
Both are references for the digital-input comparators (Nordic's own answer to
this exact question: *"The Logic Port VCC and GND are for references for the
logic lines"*), not just power/ground in the usual sense. Missing VCC alone
produces a clean, unwavering 0 on every channel — indistinguishable, from the
software side, from the pins never toggling at all. VCC ties to any 3V3 pin on
the Zio/morpho header, same voltage domain as the GPIO pins themselves.

## Firmware instrumentation

`main.c` toggles the three GPIO pins directly via `GPIOF->BSRR` writes,
hand-added in `USER CODE` blocks (not through CubeMX/the `.ioc` — GPIOF isn't
configured there at all, so regeneration can't touch or remove any of this),
at the same `DWT->CYCCNT` boundaries the cycle-timing instrumentation already
uses.

**Feature** and **backbone** each bracket exactly their own stage, once per
hop.

**Total** is used for two different spans, not one continuous window for the
whole clip:

1. Once per hop, `t0`→`t3`: feature, normalize, and backbone back to back,
   with no gap. Normalize has no dedicated line of its own, but its current
   still falls inside this window.
2. Once per clip, bracketing the head-scoring call, since head runs once
   after every hop is done, not once per hop.

A single continuous window from before the first hop to after head would
instead measure mostly idle time: each hop's `AudioIngest_WaitHopComplete()`
blocks on the *next* hop's audio arriving over UART, which at 115200 baud
takes roughly 90 ms per 512-sample hop — far longer than the few milliseconds
of actual feature+normalize+backbone compute per hop. That wait is an
artifact of feeding pre-recorded clips over a slow debug link, not part of
the model's real inference cost, so the two-bracket design deliberately
excludes it.

Head's own duration (roughly 3 µs for the euclidean head, up to ~40 µs for
knn16) sits at or below PPK2's 10 µs sample period, so it isn't reliably
resolvable as its own pulse — some captured clips show the expected
`n_hops + 1` total pulses, some show only `n_hops` (the head pulse happened to
fall entirely between two samples). This costs at most a few tens of
microseconds against a clip total dominated by backbone (over a second), so
it doesn't meaningfully affect any total figure — it only means head's *own*
current draw was never independently measured, only folded into whichever
total window it landed in.

## Pipeline overview

Five scripts run in sequence, alternating between Windows (where the
hardware lives) and WSL (where the rest of this project's tooling lives):

| # | Script | Where | Needs the physical PPK2? |
| --- | --- | --- | --- |
| 1 | `test_harness/power_sweep.py` | Windows | Yes |
| 2 | `test_harness/power_decode.py` | Windows | Yes |
| 3 | `mcu/eval/compute_current_results.py` | WSL | No |
| 4 | `mcu/eval/compute_energy_power_results.py` | WSL | No |
| 5 | `mcu/eval/plot_pec.py` | WSL | No |

Only the first two steps touch hardware; everything after stage 2 is pure
arithmetic on already-written CSVs and can be re-run at will without the
board or the PPK2 connected.

## Stage 1: power_sweep.py — capture

For each of the 100 case/model/setup combinations: build, flash, hardware
reset, then send a **fixed pair of clips** — the first `normal`-labeled and
first `anomaly`-labeled clip, by sorted name, from that setup's own test
split — over UART, reading back the embedding, head score, and DWT cycle
counts the firmware already reports (no protocol change needed; all three
were already on the wire before this work started).

The same two clips are used for every setup and every re-run, so power
figures are comparable across setups and `--resume` means something. `--resume`
skips any setup already marked `OK` in a prior run's summary CSV.

**PPK2 capture is one continuous session per setup, started before flash and
reset, not one session per clip.** With JP2 removed, PPK2 is the *only* power
path to VDD_MCU — it has to be actively measuring, with `toggle_DUT_power("ON")`
issued, before SWD can reach the target at all. Starting the capture per
clip (or after flash) leaves the board unpowered during flash/reset, which
fails with `STM32_Programmer_CLI` reporting a plausible-looking voltage
(sensed upstream of the now-open jumper) while SWD itself reports
`Unable to get core ID`.

`toggle_DUT_power("ON")` is required despite not appearing in `ppk2-api`'s
documented Ampere-mode sequence — confirmed by reading the library's actual
source (`DEVICE_RUNNING_SET`, opcode `0x0c`), not inferred from the README.

Nothing is decoded during this stage. Raw PPK2 bytes are written straight to
disk (`<setup_id>_raw.bin`), alongside a small `markers.csv` of
`time.monotonic()` timestamps (`ppk2_measuring_start`, `flash_done`,
`reset_done`, `<clip>_uart_start`/`_uart_end`) — deliberately kept separate
from decoding so a flaky PPK2 decode step can never take down the sweep
itself, the same failure-decoupling instinct `cycle_sweep.py` already applies
to build/flash/UART.

## Stage 2: power_decode.py — decode

Needs the PPK2 physically connected: `decode_capture()` opens a live
connection to re-derive calibrated current values from the raw bytes.

Per-sample timestamps are **reconstructed**, not read directly from the raw
file — the file only records one timestamp per USB read chunk (tens of
milliseconds' worth of samples), far coarser than a single hop's duration.
Timestamps are rebuilt at PPK2's fixed 100 kSps rate, anchored off each
chunk's own read time, which shares a clock (`time.monotonic()`) with the
marks recorded in stage 1's parent process.

Each clip's window is sliced out of the setup's one continuous capture using
its `uart_start`/`uart_end` marks, then the three digital channels are
edge-detected into (mean current, duration) pairs per pulse — one row per
hop, plus one `phase="head"` row when total's head pulse was actually caught
(see the firmware section above for when it isn't). Two `<clip_name>_frames.csv`
files are written per setup, one per clip, into
`mcu/deploy/case<N>/<model_hash>/setups/<setup_id>/online/stm32-nucleo-h7s3l8/current/`.

## Stage 3: compute_current_results.py — aggregate

Pure aggregation, mirrors `collect_timing_results.py`'s pattern exactly. Rolls
every setup's frame CSVs into one master `current_results.csv`
(`mcu/eval/online/stm32-nucleo-h7s3l8/pec/`), one row per (case, model,
setup, clip), reporting:

- **Time-weighted mean current** per section — `sum(current_i * delta_t_i) /
  sum(delta_t_i)`, not a naive mean of per-pulse means, since it's the
  energy-consistent way to average currents over windows of slightly unequal
  duration.
- **Total delta_t** per section — the sum of that section's pulse durations
  across the whole clip.
- **DWT cross-check columns** (`feature_cycles_dwt`, `backbone_time_dwt_s`,
  etc.) — the firmware's own `DWT->CYCCNT`-based cycle counts for the same
  clip, read from `timing.csv` on the Windows side. DWT is a small trace
  unit built into the Cortex-M7 core; its counter ticks once per CPU clock
  cycle, giving an exact duration with zero sampling error, completely
  independent of PPK2's external 10 µs-resolution sampling. This is the
  ground truth PPK2-derived durations get checked against.

## Stage 4: compute_energy_power_results.py — derive energy and power

Straightforward unit conversion, applied to both the master CSV and every
per-clip frame CSV:

```text
power_mw  = current_ua * voltage_mv * 1e-6
energy_mj = power_mw * delta_t_s
```

Voltage is a **required** argument (3300 mV, measured on the bench) rather
than defaulted from `set_source_voltage()`'s value — that call only
calibrates PPK2's own ADC math, it isn't a guarantee of the real VDD_MCU rail
voltage. `current_ua` is dropped from the derived files (redundant — it's
already in `current_results.csv`); `delta_t_s` is kept in both, since it's
free context even though power itself doesn't need it to compute. DWT columns
are deliberately not propagated past stage 3, to keep the energy/power files
a clean, minimal conversion.

## Stage 5: plot_pec.py — plot

Six PNGs: {current, energy, power} × {normal, anomaly} clip, all written into
the same `pec/` folder as the three CSVs. One horizontal bar per model per
setup, colored by model config signature, split into subplots by selectivity
role (mirroring `plot_timing.py`/`plot_footprint_highlevel.py`'s structure).

Only **total** is plotted — feature, backbone, and total are not additive the
way `plot_timing.py`'s four stages or `plot_footprint_highlevel.py`'s groups
are (total already includes feature, backbone, normalize, and head), so
stacking them the same way those two scripts stack their components would
double-count feature and backbone inside total. Per-section current/energy/
power figures are still available in the CSVs; this first pass of plotting
just doesn't attempt to show all three non-additively in one image yet.

## Output files

| File | Grain | Contents |
| --- | --- | --- |
| `online/<board>/current/<clip>_frames.csv` | One row per hop (+ head) | Per-section current (µA) and duration |
| `online/<board>/energy/<clip>_frames.csv` | Same | Per-section energy (mJ) |
| `online/<board>/power/<clip>_frames.csv` | Same | Per-section power (mW) |
| `mcu/eval/online/<board>/pec/current_results.csv` | One row per case/model/setup/clip | Time-weighted mean current, total duration, DWT cross-check |
| `mcu/eval/online/<board>/pec/energy_results.csv` | Same | Total energy per section |
| `mcu/eval/online/<board>/pec/power_results.csv` | Same | Mean power per section |
| `mcu/eval/online/<board>/pec/{current,energy,power}_total_{normal,anomaly}.png` | One bar per model per setup | Total-only comparison chart |

`pec` is the umbrella name for this whole trio of measurements (current,
energy, power) — chosen so "power" could stop doing double duty as both the
general term and one specific metric's name, which is exactly what caused a
folder-naming collision earlier in this work.

## Resolved issues worth keeping in mind

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Flash/SWD fails with "Unable to get core ID" once PPK2 is wired in series and JP2 is removed | PPK2 hadn't started measuring yet when flash/reset were attempted; with JP2 open, PPK2 is the only power path, so the target was genuinely unpowered | Start the PPK2 capture (and `toggle_DUT_power("ON")`) before flash/reset, not after |
| Same failure persists even once PPK2 is "measuring" before flash | `toggle_DUT_power("ON")` was never called — not part of `ppk2-api`'s documented Ampere-mode sequence, but required regardless | Add `toggle_DUT_power("ON")` (confirmed real in the library's own source) |
| All 8 digital channels read a constant, unwavering 0 across an entire capture, even with D0/D1/D2 wired | PPK2's logic port GND was connected, but VCC (also a required reference for the logic thresholds) was not | Connect logic-port VCC to any 3V3 pin on the Zio/morpho header |
| `total`'s pulse count is one short of `n_hops + 1` on some clips | Head's own duration (a few µs) is at or below PPK2's 10 µs sample period; whether its pulse gets caught depends on where its edges fall relative to the sample grid | Not a bug — expected some fraction of the time, and immaterial to the total given how small head's contribution is |
| Early `decode_capture()` gave nonsensical (too-coarse) per-hop durations | Every sample in a USB read chunk was stamped with the same, single per-chunk timestamp, far coarser than a hop's actual duration | Reconstruct per-sample timestamps at PPK2's fixed 100 kSps rate instead |

## What's exact and what's approximate

**Exact, within PPK2's stated accuracy.** Current readings themselves — the
PPK2's own calibrated ADC conversion, unrelated to anything in this project's
own tooling.

**Resolution-limited, not wrong.** Any single window shorter than ~10 µs
(head, specifically) can't get its own reliable pulse. Windows well above
that (feature, backbone, total) are unaffected.

**A single global assumption.** Voltage (3300 mV) is measured once and
applied uniformly to every setup and clip, on the assumption that the
regulator's output doesn't meaningfully shift with the ~100–230 mA load range
actually observed — a reasonable assumption for a regulated rail, but an
assumption, not a per-setup measurement.

**Not independently measured at all.** Normalize's and head's own current —
both are only ever visible bundled inside `total`'s window, never as their
own figure, since only three GPIO channels were instrumented.

## Known results worth recording

**Normal and anomaly clips draw essentially identical power.** On the one
setup profiled in detail during this work (`case1/086acf0275b8/
full_fp32_euclidean_fp32`), total power came out to 336.1 mW for the normal
clip and 336.5 mW for the anomaly clip — under 0.2% apart. Expected, given
the model runs the same fixed computation regardless of input content, but
worth having as a stated confirmation rather than an assumption.

**Backbone dominates both energy and power.** For that same setup: feature
consumed 60.8 mJ over its ~0.18 s of active time; backbone consumed 458.6 mJ
over ~1.36 s — more than 7x feature's energy, consistent with backbone being
the dominant cost in the cycle-count timing data as well.

**Claude comment:** if this ends up in the thesis's methodology section, the
detail worth leading with isn't a number — it's the two-bracket design for
`total`. A reviewer could reasonably ask why total power wasn't just measured
as one continuous window from clip start to clip end; the honest answer is
that doing so on this hardware would mostly measure UART idle time (~90 ms of
waiting per ~4 ms of real compute, per hop), not model inference cost. Stating
that plainly is more convincing than a number that looks precise but is
quietly dominated by a test-harness artifact.