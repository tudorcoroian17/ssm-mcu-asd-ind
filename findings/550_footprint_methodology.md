# How footprint is measured

Describes the pipeline that produces every memory number in this project, from
the firmware build through to the four CSV files used in analysis. Read this
before quoting any footprint figure, because several numbers mean less than
their column names suggest.

## Pipeline overview

Two scripts run in sequence on different machines:

1. `scripts/Invoke-FootprintSweep.ps1` — Windows, PowerShell 5.1. Builds the
   firmware once per setup and archives the build artifacts.
2. `mcu/eval/footprint_breakdown.py` — WSL, Python with `pyelftools`. Reads the
   archived artifacts and produces the analysis CSV files.

The split exists because STM32CubeIDE runs natively on Windows while the
analysis lives in the Python project. The sweep writes across the WSL share; the
breakdown reads from the Linux side. Neither step needs the other's toolchain.

## Stage 1: the firmware sweep

### What it does per setup

For each setup directory under
`mcu/deploy/case<N>/<model_hash>/setups/<setup_id>/`:

1. Copies the setup's eight source files into the CubeIDE project. The `.c` files
   go to `Appli/Core/Src`, the `.h` files to `Appli/Core/Inc`.
2. Runs a headless clean Release build.
3. Verifies the build genuinely succeeded.
4. Copies the artifacts into `<setup_id>/online/stm32-nucleo-h7s3l8/`.

The project's original sources are backed up before the sweep and restored
afterwards.

### The build

STM32CubeIDE ships a console variant of the IDE. The sweep invokes it directly:

```text
stm32cubeidec.exe --launcher.suppressErrors -nosplash
  -application org.eclipse.cdt.managedbuilder.core.headlessbuild
  -data <scratch workspace>
  -import <project>/Appli
  -cleanBuild <project>/Release
  -no-indexer -printErrorMarkers
```

The build is always clean, never incremental. Files copied off the WSL share keep
their original modification times, which can be older than object files already
in the build tree, so an incremental build would skip the rebuild and silently
measure the previous setup.

### Required project settings

These are configured once in the IDE on the Appli project's Release
configuration. Each one silently degrades an output if missing:

| Setting | Where | What breaks without it |
| --- | --- | --- |
| `-O2` | GCC Compiler > Optimization | Release defaults to `-Os`; numbers are not comparable to a stated `-O2` |
| Generate map file | GCC Linker > General | No `.map` archived |
| `-Xlinker --print-memory-usage` | GCC Linker > Miscellaneous | Per-region table absent from the build log |
| `-fstack-usage` | GCC Compiler > Miscellaneous | No `.su` files, so no stack data |
| `-ffunction-sections`, `-fdata-sections` | GCC Compiler > Optimization | Unused code and data stay linked, inflating every number |
| `--gc-sections` | GCC Linker > General | Same |

The `--print-memory-usage` flag must be entered as two separate list items,
`-Xlinker` then `--print-memory-usage`. Eclipse splits "Other flags" on commas,
so the `-Wl,--print-memory-usage` form is broken into `-Wl` and
`--print-memory-usage`, and GCC rejects the bare `-Wl`.

### Build validation

The headless build returns exit code 0 even when compilation fails, so the exit
code alone is not trusted. A build counts as successful only if all three hold:

1. The log contains no `error:`, `undefined reference to`, `cannot find -l`, or
   `No rule to make target`.
2. An `.elf` exists and its modification time is newer than the build start.
3. A `.map` exists.

A setup failing any check is recorded as `FAILED` with the reason and the sweep
continues. The script exits with code 1 if anything failed.

The sweep also verifies that each setup provides the full expected file set
before building. A missing file would otherwise leave the previous setup's file
in the project and attribute one setup's footprint to another.

### Artifacts per setup

Written to `<setup_id>/online/stm32-nucleo-h7s3l8/`:

| File | Contents |
| --- | --- |
| `<project>.elf` | Linked image, the source for all symbol-level analysis |
| `<project>.map` | Linker map, archived for manual inspection; not parsed |
| `build.log` | Full build log, kept whether the build passed or failed |
| `size.txt` | `arm-none-eabi-size` output, Berkeley and sysv formats |
| `stack_usage.csv` | Merged `.su` records, harvested before the next clean build destroys them |

Two summary CSV files are written at the deploy root after the loop finishes:
`footprint_summary_<board>.csv` with one row per setup, and
`footprint_regions_<board>.csv` in long format with one row per memory region
per setup.

## Stage 2: the breakdown

`footprint_breakdown.py` reads the archived ELF files with `pyelftools`. It needs
no ARM toolchain and no rebuild, so it can be re-run against existing artifacts
whenever the classification rules change.

### How flash and RAM are derived

Every allocated ELF section is assigned a cost from two properties:

```text
flash = size if the section carries contents in the file (SHT_PROGBITS)
ram   = size if the section is writable (SHF_WRITE)
```

That gives three buckets:

| Section kind | Example | Flash | RAM |
| --- | --- | --- | --- |
| Read-only, in file | `.text`, `.rodata` | Yes | No |
| Writable, in file | `.data` | Yes | Yes |
| Writable, not in file | `.bss` | No | Yes |

Initialized data counts twice because its startup image is stored in flash and
copied to RAM before `main`. Symbols inherit the cost rule of the section they
belong to, so a symbol's flash and RAM figures follow directly from its section.

### Symbol classification

Each symbol is matched against an ordered rule list; the first match wins. A rule
assigns three labels:

- `group` — one of eight high-level buckets, described below.
- `component` — the tensor or subsystem, for example `in_proj` or `cmsis_tables`.
- `kind` — `weight`, `bias`, `activation`, `lut`, `code`, `descriptor`,
  `quant_param`, or `table`.

Before matching, GCC-generated suffixes are stripped: `.constprop.N`, `.isra.N`,
`.part.N`, `.lto_priv.N`, and the bare `.N` disambiguator applied to file-scope
statics. Without this, `ssm_state.4` would not match a rule written for
`ssm_state`.

Unmatched `.text` symbols fall back to `platform/library_code`. Unmatched data
stays `unclassified` deliberately: all project code is covered by an explicit
rule, so an unmatched code symbol is library code, while an unmatched data symbol
is probably a tensor the rules do not yet know about. Silently absorbing it would
hide a real gap.

### The eight groups

| Group | Contents | Why separate |
| --- | --- | --- |
| `backbone` | SSM weights, recurrent state, LUTs, descriptors, backbone code | The headline number |
| `head` | Scoring code, reference data, thresholds | Varies by scoring scheme |
| `frontend` | Hann window, mel tables, FFT buffers, feature pipeline code | Project code, not a library |
| `dsp_library` | CMSIS-DSP twiddle tables and routines | A replaceable dependency |
| `platform` | HAL, startup, newlib, board support | Irreducible overhead |
| `instrumentation` | UART and DMA path, audio ingest, `main`, printf float formatting | Measurement apparatus, absent from a deployed build |
| `reserved` | `._user_heap_stack`, 1,536 bytes | A linker reservation, not measured use |
| `unclassified` | Unmatched data and unattributed residual | Must stay near zero |

The `instrumentation` split matters for reporting. The host streaming and score
reporting path exists so the board can be measured, and a deployed product would
read from a microphone instead. Reporting overhead both with and without this
group is more defensible than picking one.

### Residual attribution

Section bytes that no symbol accounts for are reported explicitly rather than
distributed. Predictable cases are absorbed by section-name rules
(`.isr_vector`, the init arrays, `._user_heap_stack`); whatever remains appears
as `unclassified/unattributed`.

On current builds the residual is 827 flash bytes, identical across every setup
and unchanged between model sizes. A constant that does not move with the model
is structural and should be reported as its own line, not hidden.

### Memory regions

Symbol addresses are mapped to named regions using the `MEMORY` block from
`STM32H7S3L8HX_ROMxspi2.ld`:

| Region | Base | Length |
| --- | --- | --- |
| ITCM | `0x00000000` | 64 KB |
| DTCM | `0x20000000` | 64 KB |
| AXI_SRAM | `0x24000000` | 455 KB |
| SRAMAHB | `0x30000000` | 32 KB |
| XSPI2_FLASH | `0x70000000` | 128 MB |

Keep `MEMORY_REGIONS` in the script synchronized with the linker script.

### Quantization mode detection

`export_deploy_matrix.py` contains two emission paths, and the tensor naming
identifies which produced a build:

| Naming | Path | Setup prefix |
| --- | --- | --- |
| `blocks_<i>_*` | Fake quant | `w*` |
| `L<i>_*` | True int8 | `true_*` |

The mode is detected from symbols rather than parsed from setup names, because
observed evidence survives a naming convention change. Every row carries a
`quant_mode` column and an `architecture` column, currently the literal
`selective`. When `export_deploy_matrix_classic.py` starts contributing builds,
`architecture` must be derived from the setup path or `diagnostics.json` rather
than hard-coded.

The two paths differ in more than naming. The true-int8 path emits `A`, the conv
and dt_proj biases, and `norm_w` as float32 while quantizing the projections to
int8. The fake-quant path quantizes according to the per-tensor decision. This is
a real mixed-precision difference and shows up as different byte counts for the
same component.

## Stage 3: stack usage

`-fstack-usage` makes GCC emit one `.su` record per function, giving that
function's own frame size and a qualifier:

| Qualifier | Meaning |
| --- | --- |
| `static` | Frame size fixed at compile time |
| `bounded` | Frame varies but has a compile-time limit |
| `dynamic` | Variable-length array or `alloca`; no compile-time bound |

The `.su` files carry no call graph, so worst-case stack cannot be computed from
them. Two bounds are reported instead:

```text
peak_ram_low  = static RAM + largest single frame
peak_ram_high = static RAM + sum of all frames
```

The low bound is a true lower bound. The high bound assumes every function in
the image sits on one call chain, which is absurd but safe. The real figure is
the sum along the deepest path from `SSMBackbone_ProcessFrame` downward.

Any `dynamic` frame on the inference path invalidates the upper bound entirely.
If that occurs, the defensible alternative is measurement: paint the stack region
with a known pattern at startup, run a full inference, and scan for the
high-water mark.

Stack data matters here because the per-timestep SSM activations — `B`, `C`,
`delta`, and the discretized forms — are function locals, not static buffers.
They appear in no ELF symbol, so static RAM understates the true peak.

## Output files

Written to `mcu/eval/online/<board>/`:

| File | Grain | Use |
| --- | --- | --- |
| `footprint_symbols_<board>.csv` | One row per symbol per setup | Per-tensor detail, region attribution |
| `footprint_components_<board>.csv` | One row per component per setup | Weights vs activations, per-tensor rollups |
| `footprint_highlevel_<board>.csv` | One row per group per setup | The four-bucket summary |
| `footprint_stack_<board>.csv` | One row per setup | Stack bounds and peak RAM bracket |

All are written with a UTF-8 BOM by the PowerShell stage and without one by the
Python stage. Read PowerShell-written CSV files with `encoding='utf-8-sig'`.

## Integrity checks

Run both after any change to the classification rules.

No component may span two groups:

```bash
python3 -c "
import csv, collections
base = 'mcu/eval/online/stm32-nucleo-h7s3l8/'
g = collections.defaultdict(set)
for r in csv.DictReader(open(base + 'footprint_components_stm32-nucleo-h7s3l8.csv', encoding='utf-8-sig')):
    g[r['component']].add(r['group'])
print({c: sorted(v) for c, v in g.items() if len(v) > 1} or 'none')
"
```

No symbol may stay unclassified:

```bash
python3 -m mcu.eval.footprint_breakdown --list-unclassified
```

Both must come back clean before any number is quoted.

## What is exact and what is not

**Exact.** Data attribution. A symbol's size in the ELF is the number of bytes
that array occupies. Every weight tensor, the recurrent state, and every static
buffer has an unambiguous figure. These are the numbers that vary across
quantization schemes.

**Approximate.** Code attribution. At `-O2`, GCC inlines aggressively, so bytes
belonging to a small static helper are absorbed into whichever caller inlined it
and the symbol disappears. Separating project code from the HAL and CMSIS-DSP
baseline is reliable; asking how many bytes a specific operation costs is not.

**Bounded, not measured.** Stack usage, as described above.

**Misleading if quoted directly.** Two columns:

- `ram_bytes` in the summary CSV aggregates DTCM, AXI SRAM, and the noncacheable
  buffer region into one number. A DTCM overflow could hide behind plentiful AXI
  SRAM. Use the region breakdown instead.
- `percent_used` for the FLASH region in `footprint_regions` is meaningless. The
  linker script declares FLASH as the full 128 MB XSPI2 memory-mapped window
  rather than the fitted part, so a 190 KB image reports as a fraction of a
  percent. Use `used_bytes`.

## Known results worth recording

**The state buffer was costing flash.** `ssm_state` carries
`__attribute__((section(".ssm_state_dtcm")))`. A custom section defaults to
`%progbits`, so GCC wrote 5,124 bytes of zeros into the ELF even though nothing
reads them — `SSMBackbone_Reset` initializes the buffer at startup. Adding
`(NOLOAD)` to the output section in the linker script removed 5,124 bytes of
flash with no behavioral change. RAM was unaffected, since `.data` and `.bss`
both cost RAM.

**CMSIS-DSP was linking every FFT size.** `arm_rfft_fast_init_f32` switches over
all supported lengths, so every twiddle and bit-reversal table was referenced and
`--gc-sections` could discard none of them: 78,896 bytes for a build that uses
one FFT size. Switching to `arm_rfft_fast_init_1024_f32` reduced `dsp_library` to
12,804 bytes.

**Feature extraction dominates RAM.** On current builds the frontend holds 66.9%
of RAM against the backbone's 21.9% — 16,412 bytes of STFT buffers against 5,380
bytes of model state. For a project about fitting an SSM on an MCU, the memory
bottleneck is the front end, not the model.

**Claude comment:** the last of those is the one I would lead with. It is an
honest result that falls directly out of these tables, and it reframes the
contribution: the model was never the hard part. It also points at the most
promising remaining optimization, since `fft_output`, `windowed_frame`, and
`frame_history` are 4,096 bytes each and the FFT input and output can usually
share storage.