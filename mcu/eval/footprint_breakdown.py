#!/usr/bin/env python3
"""Extracts per-symbol and per-component footprint from archived sweep ELF files.

Reads the ELF files already stored in the deploy tree by Invoke-FootprintSweep.ps1.
No rebuild and no ARM toolchain are required. Writes four CSV files: one row per
symbol, one per component, one per high-level group, and one per setup for stack
usage.

Example:
    python -m mcu.eval.footprint_breakdown
    python -m mcu.eval.footprint_breakdown --case case1 --list-unclassified
    python -m mcu.eval.footprint_breakdown --struct SSMBackbone_State
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile

# Memory regions, transcribed from STM32H7S3L8HX_ROMxspi2.ld. Keep in sync with
# the MEMORY block if the linker script changes.
MEMORY_REGIONS = [
    ("ITCM", 0x00000000, 0x00010000),
    ("DTCM", 0x20000000, 0x00010000),
    ("AXI_SRAM", 0x24000000, 0x00071C00),
    ("AXI_SRAM_NC", 0x24071C00, 0x00000400),
    ("SRAMAHB", 0x30000000, 0x00008000),
    ("BKPSRAM", 0x38800000, 0x00001000),
    ("XSPI2_FLASH", 0x70000000, 0x08000000),
]

# export_deploy_matrix.py emits blocks_<i>_* on the fake-quant path and L<i>_* on
# the true-int8 path. One backbone symbol identifies which produced a build.
QUANT_MODE_NAMES = {"L": "true_int8", "blocks": "fake_quant", "unknown": "unknown"}

# Every setup swept so far comes from export_deploy_matrix.py. Derive this from
# the setup path or diagnostics.json once export_deploy_matrix_classic.py starts
# contributing builds.
ARCHITECTURE = "selective"

# Matches either emission path: L0_foo or blocks_0_foo.
LAYER = r"^(?:L|blocks_)\d+_"

# Ordered rules: the first match wins, so specific patterns must precede general
# ones. Each entry is (pattern, group, component, kind). A component must belong
# to exactly one group across the whole table.
CLASSIFICATION_RULES = [
    # ----------------------------------------------------------------------
    # Backbone: runtime activations. In a selective SSM these are computed per
    # timestep from x_proj, so they appear only if the export stores them.
    # ----------------------------------------------------------------------
    (LAYER + r"[AB]_bar_q$", "backbone", "discretized_state_space", "activation"),
    (LAYER + r"(B|C)_q$", "backbone", "selective_bc", "activation"),
    (LAYER + r"delta_q$", "backbone", "selective_delta", "activation"),
    (LAYER + r"(conv_state|ssm_state|scratch|tmp|buf)", "backbone",
     "layer_scratch", "activation"),

    # ----------------------------------------------------------------------
    # Backbone: stored weights. Both emission paths are covered so true-int8 and
    # fake-quant builds classify into the same components. The (_q)? forms let a
    # single rule match the float and quantized spellings of one tensor.
    # ----------------------------------------------------------------------
    (LAYER + r"in_proj(_[uz])?_w_q$", "backbone", "in_proj", "weight"),
    (LAYER + r"out_proj_w_q$", "backbone", "out_proj", "weight"),
    (LAYER + r"x_proj_w_q$", "backbone", "x_proj", "weight"),
    (LAYER + r"dt_proj_w_q$", "backbone", "dt_proj", "weight"),
    (LAYER + r"dt_proj_b(_q)?$", "backbone", "dt_proj", "bias"),
    (LAYER + r"dt_q$", "backbone", "dt_init", "weight"),
    (LAYER + r"conv_w_q$", "backbone", "conv", "weight"),
    (LAYER + r"conv_b(_q)?$", "backbone", "conv", "bias"),
    (LAYER + r"A(_log)?_q$", "backbone", "A", "weight"),
    (LAYER + r"A$", "backbone", "A", "weight"),
    (LAYER + r"D(_q)?$", "backbone", "D", "weight"),
    (LAYER + r"norm_w(_q)?$", "backbone", "layer_norm", "weight"),
    (LAYER + r"lut_", "backbone", "activation_lut", "lut"),

    # Per-tensor quantization metadata, reported as its own component. Move this
    # rule above the per-tensor rules only if you want scales rolled into the
    # tensor they belong to.
    (LAYER + r".*_(scale|zp|zero_point|shift|mult)$", "backbone",
     "quant_params", "quant_param"),

    # ----------------------------------------------------------------------
    # Backbone: code, descriptors, final norm, persistent state.
    # ----------------------------------------------------------------------
    (r"^SSMBackbone_", "backbone", "backbone_code", "code"),
    (r"^ssm_rmsnorm$", "backbone", "backbone_code", "code"),
    (r"^ssm_(layers|blocks)$", "backbone", "layer_descriptors", "descriptor"),
    (r"^ssm_final_norm_(w|w_data_q)$", "backbone", "final_norm", "weight"),
    (r"^ssm_final_norm_(scale|w_data_scale)$", "backbone", "final_norm",
     "quant_param"),
    (r"^ssm_state$", "backbone", "recurrent_state", "activation"),
    (r"^pooled_embedding$", "backbone", "pooled_embedding", "activation"),

    # ----------------------------------------------------------------------
    # Head: scoring code and reference data, split across ssm_distance_head.c
    # and ssm_head_ref.c but one head per build.
    #
    # ssm_norm_mean and ssm_norm_std are provisionally head. If they are input
    # feature statistics consumed by SSMBackbone_NormalizeFrame rather than
    # embedding statistics used before scoring, move these two to frontend.
    # ----------------------------------------------------------------------
    (r"^SSM(Distance|Ref|Knn)Head_", "head", "scoring_code", "code"),
    (r"^ssm_ref_", "head", "reference_data", "weight"),
    (r"^ssm_norm_(mean|std)$", "head", "score_normalization", "weight"),
    (r"^ssm_threshold", "head", "threshold", "weight"),
    (r"^head_result$", "head", "scoring_buffers", "activation"),

    # ----------------------------------------------------------------------
    # Feature extraction: your own code, tables and buffers. Kept separate from
    # dsp_library so a replaceable dependency stays distinguishable.
    # ----------------------------------------------------------------------
    (r"^FeaturePipeline_", "frontend", "frontend_code", "code"),
    (r"^hann_window$", "frontend", "window_table", "table"),
    (r"^mel_", "frontend", "mel_filterbank", "table"),
    (r"^(fft_output|windowed_frame|frame_history|power_spectrum|hop_buf|"
     r"mel_energies|log_mel|rfft_instance)$", "frontend", "frontend_buffers",
     "activation"),

    # ----------------------------------------------------------------------
    # CMSIS-DSP.
    # ----------------------------------------------------------------------
    (r"^(twiddleCoef|armBitRevIndexTable|armRecipTable|realCoef)", "dsp_library",
     "cmsis_tables", "table"),
    (r"^arm_", "dsp_library", "cmsis_code", "code"),

    # ----------------------------------------------------------------------
    # Instrumentation: the host audio and reporting path. A deployed product
    # reads from a microphone and does not format floats over UART, so these
    # bytes are measurement apparatus rather than deployment cost.
    # ----------------------------------------------------------------------
    (r"^AudioIngest_", "instrumentation", "audio_ingest", "code"),
    (r"^(hop_rx_|TransmitAcked)", "instrumentation", "audio_ingest", "data"),
    (r"^main$", "instrumentation", "harness_main", "code"),
    (r"^timing_values$", "instrumentation", "timing", "data"),
    (r"^(__io_putchar|printf|iprintf|fprintf|fiprintf|puts|_puts_r)$",
     "instrumentation", "reporting_printf", "code"),
    (r"^(UART_|huart|hcom_|handle_GPDMA|handle_UART|UARTPrescTable)",
     "instrumentation", "uart_dma", "code"),
    (r"^(BSP_COM|COM_ActiveLogPort|BspCOMInit)", "instrumentation", "uart_dma",
     "code"),

    # ----------------------------------------------------------------------
    # Platform: libc, libm, libgcc. The printf float-formatting cluster is
    # reached only through the reporting path but lives in newlib, so it is
    # counted here rather than as instrumentation.
    # ----------------------------------------------------------------------
    (r"^(_dtoa_r|_printf_|_vfi?printf_r|__sf|__sw|__sr|__ss|__sc|__smakebuf|"
     r"__exponent|__cvt|quorem|__(multiply|multadd|pow5mult|lshift|mdiff|d2b|"
     r"i2b|hi0bits|lo0bits|mcmp)|_B(alloc|free)|p05|__mprec)",
     "platform", "libc_printf", "code"),
    (r"^(_?m(alloc|emcpy|emset)|_free_r|_calloc_r|_malloc_r|sbrk_aligned|"
     r"__malloc_|__lock___|_sbrk|__sbrk_heap_end|strlen)",
     "platform", "libc_core", "code"),
    (r"^(logf|sqrtf|__ieee754_|__math_|with_errnof|__logf_data|__exp2f_data)$",
     "platform", "libm", "code"),
    (r"^(__aeabi_|__udivmoddi4)", "platform", "libgcc", "code"),
    (r"^(_exit|abort|raise|_raise_r|__assert_func|_(read|write|close|lseek|"
     r"fstat|isatty|kill|getpid)(_r)?|__errno|errno|__libc_init_array|"
     r"__retarget_lock_|_localeconv_r|__ascii_|std|stdio_exit_handler|"
     r"cleanup_stdio|global_stdio_init|_fwalk_sglue|__sinit|__sfp_lock_|"
     r"__stdio_exit_handler|_fflush_r|_impure|__global_locale|__sglue|_ctype_)",
     "platform", "libc_core", "code"),

    # ----------------------------------------------------------------------
    # Platform: board support, HAL, startup.
    # ----------------------------------------------------------------------
    (r"^(BSP_LED|BSP_PB|BUTTON_|LED_PORT|LED_PIN|FatalBlink)", "platform",
     "board_support", "code"),
    (r"^(HAL_|__HAL_|RCC_|MPU_|MX_|Error_Handler|SystemClock|SystemInit|"
     r"SystemCore|AHBPrescTable|APBPrescTable|uwTick|hpb_exti)", "platform",
     "hal", "code"),
    (r"_Handler$|_IRQHandler$|^g_pfnVectors$|^Reset_Handler$", "platform",
     "startup", "code"),
]

# Sections without symbols, attributed by section name so their bytes do not
# inflate the unattributed residual.
SECTION_RULES = [
    (r"^\._user_heap_stack$", "reserved", "heap_stack_reservation"),
    (r"^\.isr_vector$", "platform", "startup"),
    (r"^\.(init|fini|preinit)_array$", "platform", "startup"),
    (r"^RW_NONCACHEABLE$", "platform", "noncacheable_buffer"),
]

COMPILED_RULES = [(re.compile(p), g, c, k) for p, g, c, k in CLASSIFICATION_RULES]
COMPILED_SECTION_RULES = [(re.compile(p), g, c) for p, g, c in SECTION_RULES]

GROUP_ORDER = [
    "backbone", "head", "frontend", "dsp_library",
    "platform", "instrumentation", "reserved", "unclassified",
]

# GCC appends suffixes for clones and file-scope statics: .constprop.N, .isra.N,
# .part.N, .lto_priv.N, and a bare .N disambiguator.
SUFFIX_PATTERN = re.compile(
    r"\.(constprop|isra|part|cold|lto_priv|localalias|clone)\.\d+$|\.\d+$"
)


@dataclass
class SymbolRow:
    """One symbol from one ELF file, with its footprint attribution."""

    case: str
    model: str
    setup: str
    symbol: str
    group: str
    component: str
    kind: str
    layer: str
    section: str
    region: str
    address: int
    bytes: int
    flash_bytes: int
    ram_bytes: int


def normalize(symbol: str) -> str:
    """Removes GCC-generated suffixes so rules match the source-level name."""
    previous = None
    while previous != symbol:
        previous = symbol
        symbol = SUFFIX_PATTERN.sub("", symbol)
    return symbol


def region_for(address: int) -> str:
    """Returns the memory region name containing an address."""
    for name, origin, length in MEMORY_REGIONS:
        if origin <= address < origin + length:
            return name
    return "unmapped"


def classify(symbol: str, section: str) -> tuple[str, str, str]:
    """Returns the (group, component, kind) for a symbol.

    Falls back to platform library code for unmatched .text symbols, since all
    project code is covered by an explicit rule. Unmatched data stays
    unclassified so a new tensor name is never silently absorbed.
    """
    name = normalize(symbol)
    for pattern, group, component, kind in COMPILED_RULES:
        if pattern.search(name):
            return group, component, kind
    if section.startswith(".text"):
        return "platform", "library_code", "code"
    return "unclassified", "unclassified", "unknown"


def classify_section(section: str) -> tuple[str, str] | None:
    """Returns the (group, component) for a symbol-less section, or None."""
    for pattern, group, component in COMPILED_SECTION_RULES:
        if pattern.search(section):
            return group, component
    return None


def layer_of(symbol: str) -> str:
    """Returns the layer index encoded in a symbol name, or an empty string."""
    match = re.match(r"^(?:L|blocks_)(\d+)_", normalize(symbol))
    return match.group(1) if match else ""


def detect_quant_mode(rows: list[SymbolRow]) -> str:
    """Infers the quantization emission path from tensor naming."""
    for row in rows:
        if row.symbol.startswith("blocks_"):
            return QUANT_MODE_NAMES["blocks"]
        if row.symbol.startswith(("L0_", "L1_")):
            return QUANT_MODE_NAMES["L"]
    return QUANT_MODE_NAMES["unknown"]


def split_cost(flags: int, sh_type: str, size: int) -> tuple[int, int]:
    """Returns the (flash, RAM) cost of an allocated section or symbol.

    A section occupies flash when it is allocated and carries contents in the
    file. It occupies RAM when it is allocated and writable. Initialized data is
    both, because its startup image is stored in flash and copied to RAM.
    """
    if not flags & SH_FLAGS.SHF_ALLOC:
        return 0, 0
    flash = size if sh_type == "SHT_PROGBITS" else 0
    ram = size if flags & SH_FLAGS.SHF_WRITE else 0
    return flash, ram


def read_elf(elf_path: Path, case: str, model: str, setup: str,
             min_bytes: int) -> tuple[list[SymbolRow], dict, int, int]:
    """Extracts symbol rows and section totals from one ELF file.

    Returns the symbol rows, a mapping of section name to (flash, RAM, type,
    flags, size), and the whole-image flash and RAM totals.
    """
    rows: list[SymbolRow] = []
    sections: dict[str, tuple] = {}
    total_flash = 0
    total_ram = 0

    with elf_path.open("rb") as handle:
        elf = ELFFile(handle)

        for section in elf.iter_sections():
            flags = section["sh_flags"]
            sh_type = section["sh_type"]
            size = section["sh_size"]
            if not flags & SH_FLAGS.SHF_ALLOC:
                continue
            flash, ram = split_cost(flags, sh_type, size)
            sections[section.name] = (flash, ram, sh_type, flags, size)
            total_flash += flash
            total_ram += ram

        symtab = elf.get_section_by_name(".symtab")
        if symtab is None:
            raise RuntimeError(f"{elf_path} has no symbol table; was it stripped?")

        for symbol in symtab.iter_symbols():
            size = symbol["st_size"]
            if size < max(1, min_bytes):
                continue
            if symbol["st_info"]["type"] not in ("STT_OBJECT", "STT_FUNC"):
                continue

            shndx = symbol["st_shndx"]
            if isinstance(shndx, str):
                continue  # SHN_UNDEF, SHN_ABS and friends have no section.

            section = elf.get_section(shndx)
            flash, ram = split_cost(section["sh_flags"], section["sh_type"], size)
            if flash == 0 and ram == 0:
                continue

            group, component, kind = classify(symbol.name, section.name)
            address = symbol["st_value"]
            rows.append(SymbolRow(
                case=case, model=model, setup=setup, symbol=symbol.name,
                group=group, component=component, kind=kind,
                layer=layer_of(symbol.name), section=section.name,
                region=region_for(address), address=address, bytes=size,
                flash_bytes=flash, ram_bytes=ram,
            ))

    return rows, sections, total_flash, total_ram


def residual_rows(rows: list[SymbolRow], sections: dict, case: str, model: str,
                  setup: str) -> list[dict]:
    """Attributes section bytes that no symbol accounts for.

    Alignment padding, symbol-less sections, and the heap and stack reservation
    all land here. Section rules absorb the predictable ones; whatever is left is
    reported as unattributed rather than silently distributed.
    """
    claimed_flash: dict[str, int] = defaultdict(int)
    claimed_ram: dict[str, int] = defaultdict(int)
    for row in rows:
        claimed_flash[row.section] += row.flash_bytes
        claimed_ram[row.section] += row.ram_bytes

    out = []
    for name, (flash, ram, _sh_type, _flags, _size) in sections.items():
        gap_flash = flash - claimed_flash.get(name, 0)
        gap_ram = ram - claimed_ram.get(name, 0)
        if gap_flash <= 0 and gap_ram <= 0:
            continue

        matched = classify_section(name)
        group, component = matched if matched else ("unclassified", "unattributed")
        out.append({
            "case": case, "model": model, "setup": setup, "group": group,
            "component": component, "kind": "residual", "symbol_count": 0,
            "flash_bytes": max(0, gap_flash), "ram_bytes": max(0, gap_ram),
        })
    return out


def read_stack_usage(artifact_dir: Path) -> list[dict]:
    """Reads the per-setup stack_usage.csv archived by the firmware sweep.

    Returns an empty list when the file is absent, which is expected for setups
    swept before -fstack-usage was enabled.
    """
    path = artifact_dir / "stack_usage.csv"
    if not path.is_file():
        return []

    # PowerShell's Export-Csv writes a BOM, so utf-8-sig rather than utf-8.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = []
        for record in csv.DictReader(handle):
            try:
                record["frame_bytes"] = int(record["frame_bytes"])
            except (KeyError, ValueError):
                continue
            rows.append(record)
    return rows


def summarize_stack(rows: list[dict], entry: str | None) -> dict:
    """Summarizes stack frames into a lower and upper bound.

    The lower bound is the largest single frame: worst-case stack cannot be less
    than the deepest function's own frame. The upper bound sums every frame,
    which assumes one chain through all functions and so is loose. The true
    figure is the sum along the deepest call path, which .su files cannot supply
    because they carry no call graph.

    Only the 'dynamic' qualifier is unbounded. GCC also emits 'bounded', which
    varies but has a compile-time limit.
    """
    if not rows:
        return {
            "stack_frames": 0, "stack_max_bytes": "", "stack_max_function": "",
            "stack_sum_bytes": "", "stack_entry_bytes": "",
            "stack_bounded_frames": "", "stack_dynamic_frames": "",
        }

    largest = max(rows, key=lambda r: r["frame_bytes"])
    entry_bytes = ""
    if entry:
        matches = [r for r in rows if r.get("function", "").startswith(entry)]
        if matches:
            entry_bytes = max(r["frame_bytes"] for r in matches)

    return {
        "stack_frames": len(rows),
        "stack_max_bytes": largest["frame_bytes"],
        "stack_max_function": largest.get("function", ""),
        "stack_sum_bytes": sum(r["frame_bytes"] for r in rows),
        "stack_entry_bytes": entry_bytes,
        "stack_bounded_frames": sum(
            1 for r in rows if r.get("qualifier", "static") == "bounded"
        ),
        "stack_dynamic_frames": sum(
            1 for r in rows if r.get("qualifier", "static") == "dynamic"
        ),
    }


def dump_struct(elf_path: Path, struct_name: str) -> None:
    """Prints the field layout of a struct, read from DWARF debug info.

    Field sizes are derived from consecutive member offsets, which avoids
    resolving the full type graph and handles arrays and typedefs uniformly.
    """
    with elf_path.open("rb") as handle:
        elf = ELFFile(handle)
        if not elf.has_dwarf_info():
            print("No DWARF info; the build was stripped.", file=sys.stderr)
            return

        for unit in elf.get_dwarf_info().iter_CUs():
            for die in unit.iter_DIEs():
                if die.tag != "DW_TAG_structure_type":
                    continue
                name = die.attributes.get("DW_AT_name")
                if name is None or name.value.decode() != struct_name:
                    continue

                total = die.attributes.get("DW_AT_byte_size")
                total = total.value if total else 0
                members = []
                for child in die.iter_children():
                    if child.tag != "DW_TAG_member":
                        continue
                    offset = child.attributes.get("DW_AT_data_member_location")
                    field_name = child.attributes.get("DW_AT_name")
                    if offset is None or field_name is None:
                        continue
                    members.append((offset.value, field_name.value.decode()))

                if not members:
                    continue

                members.sort()
                print(f"\n{struct_name}: {total} bytes")
                print(f"{'offset':>8}  {'bytes':>8}  field")
                for index, (offset, field_name) in enumerate(members):
                    end = members[index + 1][0] if index + 1 < len(members) else total
                    print(f"{offset:>8}  {end - offset:>8}  {field_name}")
                return

    print(f"Struct '{struct_name}' not found in DWARF info.", file=sys.stderr)


def find_targets(deploy_root: Path, board: str, case_glob: str, model_glob: str,
                 setup_glob: str) -> list[tuple[str, str, str, Path]]:
    """Locates archived ELF files matching the requested filters.

    Reports the deepest level reached when nothing matches, so a missing artifact
    directory is distinguishable from a wrong deploy root or a bad filter.
    """
    if not deploy_root.is_dir():
        raise SystemExit(f"Deploy root does not exist: {deploy_root}")

    targets = []
    reached = {"case": 0, "model": 0, "setups_dir": 0, "setup": 0,
               "artifact_dir": 0, "elf": 0}

    for case_dir in sorted(deploy_root.glob(case_glob)):
        if not case_dir.is_dir():
            continue
        reached["case"] += 1

        for model_dir in sorted(case_dir.glob(model_glob)):
            if not model_dir.is_dir():
                continue
            reached["model"] += 1

            setups_dir = model_dir / "setups"
            if not setups_dir.is_dir():
                continue
            reached["setups_dir"] += 1

            for setup_dir in sorted(setups_dir.glob(setup_glob)):
                if not setup_dir.is_dir():
                    continue
                reached["setup"] += 1

                artifact_dir = setup_dir / "online" / board
                if not artifact_dir.is_dir():
                    continue
                reached["artifact_dir"] += 1

                elves = sorted(artifact_dir.glob("*.elf"))
                if not elves:
                    continue
                reached["elf"] += 1
                targets.append(
                    (case_dir.name, model_dir.name, setup_dir.name, elves[0])
                )

    if not targets:
        print(f"No ELF files found under {deploy_root}", file=sys.stderr)
        print("Directories matched at each level:", file=sys.stderr)
        for level, count in reached.items():
            print(f"  {level:<14} {count}", file=sys.stderr)
        if reached["setup"] and not reached["artifact_dir"]:
            print(f"\nSetups exist but none contain online/{board}. "
                  "Run the firmware sweep first.", file=sys.stderr)
        elif reached["artifact_dir"] and not reached["elf"]:
            print("\nArtifact directories exist but contain no .elf. "
                  "Check build.log — the builds probably failed.", file=sys.stderr)

    return targets


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Writes rows to a CSV file with a fixed column order.

    Fields absent from the column list are dropped, so a new field must be added
    here as well as at its source.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deploy-root",
        type=Path,
        default=Path.home() / "projects/ssm-mcu-asd-ind/mcu/deploy",
    )
    parser.add_argument("--board", default="stm32-nucleo-h7s3l8")
    parser.add_argument("--case", default="case*")
    parser.add_argument("--model", default="*")
    parser.add_argument("--setup", default="*")
    parser.add_argument("--min-bytes", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Defaults to <deploy-root>/../eval/online/<board>.")
    parser.add_argument("--stack-entry", default="SSMBackbone_ProcessFrame",
                        help="Function whose frame is reported separately.")
    parser.add_argument("--struct", default=None,
                        help="Print a struct's field layout from DWARF and exit.")
    parser.add_argument("--list-unclassified", action="store_true",
                        help="Print every unmatched symbol, largest first.")
    args = parser.parse_args()

    targets = find_targets(args.deploy_root, args.board, args.case, args.model,
                           args.setup)
    if not targets:
        return 1

    if args.struct:
        dump_struct(targets[0][3], args.struct)
        return 0

    out_dir = args.out_dir or (
        args.deploy_root.parent / "eval" / "online" / args.board
    )

    symbol_rows: list[dict] = []
    component_rows: list[dict] = []
    highlevel_rows: list[dict] = []
    stack_rows: list[dict] = []
    unclassified: list[SymbolRow] = []

    for case, model, setup, elf_path in targets:
        print(f"{case}/{model}/{setup}")

        rows, sections, total_flash, total_ram = read_elf(
            elf_path, case, model, setup, args.min_bytes
        )
        unclassified.extend(r for r in rows if r.group == "unclassified")

        quant_mode = detect_quant_mode(rows)
        if quant_mode == "unknown":
            print("  WARNING: no backbone tensors found; quant mode unknown")
        tag = {"architecture": ARCHITECTURE, "quant_mode": quant_mode}

        for row in rows:
            symbol_rows.append({"board": args.board, **tag, **row.__dict__})

        grouped: dict[tuple, dict] = defaultdict(
            lambda: {"symbol_count": 0, "flash_bytes": 0, "ram_bytes": 0}
        )
        for row in rows:
            entry = grouped[(row.group, row.component, row.kind)]
            entry["symbol_count"] += 1
            entry["flash_bytes"] += row.flash_bytes
            entry["ram_bytes"] += row.ram_bytes

        setup_components = [
            {"board": args.board, **tag, "case": case, "model": model,
             "setup": setup, "group": group, "component": component,
             "kind": kind, **values}
            for (group, component, kind), values in grouped.items()
        ]
        for residual in residual_rows(rows, sections, case, model, setup):
            setup_components.append({"board": args.board, **tag, **residual})
        component_rows.extend(setup_components)

        by_group: dict[str, dict] = defaultdict(
            lambda: {"flash_bytes": 0, "ram_bytes": 0}
        )
        for entry in setup_components:
            by_group[entry["group"]]["flash_bytes"] += entry["flash_bytes"]
            by_group[entry["group"]]["ram_bytes"] += entry["ram_bytes"]

        for group in GROUP_ORDER:
            if group not in by_group:
                continue
            values = by_group[group]
            highlevel_rows.append({
                "board": args.board, **tag, "case": case, "model": model,
                "setup": setup, "group": group,
                "flash_bytes": values["flash_bytes"],
                "ram_bytes": values["ram_bytes"],
                "flash_pct": round(100 * values["flash_bytes"] / total_flash, 2)
                if total_flash else 0,
                "ram_pct": round(100 * values["ram_bytes"] / total_ram, 2)
                if total_ram else 0,
            })

        stack = summarize_stack(
            read_stack_usage(elf_path.parent), args.stack_entry
        )
        stack_rows.append({
            "board": args.board, **tag, "case": case, "model": model,
            "setup": setup, "static_ram_bytes": total_ram, **stack,
            "peak_ram_low": total_ram + (stack["stack_max_bytes"] or 0),
            "peak_ram_high": total_ram + (stack["stack_sum_bytes"] or 0),
        })

        by_region: dict[str, int] = defaultdict(int)
        for row in rows:
            if row.ram_bytes:
                by_region[row.region] += row.ram_bytes
        regions = ", ".join(f"{k} {v}" for k, v in sorted(by_region.items()))

        if stack["stack_frames"]:
            print(f"  {quant_mode}  flash {total_flash}  static ram {total_ram}"
                  f"  peak ram {total_ram + stack['stack_max_bytes']}"
                  f"-{total_ram + stack['stack_sum_bytes']}  [{regions}]")
        else:
            print(f"  {quant_mode}  flash {total_flash}  ram {total_ram}"
                  f"  [{regions}]")

    suffix = args.board
    tag_columns = ["board", "architecture", "quant_mode"]

    write_csv(out_dir / f"footprint_symbols_{suffix}.csv", symbol_rows,
              tag_columns + [
                  "case", "model", "setup", "symbol", "group", "component",
                  "kind", "layer", "section", "region", "address", "bytes",
                  "flash_bytes", "ram_bytes",
              ])
    write_csv(out_dir / f"footprint_components_{suffix}.csv", component_rows,
              tag_columns + [
                  "case", "model", "setup", "group", "component", "kind",
                  "symbol_count", "flash_bytes", "ram_bytes",
              ])
    write_csv(out_dir / f"footprint_highlevel_{suffix}.csv", highlevel_rows,
              tag_columns + [
                  "case", "model", "setup", "group", "flash_bytes", "ram_bytes",
                  "flash_pct", "ram_pct",
              ])

    if any(r["stack_frames"] for r in stack_rows):
        write_csv(out_dir / f"footprint_stack_{suffix}.csv", stack_rows,
                  tag_columns + [
                      "case", "model", "setup", "static_ram_bytes",
                      "stack_frames", "stack_max_bytes", "stack_max_function",
                      "stack_sum_bytes", "stack_entry_bytes",
                      "stack_bounded_frames", "stack_dynamic_frames",
                      "peak_ram_low", "peak_ram_high",
                  ])
        print(f"\nWrote four CSV files to {out_dir}")
    else:
        print(f"\nWrote three CSV files to {out_dir}")
        print("No stack data found. Re-run the sweep with the .su harvester.")

    if unclassified:
        print(f"\n{len(unclassified)} unclassified symbol(s):")
        shown = sorted(unclassified, key=lambda r: r.bytes, reverse=True)
        for row in shown[: None if args.list_unclassified else 15]:
            print(f"  {row.bytes:>8}  {row.section:<20} {row.symbol}")
        if not args.list_unclassified and len(shown) > 15:
            print(f"  ... {len(shown) - 15} more; use --list-unclassified")
        print("\nAdd rules to CLASSIFICATION_RULES until this list is empty.")

    return 0


if __name__ == "__main__":
    sys.exit(main())