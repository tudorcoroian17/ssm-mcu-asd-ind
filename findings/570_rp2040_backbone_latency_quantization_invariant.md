# RP2040 backbone latency is nearly invariant to quantization scheme

~~Status: open — mechanism is code-supported but not yet confirmed by direct
sub-stage measurement. See "Further investigation" at the end.~~
Status: proven as **FALSIFIED**. Every one of the 100 sweep setups flashed the 
same binary, so the timing, score and embedding columns measure one build.
The upload step doesn't flash the sweep's build. upload.log for true_int8_h16_euclidean_fp32 
shows the flashed file was `...\AppData\Local\arduino\sketches\E712AAE8...\arduino-nano-rp2040-connect-ssm-mamba-asd.ino.elf.` 
That is arduino-cli's default cache. The sweep compiles into .cycle-sweep-rp2040\build 
with --build-path, but run_upload() passes no --input-dir. The arduino-cli docs say upload 
doesn't compile and only flashes what it finds, so it flashed whatever ELF was last compiled 
into the default cache.

## Observation

Across the RP2040's full 100-setup sweep (`test_harness/output_rp2040/
cycle_sweep_rp2040_summary.csv`), `mean_backbone_us` is nearly constant
regardless of quantization scheme:

- Every setup in `case1/16662b29beb3` (and every other case/model
  combination) falls between roughly 25,610,400 and 25,611,236 µs total
  backbone time across 5 timing clips — a spread of about 0.003%.
- `true_int8_h8_euclidean_fp32` (25,610,842.8 µs) and
  `true_int8_h16_euclidean_fp32` (25,610,738.0 µs) differ by 105 µs out of
  25.6 million.
- `full_fp32_euclidean_fp32` (25,610,604.0 µs) is statistically
  indistinguishable from the `true_int8` setups above.

This is on the Nucleo's opposite: on that board, `true_int8` setups measure
meaningfully faster backbone time than `full_fp32` (see
`440_flash_usage_and_ram_cycles_debug.md` and related). The RP2040 not
showing this gap at all is the finding.

## Investigation timeline

### Ruled out: sweep environment vs. manual test

`full_fp32_euclidean_fp32` was first measured with a one-off manual
round-trip test, before GPIO power markers and `NINA_RESETN` were added to
the firmware: `backbone=42,817,694 us` total. The same setup, measured
again after those firmware additions — once inside the full 100-setup
sweep (`25,610,604 us`) and once via a fresh standalone reflash-and-test
outside the sweep (`25,602,334 us`) — landed within 0.03% of each other,
nowhere near the original reading.

Three independent runs now agree; only the first one doesn't. Treated as a
one-off anomaly from that specific cold run, not a real property of either
firmware version. The 42.8M-µs reading is preserved here as a data point,
not deleted, since re-observing it would be worth reopening this.

### Falsified: per-access dequantize-to-float in the true-int8 path

First hypothesis: the `true_int8` backbone dequantizes every weight to
`float32` before use, making its actual arithmetic no different from the
`full_fp32` path — same soft-float cost, different storage format.

Checked directly against `ssm_backbone.c` for
`true_int8_h8_euclidean_int8`: false. The file's own header comment states
plainly that this is not the dequantize-to-float pattern, and the
implementation backs it up — `ssm_qlinear_real()` accumulates genuine
`int32` sums of `int8 × int8` products for every matvec (`in_proj`,
`x_proj`, `dt_proj`, `out_proj`) and the depthwise conv, with exactly one
float rescale per output element at the end of each. That part of "true
int8" is real integer arithmetic, not simulated.

## Current best-supported explanation

Read `ssm_backbone.c` for both `full_fp32_euclidean_fp32` (the `fake_quant`
family) and `true_int8_h8_euclidean_int8` side by side. Two separate
mechanisms are responsible, for two separate groups of setups.

### The `fake_quant` family (full_fp32 and every `w_*` weight-quantized setup)

`quant_mode` in the footprint CSVs groups `full_fp32_euclidean_fp32`
alongside every `w_all_*`/`w_proj_*` setup under `fake_quant`. All of them
compile from the same `ssm_backbone.c` structure, parameterized by
`SSM_IN_PROJ_W`/`SSM_A`/etc. macros from `ssm_weights.h`. Weight-only
quantization in this family means: quantize at export time, dequantize
back to `float32` once the macro resolves, then run the identical
float-arithmetic code path as the unquantized baseline. Instruction count
per hop does not change — only the specific float values read from flash
differ. Near-identical timing across this entire family is expected on
that basis alone and isn't itself surprising.

The `_actboundaries` variants add real activation quantization at four
per-layer boundary points (`z`, `u`, `y_gated`, `block_out` — via
`SSM_QUANT_Z`/`_U`/`_Y_GATED`/`_BLOCK_OUT`), each applying one
`ssm_quant_dequant()` call per element, so `SSM_D_INNER` (64) or
`SSM_D_MODEL` (64) times per layer. That's on the order of 250 extra
rescale calls per layer — small next to the recurrence loop's iteration
count below, and consistent with `_actboundaries` setups showing no
measurable difference either.

### `true_int8` vs. `full_fp32`

This is the pairing where the two schemes run genuinely different code,
and where the near-equal total time needs an actual explanation rather
than "same instructions, different data."

The innermost recurrence loop — per channel, per state, `SSM_D_INNER (64)
× SSM_D_STATE (16) = 1,024` iterations per layer per hop — differs
structurally between the two:

- **`full_fp32`**: computes `deltaA`, `A_bar`, `B_bar`, `new_h`, and the
  `y_c` accumulation directly in float. One compare (the `-1.9` clamp). No
  divide, no `roundf()`, anywhere in this loop.
- **`true_int8`**: computes the same conceptual quantities, but `A_bar`,
  `B_bar`, and `new_h` are each passed through `roundf(real / scale)` plus
  a clamp before the next step can consume them — three separate
  divide-and-`roundf()` pairs, every iteration, that `full_fp32` never
  does. This exists purely to keep intermediates representable in the
  fixed range the next `int32` accumulation needs, not to compute anything
  additional.

Per layer, per hop: `1,024 × 3 = 3,072` extra divide+`roundf()` pairs for
`true_int8` that `full_fp32` doesn't pay. Across both layers, `6,144`.

Claude comment: software-emulated divide is typically the most expensive
basic float operation on a core with no hardware support for it, and
`roundf()` is a full function call on top of that. My reading is that the
two effects are opposed on this specific chip: `true_int8` genuinely wins
in the matvecs (thousands of multiply-accumulates become native integer
instead of soft-float), and genuinely loses in the recurrence (thousands
of extra divide-and-round calls the float path never needed). On a core
with a hardware FPU — the Nucleo — the recurrence's extra rescaling barely
registers next to hardware-speed float, so the matvec win shows through
cleanly, matching the Nucleo's own numbers. On the RP2040, where every
float operation costs roughly the same regardless of why it's there, the
two effects land close enough to cancel — which is what the sweep data
shows. This is a mechanistic account built from reading both source files
and counting operations by hand, not from isolated measurement of either
loop — see below for what would actually confirm it.

## Further investigation

- **Not yet measured directly**: everything above is an instruction-count
  argument from reading two `.c` files, not a measurement that isolates
  matvec time from recurrence time. Confirming it needs either finer GPIO
  markers around just the recurrence loop specifically (separate from the
  existing single `BACKBONE` marker spanning all of
  `SSMBackbone_ProcessFrame`), or comparing against a version of one
  backbone with the recurrence loop's rescaling temporarily stubbed out.
- **Not yet checked**: whether this near-cancellation is a coincidence of
  this specific model's dimensions (`d_inner=64`, `d_state=16`) or would
  generalize. A model with a larger `d_state` relative to `d_model` would
  weight the recurrence's cost (and therefore `true_int8`'s extra
  rescaling) more heavily; a model with a larger `d_model` would weight
  the matvecs (and therefore `true_int8`'s savings) more heavily. Nothing
  here establishes which side would win for a different configuration.
- **Not yet checked**: whether the `_actboundaries` fake-quant setups'
  small number of extra boundary rescales (estimated above, not measured)
  are actually invisible in the data or just small — worth confirming
  against the raw per-setup numbers rather than assumed.
- **Unexplained, low priority**: the original 42.8M-µs `full_fp32` reading
  has no mechanism identified. Left as a one-off unless it recurs.