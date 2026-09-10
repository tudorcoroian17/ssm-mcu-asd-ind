# All four weight-only int8 schemes -- parity and footprint on the Nucleo-H7S3L8

**Feeds:** `06_phase5_nucleo_deployment_matrix.md` (items 1-3, all four
weight-mode x granularity combinations); `findings/530` (the design these
results validate); `findings/520`/`521` (the Python predictions confirmed
here on real hardware).

## Result

All four weight-only int8 schemes -- `projections`/`all` crossed with
`per-tensor`/`per-channel` -- are parity-validated on the Nucleo-H7S3L8 and
measured for footprint. Every scheme's on-device output matches its own
quantized Python reference within the same reordering-noise floor the fp32
baseline showed; footprint forms a complete, mechanistically-explained 2x2
grid. This is the accuracy-and-footprint-per-rung table
`06_phase5_nucleo_deployment_matrix.md` set out to produce.

Scope: `f4cd557b7e3b`, case 1, run `16662b29beb3`. One fold.

## Parity

Compared against each scheme's own quantized reference (not the fp32
reference -- see the note below), generated via
`checks/smoke/quant/weight_quant_parity.py --streaming --dump-reference-dir`,
matching the C port's per-timestep evaluation order:

| Scheme | max abs error | mean abs error |
| :--- | ---: | ---: |
| fp32 baseline (`findings/430`) | 5.99e-4 | 1.45e-4 |
| `proj_pertensor` | 6.07e-4 | 1.49e-4 |
| `all_pertensor` | 6.11e-4 | 1.46e-4 |
| `proj_perchannel` | 5.95e-4 | 1.48e-4 |
| `all_perchannel` | 5.99e-4 | 1.47e-4 |

All four sit within the same narrow band as the fp32 figure -- confirming
the per-access design (`findings/530`) reproduces each scheme's
Python-validated quantization behavior exactly, with only the same
C-vs-Python floating-point reordering noise on top that every prior parity
check in this project has shown.

**Comparison note, worth restating plainly**: a quantized scheme's device
output should be compared against that scheme's own quantized Python
reference, not the fp32 reference. The fp32-vs-quantized gap
(`findings/520`: ~2.2e-2 mean) measures something different -- how much the
scheme perturbs the model -- and is not a pass/fail bar for whether the C
port is correct. Conflating the two produced a lengthy false-alarm
investigation during this work (`findings/530`).

## Footprint

| Scheme | Flash | Δ vs fp32 (262,660 B) | RAM |
| :--- | ---: | ---: | ---: |
| fp32 baseline | 262,660 B | -- | ~16,728 B |
| `proj_pertensor` | 176,372 B | -32.8% | 17,868 B |
| `all_pertensor` | 169,284 B | -35.6% | 17,868 B |
| `proj_perchannel` | 179,236 B | -31.8% | 17,868 B |
| `all_perchannel` | 172,660 B | -34.3% | 17,868 B |

Every relationship in this table has an identified mechanism, not just a
measured direction:

- **`all` beats `projections` at matched granularity** (e.g. `all_pertensor`
  < `proj_pertensor`): more tensors are quantized, more flash is freed.
- **`per-tensor` beats `per-channel` at matched weight-mode** (e.g.
  `proj_pertensor` < `proj_perchannel`): per-channel stores one fp32 scale
  per output row instead of one scale total, and that overhead shows up
  directly in flash.
- **RAM is identical (17,868 B) across all four schemes.** Under the
  per-access design, every weight -- quantized or not -- lives in flash;
  RAM usage depends only on things unrelated to quantization scheme (buffer
  sizes, stack, struct layout), never on which tensors were quantized or at
  what granularity. This is a structural property of the design
  (`findings/530`), confirmed directly by measurement here across four
  independent builds.

## Consequence

This is validation and a complete comparison table, not a change to the
deployment recommendation. `findings/521`'s conclusion stands:
`projections`/`per-tensor` remains the scheme to actually ship -- cheapest,
simplest, and most conservative on the recurrence parameters
`findings/150` flagged as sensitive. The other three schemes now exist as
measured comparison points for the paper's results table, not as
competing recommendations.

## Open items

- **One fold.** As with every quantization finding in this project, case-1
  only; the cross-machine picture needs the other three folds before a
  final accuracy table.
- **Weight+activation schemes (items 4-6) not yet started.** This finding
  closes out the weight-only half of the deployment matrix; the activation
  side has its own, larger C effort ahead (`06_phase5_nucleo_deployment_matrix.md`
  Phase 3-5).

<!-- Claude comment: the RAM-flatness result is the one I'd highlight
first if this table needs a single headline sentence -- "flash cost scales
with how aggressively you quantize; RAM cost doesn't, on this board, under
this design" is a clean, general, and slightly non-obvious statement that
required cross-checking to actually believe rather than assume from the
structural argument alone. -->