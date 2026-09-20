# RP2040 microbenchmark and per-hop cost model

Describes how the RP2040 microbenchmark measures the cost of each operation
in the backbone, how to turn those costs into a per-hop time estimate, and how
well the estimate matches the sweep. Read this before you quote any RP2040
cycle count per operation. Finding 570 records the harness bug that made the
first RP2040 sweep invalid. This finding starts from the corrected sweep.

## Summary

- The RP2040 has no hardware FPU. Every float operation is a library call.
  Costs range from 28 cycles (`f2i`) to 4,711 cycles (`log1pf`) on the Mbed
  core.
- On the Mbed core (GCC 7.2.1, libgcc float), a float MAC costs 285 to 348
  cycles. An int8 MAC costs 14 to 27 cycles. The int8 MAC is 13 to 20 times
  cheaper.
- The int8 backbone is not dominated by integer MACs. The scan loop takes 76%
  of the modeled int8 hop. The integer MACs take 8%.
- One `ssm_requantize` call costs 656 cycles on the Mbed core. The float
  divide is 69% of that cost.
- The `arduino-pico` core links the RP2040 bootrom float routines. The
  linker map shows `__wrap___aeabi_*` symbols from `libpico.a`. On the same
  125 MHz clock, `fdiv` is 5.1 times faster, `expf` is 6.4 times faster, and
  `scan_int8_iter` is 2.2 times faster.
- The size of the operands changes the cost of a soft-float operation by 3 to
  13 cycles on the Mbed core. `fadd` is the most sensitive (up to 12%).
- With realistic h16 states, the scan kernel is 0.9% to 1.0% slower than with
  h8 on both cores. The sweep shows 3% on the deployed model. The
  microbenchmark explains about a quarter of the sweep penalty on the Mbed
  core.
- Some Mbed composite rows (`silu`, `softplus`) change by up to 19% between
  builds of the microbenchmark. The `arduino-pico` rows do not change. A
  difference in the code layout in flash is a possible cause. It is not
  tested.
- The cost model predicts the `full_fp32` backbone within +0.9% to −1.4% of
  the sweep, depending on the microbenchmark build. It predicts
  `true_int8_h8` within 0.0% to 4.1%.

## What the microbenchmark measures

The microbenchmark answers one question: how many cycles does each basic
operation cost on this core? It measures each operation in isolation, then
measures the larger kernels that the backbone actually runs. You combine the
two with an operation count to estimate a full hop (see "How to compute the
number of operations").

The sketch is in `ssm-mcu-asd-deploy/rp2040-microbench/` and has three files:

| File | Role |
| --- | --- |
| `bench_kernels.h` | Declares the kernels. Sets `BENCH_H_BITS` (8 or 16), which selects the width of the recurrence state `h`. It also sets `BENCH_H_SCALE_DIV` (1 for h8, 258 for h16). |
| `bench_kernels.c` | Holds every kernel. It is compiled as C, like `ssm_backbone.c`. |
| `rp2040-microbench.ino` | Runs each case, times it, and prints one CSV line per case over USB `Serial`. |

The function bodies for `ssm_softplus`, `ssm_silu`, `ssm_quant_dequant`,
`ssm_requantize`, and `ssm_rmsnorm` are copies of the deployed functions. The
two scan kernels copy the inner loops of `ssm_block_step` (fp32) and
`ssm_block_step_true_int8`.

### Builds

The microbenchmark exists in two builds. The results in this finding come from
both.

| Build | Content |
| --- | --- |
| A | The first version. It has no magnitude cases. Both h widths use the same `s_h`. The cost model and the worked examples use build A. |
| B | Adds the magnitude cases (`_m7`, `_m15`, `_m22`), scales `s_h` by 1/258 for h16, and prints a state report. |

## How the measurement works

### Timer

The sketch reads `TIMERAWL` (address `0x40054000 + 0x28`). This register
counts microseconds. It is the same register the deployed firmware uses for
its stage timers. The RP2040 (Cortex-M0+) has no DWT cycle counter, so the
sketch cannot read a true cycle count. Cycles are derived:

$$\text{cycles} = t_{\mu s} \cdot f_{\text{MHz}}$$

The resolution is 1 µs, which is 125 cycles at 125 MHz. Every timed run lasts
at least 50 ms, so the quantization error is below 0.01%.

### Clock

The sketch measures the core clock instead of assuming it. The function
`bench_spin(n)` runs exactly $3n$ cycles: `subs` takes 1 cycle and the taken
`bne` takes 2 cycles. The sketch runs it with $n = 3{,}000{,}000$ five times
and keeps the shortest time:

$$f_{\text{MHz}} = \frac{3n}{t_{\text{spin},\mu s}}$$

| Build | Nominal clock | Measured clock |
| --- | --- | --- |
| Mbed core (`arduino:mbed_nano` 4.4.1) | 125 MHz | 124.79 MHz |
| `arduino-pico` core 6.1.0, 125 MHz | 125 MHz | 124.42 MHz (124.43 MHz in build B) |
| `arduino-pico` core 6.1.0, 200 MHz | 200 MHz | 199.42 MHz |

All measured values are 0.2% to 0.5% below nominal. The cause is not
confirmed. Interrupts (the microbenchmark does not disable them) are one
candidate. Treat every cycle count as accurate to about ±0.5%.

### Timing loop

Each case runs through the same procedure:

1. **Choose the iteration count.** Start at $n = 1$. Double $n$ until one run
   lasts at least 50,000 µs. These probe runs also warm the caches.
2. **Time the kernel.** Run the kernel three times with the chosen $n$. Keep
   the shortest time. The shortest time removes the effect of an interrupt
   that lands inside one run.
3. **Time the baseline.** For scalar kernels, run `bench_base` with the same
   $n$ and keep the shortest of three runs.
4. **Compute the cost.** With $m$ operations counted per iteration:

$$c_{\text{cycles/op}} = \frac{(t_{\text{kernel}} - t_{\text{base}}) \cdot f_{\text{MHz}}}{n \cdot m}$$

The sketch prints one line per case:

```text
RESULT,name,iters,ops_per_iter,total_us,base_us,ns_per_op,cycles_per_op
```

### Scalar kernels

A scalar kernel measures one operation. All scalar kernels have the same
skeleton. Each iteration loads five values from `volatile` arrays, runs one
operation, and stores the result to a `volatile` variable.

The `volatile` qualifier stops the compiler from folding the operation,
moving it out of the loop, or removing it. The baseline kernel `bench_base`
has the same loads and the same store, but no operation. The subtraction in
step 4 removes the cost of the loads, the store, and the loop counter.

The operands come from 16-entry pools:

| Pool | Content |
| --- | --- |
| `bench_fs` | Signed floats, magnitude 0.25 to 4 |
| `bench_fp` | Positive floats, 0.25 to 4 |
| `bench_is` | Signed int32 values, about ±262,144 (±2¹⁸) |
| `bench_scale` | The float 0.037 |

A library call such as `__aeabi_fadd` is part of the cost. The deployed code
pays the same call cost.

### Magnitude kernels (build B)

The magnitude kernels measure whether the cost of an operation depends on the
size of its operands. They have the same skeleton as the scalar kernels, so
`bench_base` is their baseline. Only the operands come from a different pool.
Each pool holds values with random signs and magnitudes in one range:

| Class | Magnitude range | What it stands for |
| --- | --- | --- |
| `_m7` | [2⁶, 2⁷) | An int8 state, or a rounded int8 value |
| `_m15` | [2¹⁴, 2¹⁵) | An int16 state, or a product of two int8 values |
| `_m22` | [2²¹, 2²²) | The product of an int8 value and an int16 state (127 × 32,767 is below 2²²) |

The integer pools use `lo + (random % lo)` with $lo = 2^{k-1}$. The float
pools use $lo \cdot (1 + u)$ with $u$ in [0, 1), so the mantissa is full and
random. The cases are `i2f`, `fadd`, `f2i`, and `roundf`, each in the three
classes.

### Compound kernels

A compound kernel measures a function from the backbone. These kernels use the
same skeleton and the same baseline subtraction as the scalar kernels.

| Case | What it times | Counted as |
| --- | --- | --- |
| `silu` | `x / (1 + expf(-x))` | 1 call |
| `softplus` | `x > 20 ? x : log1pf(expf(x))` | 1 call |
| `requantize` | `roundf(real / scale)`, clamp to ±127, cast to `int8_t` | 1 call |
| `quant_dequant` | `requantize` followed by a multiply by the scale | 1 call |
| `rescale` | `(float) acc * a * b` (an int32 sum back to real units) | 1 call |
| `lut_ram` | One lookup in a 256-entry int8 table in RAM | 1 lookup |
| `lut_flash` | The same lookup, with the table in flash | 1 lookup |
| `rmsnorm_64` | The full RMSNorm on 64 elements | 1 call |

The `rmsnorm_64` case runs the whole function, so it has no baseline
subtraction. Its cost includes the loops and the memory access, which the
deployed code also pays.

### Matvec kernels

A matvec kernel runs the inner loop of `ssm_qlinear_real` (int8) or the
`in_proj` loop (fp32). One row is 64 MACs, and the sketch counts 64 operations
per iteration. The rows wrap with a mask, so the kernel reads a chosen amount
of weight data over and over.

The RP2040 runs code and reads constants from external flash through a 16 KB
XIP cache. The three placements of the weights test the cache:

| Suffix | Weights | Data read | Meaning |
| --- | --- | --- | --- |
| `_ram` | In RAM | 4 KB (int8), 16 KB (float) | No flash access |
| `_flash_warm` | In flash | 4 KB | Fits in the XIP cache |
| `_flash_cold` | In flash | 64 KB | Does not fit. Every pass streams from flash. |

The per-hop weight data of the deployed model does not fit in the cache. The
matvec and conv weights are 30,208 bytes as int8 and 120,832 bytes as float.
The cold rows are the upper bound and the warm rows are the lower bound.

The matvec rows have no baseline subtraction. The loop and the loads are part
of the cost, as they are in the deployed code.

### Scan kernels

`scan_fp32_iter` and `scan_int8_iter` run one layer's recurrence: 64 channels
with 16 states each, 1,024 iterations per repetition. The sketch counts 1,024
operations per repetition, so the result is the cost of one scan iteration.
The loop bodies copy the deployed code, including the `-1.9` clamp on
`deltaA`.

The int8 kernel uses the type `bench_h_t`, which is `int8_t` or `int16_t`
depending on `BENCH_H_BITS`. The scan kernels have no baseline subtraction.

In build A, both h widths used the same state scale `s_h`, so the h16 states
stayed small. In build B, `s_h = 0.02 / BENCH_H_SCALE_DIV`. For h16 the
divider is 258 (32,767 / 127), so the same real values become 258 times larger
integers.

**State report (build B).** After the last case, the sketch prints the largest
state magnitude that the int8 scan kernel reached and the number of states at
the clip value. Use this to check that the h16 test is valid. It depends on how
many repetitions ran, so it differs a little between cores.

| Build | `scan_int8_max_abs_h` | Clip | States at clip |
| --- | --- | --- | --- |
| h8 (both cores) | 91 | 127 | 0 of 1,024 |
| h16, Mbed | 22,320 | 32,767 | 0 of 1,024 |
| h16, `arduino-pico` | 23,881 | 32,767 | 0 of 1,024 |

### How to reproduce

1. Save `bench_kernels.h`, `bench_kernels.c`, and `rp2040-microbench.ino` in
   a folder named `rp2040-microbench`.
2. Build and flash. For the Mbed core, use a fixed build path and upload from
   the same path, so that the upload cannot flash a stale binary (see
   finding 570):

```text
   arduino-cli compile --fqbn arduino:mbed_nano:nanorp2040connect --build-path <build-dir> <sketch-dir>
   arduino-cli upload -p <port> --fqbn arduino:mbed_nano:nanorp2040connect --input-dir <build-dir> <sketch-dir>
```

   For the `arduino-pico` core, select the board and the CPU speed in the
   Arduino IDE.
3. Open a serial monitor on the USB port. One pass takes about 70 seconds.
4. To run the h16 variant, set `BENCH_H_BITS` to 16 in `bench_kernels.h` and
   repeat. The state scale changes with it.

The `INFO` lines report the optimization level, the compiler version, the
value of `BENCH_H_BITS`, the state scale divider, the measured clock, and (in
build B) the state report. Keep them with the results.

Raw results for build A are in
`ssm-mcu-asd-deploy/rp2040-microbench/results.txt` (Mbed) and `results-pico.txt`
(`arduino-pico`). Raw results for build B are in `results-magnitude.txt` (Mbed)
and `results-pico-magnitude.txt` (`arduino-pico`, 125 MHz).

## Results

Cycles per operation, build A. The h8 build is used, except where noted. The
Mbed build uses GCC 7.2.1 with `-Os` and the libgcc float library. The
`arduino-pico` builds use GCC 16.1.0 with `-Os`. Their map file shows the
bootrom float shims (`float_aeabi_rp2040.S`, `float_v1_rom_shim_rp2040.S`).
The `arduino-pico` rows used the board entry "Raspberry Pi Pico".

| Case | Mbed, 125 MHz | pico, 125 MHz | pico, 200 MHz |
| --- | --- | --- | --- |
| `fadd` | 112.69 | 84.82 | 84.81 |
| `fmul` | 163.31 | 71.26 | 71.25 |
| `fdiv` | 454.08 | 89.19 | 89.19 |
| `i2f` | 69.13 | 52.75 | 52.75 |
| `f2i` | 27.75 | 23.75 | 23.75 |
| `fcmp` | 70.81 | 50.44 | 50.44 |
| `roundf` | 32.75 | 28.75 | 28.75 |
| `sqrtf` | 453.65 | 80.00 | 80.00 |
| `expf` | 3,449.45 | 538.76 | 538.76 |
| `log1pf` | 4,710.96 | 1,094.15 | 1,094.12 |
| `silu` | 3,990.30 | 708.70 | 708.72 |
| `softplus` | 9,373.17 (see "Layout sensitivity") | 1,697.71 | 1,697.76 |
| `requantize` | 656.27 | 255.75 | 255.75 |
| `quant_dequant` | 780.89 | 295.76 | 295.76 |
| `rescale` | 392.32 | 187.56 | 187.57 |
| `lut_ram` | 9.00 | 9.00 | 9.00 |
| `lut_flash` | 9.00 | 9.00 | 9.00 |
| `rmsnorm_64` | 40,078.71 | 19,897.72 | 19,897.21 |
| `dot_i8_ram` (per MAC) | 14.27 | 13.27 | 13.27 |
| `dot_i8_flash_warm` | 14.29 | 13.32 | 13.30 |
| `dot_i8_flash_cold` | 26.91 | 26.24 | 26.25 |
| `dot_f32_ram` (per MAC) | 285.01 | 167.50 | 167.50 |
| `dot_f32_flash_warm` | 296.08 | 171.66 | 170.56 |
| `dot_f32_flash_cold` | 348.19 | 221.97 | 221.86 |
| `scan_fp32_iter` | 1,270.37 | 675.05 | 675.17 |
| `scan_int8_iter`, h8 | 3,616.96 | 1,629.24 | 1,629.25 |
| `scan_int8_iter`, h16, same `s_h` (storage type only) | 3,622.14 | 1,635.26 | 1,635.34 |

### Observations

- **Relative costs (Mbed).** `fdiv` costs 2.8 times `fmul` and 4.0 times
  `fadd`. `expf` costs 21 times `fmul`. `roundf` is cheap at 33 cycles.
  The divide, not the rounding, is the expensive part of `roundf(x / scale)`.
- **Integer against float MACs.** The float MAC is 20.0 times (RAM) to 12.9
  times (cold flash) more expensive than the int8 MAC on the Mbed core. On the
  `arduino-pico` core the ratio is 12.6 to 8.5.
- **Clock independence.** The `arduino-pico` cycle counts at 125 MHz and
  200 MHz agree within 0.1% for every row, including the cold-flash rows. The
  flash clock scales with the core clock on this core.
- **Compiler against library.** The integer-only rows (`lut_*`, `dot_i8_*`)
  differ by 0% to 7% between the two cores. The float rows differ by 1.1 to
  6.4 times. The float library causes the large differences, not the newer
  compiler.

### Operand size (build B)

Cycles per operation, by operand magnitude. The plain scalar rows above use
small operands (floats of 0.25 to 4, integers up to 2¹⁸).

| Operation | Mbed `_m7` | `_m15` | `_m22` | pico `_m7` | `_m15` | `_m22` |
| --- | --- | --- | --- | --- | --- | --- |
| `i2f` | 69.00 | 73.00 | 73.00 | 53.00 | 54.00 | 57.00 |
| `fadd` | 109.51 | 117.63 | 122.06 | 89.07 | 92.26 | 97.00 |
| `f2i` | 31.00 | 34.00 | 34.00 | 26.00 | 29.00 | 29.00 |
| `roundf` | 33.00 | 36.00 | 34.00 | 31.00 | 34.00 | 34.00 |

- The cost of `i2f`, `f2i`, and `roundf` changes by 3 to 4 cycles across the
  three classes on both cores.
- The cost of `fadd` changes by 12.6 cycles (11.5%) on the Mbed core and 7.9
  cycles (8.9%) on the `arduino-pico` core.
- The `roundf` cost on the Mbed core is not monotonic (33, 36, 34 cycles). It
  is not explained.
- The plain `f2i` row (27.75) used small operands. With state-size operands
  the `f2i` costs 31 to 34 cycles. In one `requantize` call, the `f2i` and
  `roundf` steps together cost 3 to 6 cycles more than the row in the main
  table (under 1% of 656 cycles). The cost model stays valid.
- `fmul`, `fdiv`, and `fcmp` were not tested with large operands.

### The h16 penalty in the scan kernel (build B)

With `s_h` scaled, the h16 states reach the real range (see the state report).
The scan iteration costs:

| Term | Mbed | pico, 125 MHz |
| --- | --- | --- |
| `scan_int8_iter`, h8 | 3,620.37 | 1,631.48 |
| `scan_int8_iter`, h16, range-scaled `s_h` | 3,655.65 | 1,646.28 |
| Total penalty | +35.28 (+0.97%) | +14.80 (+0.91%) |
| Storage-type effect (build A, same `s_h`) | +5.18 | +6.02 |
| Range effect (total minus storage type) | about +30 | about +9 |
| Range effect predicted from the operand-size rows | +6 | +9 |
| Not explained | about +24 | about 0 |

The prediction uses the operations whose operand size changes from h8 to h16.
The values are approximate, because a class is a range of magnitudes:

| Operation in the scan iteration | h8 class | h16 class |
| --- | --- | --- |
| `i2f` of `A_bar_q * h` | about 2¹³ (`_m15`) | about 2²¹ (`_m22`) |
| `roundf` of `new_h_real / s_h` | at most 91 (`_m7`) | at most 23,881 (`_m15`) |
| `f2i` of the clamped result | `_m7` | `_m15` |

The change in cost is the `i2f` step from `_m15` to `_m22`, plus the `roundf`
and `f2i` steps from `_m7` to `_m15`. On the Mbed core: 0 + 3 + 3 = +6 cycles.
On the `arduino-pico` core: 3 + 3 + 3 = +9 cycles.

- **`arduino-pico`:** the operand-size rows explain the range effect.
- **Mbed:** they explain about a fifth of it. About 24 cycles per iteration
  are not explained.
- **Against the sweep:** on the Mbed core, +35.28 cycles per iteration is
  72,300 cycles per hop (2,048 iterations), or 0.58 ms. The sweep shows +2.27 ms
  on the deployed model (finding 572). The microbenchmark explains about 25%.
- The storage-type effect comes from build A, and the total penalty comes from
  build B. The difference mixes two builds, so the split carries the layout
  uncertainty described next.

### Layout sensitivity (Mbed)

Some Mbed composite rows changed between builds although their code did not
change. The `arduino-pico` rows did not change.

| Row | Mbed, build A | Mbed, build B h8 | Mbed, build B h16 | pico (all builds) |
| --- | --- | --- | --- | --- |
| `silu` | 3,990.30 | 4,741.49 | 4,771.10 | 708.70 to 708.74 |
| `softplus` | 8,957 to 9,373 (4 runs) | 10,751.49 | 11,088.33 | 1,697.71 to 1,697.77 |
| `expf` | 3,449.45 | 3,449.51 | 3,449.57 | 538.74 to 538.76 |
| `log1pf` | 4,710.96 | 4,739.23 | 4,738.68 | 1,094.12 to 1,094.17 |
| `fdiv`, `fadd`, `fmul` | 454.08, 112.69, 163.31 | same | same | same |

- The `silu` row is `expf`, `fadd`, and `fdiv`. The sum of the parts is about
  4,016 cycles. The row in build A matched this. The row in build B is 725
  cycles higher.
- Between the h8 and h16 builds of build B, the rows that do not depend on `h`
  also moved: `silu` by +0.6% and `softplus` by +3.1%.
- Between builds A and B, the rows moved by up to 19%.
- The primitives (`expf`, `fdiv`, `fadd`, `fmul`) stayed within 0.6%.
- On the `arduino-pico` core the float routines run from the bootrom, not from
  flash behind the XIP cache. Those rows do not move.

**Possible cause, not tested:** the code layout in flash. Each build places the
libgcc and libm routines at different addresses. Hot routines can conflict in
the 16 KB XIP cache. No experiment has tested this. In particular, no padding
test was run.

The effect of this on the cost model:

- The `full_fp32` model with the build B `silu` and `softplus` rows gives
  127.5 ms per hop. With the build A rows it gives 124.5 ms. The sweep gives
  126.3 ms. The model error is +0.9% or −1.4%.
- The int8 model does not use `silu` or `softplus`. Its rows changed by 0.1% or
  less.
- Treat Mbed timing differences below about 2% to 3% with care. A composite
  kernel can move by more than that with no change in its source.

## How to compute the number of operations

The per-hop estimate has three steps: fix the model dimensions, count the
operations in each stage from the C source, and multiply each count by the
measured cost.

### Step 1: fix the dimensions

The deployed model (`case1/16662b29beb3`) has these dimensions. The values
are `#define` constants in `ssm_backbone.h`.

| Symbol | Meaning | C constant | Value |
| --- | --- | --- | --- |
| $D$ | Model width | `SSM_D_MODEL` | 64 |
| $E$ | Inner width | `SSM_D_INNER` | 64 |
| $N$ | State size | `SSM_D_STATE` | 16 |
| $K$ | Conv kernel size | `SSM_D_CONV` | 4 |
| $R$ | Delta rank | `SSM_DT_RANK` | 4 |
| $L$ | Layers | `SSM_N_LAYERS` | 2 |

### Step 2: map each C expression to primitives

Read the deployed `ssm_backbone.c` line by line. Do not count from the Python
reference, because the C code is what runs. Replace each expression with the
primitives that the microbenchmark measures:

| C expression | Primitives |
| --- | --- |
| `acc += w * x` (float) | 1 `fmul`, 1 `fadd` |
| `acc += (int32_t) w * (int32_t) x` | 1 int8 MAC |
| `(float) acc * a * b` | 1 `i2f`, 2 `fmul` (the `rescale` case) |
| `roundf(r / s)`, clamp, `(int8_t)` | 1 `fdiv`, 1 `roundf`, 2 `fcmp`, 1 `f2i` (the `requantize` case) |
| `x / (1 + expf(-x))` | The `silu` case |
| `x > 20 ? x : log1pf(expf(x))` | The `softplus` case |
| `lut[(int) q + 128]` | 1 LUT lookup |
| `a < b` (float) | 1 `fcmp` |

For each loop, multiply the loop bounds by the number of primitives in the
body. The compiler can merge operations, so check the disassembly for any term
that you find important.

### Step 3: count per stage

Counts per hop. A hop runs once for each layer and one time for the final
stage.

**Matvec and conv MACs.** Each weight is used once per hop. The count per
layer is:

$$M = 2DE + EK + (R + 2N)E + ER + ED$$

The five terms are `in_proj`, `conv`, `x_proj`, `dt_proj`, and `out_proj`.
For the deployed model:

$$M = 8{,}192 + 256 + 2{,}304 + 256 + 4{,}096 = 15{,}104$$

The total is $LM = 30{,}208$ MACs per hop.

Sanity check: the MAC count must equal the number of matvec and conv weights.
The `full_fp32` weight file is 131,924 bytes, which is 32,981 floats. Of these,
30,208 are matvec and conv weights (91.6%). The rest are `A` (2,048), `D`,
biases, and norm weights.

**Scan iterations.** $S = L \cdot E \cdot N = 2 \cdot 64 \cdot 16 = 2{,}048$.

**Nonlinearities (fp32).** Each layer calls `silu` $2E$ times (conv output and
gate) and `softplus` $E$ times. Per hop: 256 `silu` and 128 `softplus`.

**Nonlinearities (int8).** The same positions use LUT lookups: $3EL = 384$
lookups per hop.

**Requantize calls outside the scan (int8).** Count each call site in
`ssm_block_step_true_int8` and `SSMBackbone_ProcessFrame`:

| Call site | Count per layer |
| --- | --- |
| `u_raw_q` and `z_q` | $2E$ |
| `conv_out_q` | $E$ |
| `delta_low_q`, `B_q`, `C_q` | $R + 2N$ |
| `delta_raw_q` | $E$ |
| `y_scan_q` and `y_gated_q` | $2E$ |
| `block_out_q` | $D$ |
| `x_norm_q` (in `ProcessFrame`) | $D$ |

The sum per layer is $6E + 2D + R + 2N = 548$. The final norm adds $D = 64$.
The total is:

$$Q = L(6E + 2D + R + 2N) + D = 2 \cdot 548 + 64 = 1{,}160$$

The calls inside the scan loop are not in $Q$. The scan kernel cost already
includes them.

**Output rescales (int8).** Each `ssm_qlinear_real` output needs one rescale
(`i2f` and 2 `fmul`). Per layer: $4E + D + R + 2N = 356$. Per hop: 712. The
`conv` and `dt_proj` outputs also add a bias (`fadd`): $2E$ per layer, 256 per
hop.

**Other terms.**

| Term | Count per hop |
| --- | --- |
| RMSNorm calls | $L + 1 = 3$ |
| Channel epilogues (after the scan loop) | $LE = 128$ |
| Residual adds | $LD = 128$ |
| Final norm and pooling elements | $D = 64$ |

An int8 channel epilogue holds 4 `i2f`, 7 `fmul`, and 1 `fadd`. The
`delta_real` line (1 `i2f` and 1 `fmul` per channel) is not counted here,
because the `scan_int8_iter` kernel already includes it. An fp32 channel
epilogue holds 2 `fmul` and 1 `fadd` (the `silu` of the gate is in the
`silu` count).

### Step 4: multiply and add

$$T_{\text{hop}} = \sum_i n_i \cdot c_i \qquad t_{\text{hop}} = \frac{T_{\text{hop}}}{f_{\text{MHz}}}$$

Here $n_i$ is the count of term $i$ and $c_i$ is its measured cost in cycles.
The worked examples use the build A rows.

**Worked example: `full_fp32` on the Mbed core.**

| Term | Count | Cost (cycles) | Total (M cycles) |
| --- | --- | --- | --- |
| MACs (cold flash) | 30,208 | 348.19 | 10.518 |
| Scan iterations | 2,048 | 1,270.37 | 2.602 |
| `silu` | 256 | 3,990.30 | 1.022 |
| `softplus` | 128 | 9,373.17 | 1.200 |
| RMSNorm | 3 | 40,078.71 | 0.120 |
| Channel epilogue | 128 | 439.31 | 0.056 |
| Residual and pooling (`fadd`) | 192 | 112.69 | 0.022 |
| **Total** | | | **15.539** |

The total is 15.539M cycles, or 124.5 ms at 124.79 MHz. With warm-flash MACs
(296.08 cycles each) the total is 13.96M cycles, or 111.9 ms. With the build B
`silu` and `softplus` rows (4,741.49 and 10,751.49 cycles), the total is
15.907M cycles, or 127.5 ms.

**Worked example: `true_int8_h8` on the Mbed core.**

| Term | Count | Cost (cycles) | Total (M cycles) |
| --- | --- | --- | --- |
| Scan iterations | 2,048 | 3,616.96 | 7.407 |
| MACs (cold flash) | 30,208 | 26.91 | 0.813 |
| Requantize outside the scan | 1,160 | 656.27 | 0.761 |
| Output rescales | 712 | 392.32 | 0.279 |
| Bias adds | 256 | 112.69 | 0.029 |
| Channel epilogue | 128 | 1,532.38 | 0.196 |
| LUT lookups | 384 | 9.00 | 0.003 |
| Residual and final norm | 192 | 345.13 | 0.066 |
| RMSNorm | 3 | 40,078.71 | 0.120 |
| **Total** | | | **9.676** |

The total is 9.676M cycles, or 77.5 ms at 124.79 MHz. With warm-flash MACs
(14.29 cycles each) the total is 9.29M cycles, or 74.5 ms.

### Step 5: split the scan iteration

A scan iteration is a good check of the primitive costs, because the sum of
its parts is independent of the scan measurement. The parts are counted from
the loop body:

| Part (int8, Mbed) | Count | Cycles | Share |
| --- | --- | --- | --- |
| Requantize-like steps (`A_bar`, `B_bar`, `new_h`) | 3 | 1,968.8 | 54.4% |
| `fmul` | 7 | 1,143.2 | 31.6% |
| `fadd` | 2 | 225.4 | 6.2% |
| `i2f` | 3 | 207.4 | 5.7% |
| `fcmp` (the `-1.9` clamp) | 1 | 70.8 | 2.0% |
| **Sum** | | **3,615.6** | |

The measured `scan_int8_iter` is 3,616.96 cycles. The difference is 0.04%.

For the `full_fp32` scan (5 `fmul`, 3 `fadd`, 1 `fcmp`) the sum is 1,225.4
cycles. The measured value is 1,270.37 cycles. The 45 cycles that remain are
the loads, stores, and loop overhead that the primitive costs do not include.

One `requantize` call (656.27 cycles) splits into `fdiv` 454.08 (69.2%),
2 `fcmp` 141.62 (21.6%), `roundf` 32.75 (5.0%), and `f2i` 27.75 (4.2%). The
sum is 656.20.

On the `arduino-pico` core one `requantize` call (255.75 cycles) splits into
`fdiv` 89.19 (34.9%), 2 `fcmp` 100.88 (39.4%), `roundf` 28.75 (11.2%), and
`f2i` 23.75 (9.3%). The two compares now cost more than the divide.

## Model against the sweep

The sweep is `test_harness/output_rp2040_naive/cycle_sweep_rp2040_summary.csv`
in `ssm-mcu-asd-deploy`. It ran on the Mbed core at 125 MHz with the fixed
upload step. Each setup used 2 clips. The table uses 344 hops per clip, the
value from plan 540. The firmware prints `hops=` on USB `Serial`, so you can
check it.

| Setup | Sweep backbone (µs per clip) | Per hop | Model, cold | Model, warm |
| --- | --- | --- | --- | --- |
| `full_fp32_euclidean_fp32` | 43,459,319 | 126.3 ms | 124.5 ms (−1.4%) | 111.9 ms (−11.4%) |
| `true_int8_h8_euclidean_int8` | 25,626,601 | 74.5 ms | 77.5 ms (+4.1%) | 74.5 ms (0.0%) |
| `true_int8_h16_euclidean_int8` | 26,408,824 | 76.8 ms | not modeled | not modeled |

The `full_fp32` model uses the build A rows. With the build B `silu` and
`softplus` rows the cold model gives 127.5 ms (+0.9%).

The int8 measurement matches the warm prediction and is 4.1% below the cold
one. The fp32 measurement is close to the cold prediction, which matches the
size of the fp32 weights (121 KB per hop against a 16 KB cache).

### Model prediction for the `arduino-pico` core

These are predictions from the same method. They are not measurements. The
model uses the `arduino-pico` costs at 125 MHz and the cold-flash MAC cost.

| Setup | Cycles per hop | At 124.42 MHz | At 199.42 MHz |
| --- | --- | --- | --- |
| `full_fp32` | 8.591M | 69.0 ms | 43.1 ms |
| `true_int8_h8` | 4.786M | 38.5 ms | 24.0 ms |

The budget for one hop is 32 ms. The feature stage is not modeled here. It
takes about 38 ms per hop on the Mbed core.

For the h16 build on the `arduino-pico` core, the scan penalty of +14.80 cycles
per iteration is 30,300 cycles per hop, or 0.24 ms at 124.43 MHz (0.6% of the
modeled hop).

## Open questions

- **The cause of the h16 penalty in the sweep is not established.** The sweep
  shows that h16 is slower than h8 on all six model pairs, by 1.4% to 3.4%. On
  the deployed model the penalty is 2.3 ms per hop, about 284,000 cycles. With
  realistic h16 states, the microbenchmark gives 72,300 cycles per hop on the
  Mbed core (about a quarter) and 30,300 cycles per hop on the `arduino-pico`
  core. Within one model, the `qab` and `qabar` variants show the same
  absolute penalty. So the cause is in code that both variants share, not in
  the `A_bar` and `B_bar` requantize steps.

  Candidates for the rest of the Mbed penalty, none of them tested:

  1. The code layout in flash. The h16 build differs from the h8 build by 12
     bytes in the backbone, which shifts the code placed after it. Composite
     rows that do not depend on `h` also moved between the h8 and h16
     microbenchmark builds (see "Layout sensitivity"). No padding test was
     run.
  2. The RAM layout. The h16 state is 2,048 bytes larger.
  3. The operand-size effect of `fmul`, `fdiv`, and `fcmp`. These were not
     measured with large operands.

  **Claude comment:** The operand-size hypothesis from the earlier version of
  this finding is partly supported. It explains the `arduino-pico` penalty, and
  about a fifth of the Mbed penalty. The pico sweep is a free test of the
  layout candidate. The microbenchmark predicts about 0.6% on the modeled
  backbone for `arduino-pico`. If the pico sweep shows about the same, the
  extra Mbed penalty is specific to the Mbed build.
- **The int8 model is 4.1% high with cold-flash MACs.** The measured time
  matches the warm prediction. The XIP cache hit rate for the real weight
  access pattern is not measured.
- **The terms outside the MACs, the scan, and the requantize calls are
  estimated from the primitive costs.** Stage timers in the real firmware
  (plan 540, L0.1) confirm them.
- **Mbed composite rows depend on the build.** `silu` and `softplus` moved by
  up to 19% between builds. The cause is not established (see "Layout
  sensitivity").
- **The `roundf` cost on the Mbed core is not monotonic** across the operand
  classes (33, 36, 34 cycles). It is not explained.
- **The clock is 0.2% to 0.5% below nominal.** The cause is not confirmed.
- **The `arduino-pico` rows used the generic "Raspberry Pi Pico" board
  entry.** The Nano RP2040 Connect has 16 MB of flash. The compute rows do not
  depend on the flash size. A port to this core needs the Nano board entry.
- **The 200 MHz build is an overclock.** The RP2040 is rated for 133 MHz.
  Report the 200 MHz results as a separate, labeled configuration. Build B was
  run at 125 MHz only.
- **The feature stage is not covered.** It takes about 38 ms per hop on the
  Mbed core, which is more than the 32 ms hop budget.
- **The model covers the steady-state hop.** It excludes `SSMBackbone_GetPooled`
  and the head. Both run once per clip.

## What this finding does not show

- It does not show that ROM float gives the same numerical results as libgcc.
  The `arduino-pico` core wraps the SDK float routines, and other projects
  have reported differences in edge cases and in the accuracy of the math
  functions. Run the parity check (finding 542) before you use ROM float for
  any accuracy result.
- It does not show a latency for the `arduino-pico` core on the real
  firmware. The `arduino-pico` figures above are model predictions.
- It does not show that the code layout causes the Mbed differences. That is a
  possible cause and it is not tested.

**Claude comment:** For the dissertation, I suggest that you report the RP2040
in two configurations: the stock Mbed build and a ROM-float build. The result
that the latency benefit of quantization depends on the float runtime, because
requantization needs a float divide, is more useful than a single number. A
reciprocal multiply in place of the divide is an obvious optimization, but it
can flip a rounding at a near-tie and break the bit-exactness of finding 542.
Test it against the parity vectors first.