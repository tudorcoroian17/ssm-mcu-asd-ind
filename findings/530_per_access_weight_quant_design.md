# Per-access weight dequantization -- one backbone for every scheme, and the RXFAIL race eliminated structurally

**Feeds:** `06_phase5_nucleo_deployment_matrix.md` Phase 1 (weight-only C
support); `findings/510` (where the RXFAIL race was first found and
partially fixed); `findings/520`/`521` (the Python-validated schemes this
design serves on real hardware).

## Result

`ssm_backbone.c` reads every weight through a per-field macro
(`SSM_<FIELD>(w, ...)`), generated once per export as either a plain float
read or an inline int8-dequantize expression, depending on whether that
field was quantized in that specific export. The `.c` source text is
identical across the fp32 baseline and all four weight-only schemes -- no
per-scheme branches, no runtime "is this quantized" check anywhere. This
design has no lazy initialization step at all, which turned out to
structurally eliminate a UART race condition (see below) that an earlier
design had introduced and only partially fixed.

## First attempt: boot-scratch, and why it was abandoned

The first working design dequantized every int8 field once, at boot, into
static fp32 scratch buffers (`SSM_InitWeights()`), and pointed
`ssm_block_step` at those buffers instead of the raw exported arrays. It
validated correctly (bit-identical fp32 parity, `findings/430`'s
`5.99e-4`/`1.45e-4`), but on inspection it doubled memory rather than saving
it: flash is memory-mapped, so the `const int8_t` arrays are never freed --
the fp32 scratch buffers land on top of them in RAM, permanently. For any
quantized tensor the design paid for both the compact form and the
expanded form at once.

It also introduced a race. `SSM_InitWeights()` ran inside
`SSMBackbone_Reset()`, at the top of the per-clip loop, guarded to do real
work only once. On a fresh flash, that one real execution landed in the gap
between receiving the clip header and arming the first hop's DMA -- and the
host streams hop bytes with no per-hop handshake, straight into a UART with
no FIFO (the same finding that motivated DMA reception in the first place,
`findings/510`). If the dequant work took long enough, the first hop's
earliest bytes arrived before DMA was armed to catch them, and were lost.
Every clip after the first hit the now-free idempotent path with no gap, so
only clip one ever failed -- on every fresh flash, reliably.

Moving the call to `main()`, before `BSP_COM_Init()` and any host
communication, fixed the specific symptom (nothing was racing it there) but
didn't address the underlying design cost.

## The design used instead: per-access, no scratch, no boot step

Weights stay exactly where they're exported -- plain float array or int8
array + scale, always in flash. `ssm_block_step` dequantizes inline, at the
point of use, via the generated macro. Consequence: there is no lazy
one-time work anywhere in the backbone. `SSMBackbone_Reset()` no longer
calls anything weight-related. The RXFAIL race didn't get fixed again here
-- it became structurally impossible, since nothing is left to race against
the first hop's DMA arming.

Two special cases:
- **`A`**: the checkpoint stores `A_log`; `findings/521` established that
  the quantized tensor must be `A_log`, not the exponentiated `A`. The
  macro applies `-expf(...)` after dequantizing, every access, when this
  field is quantized.
- **Norm weights**: a bare `const float *` can't represent "maybe
  int8+scale instead," so norm weights use a small nested
  `ssm_norm_weights_t` struct rather than a raw pointer.

Verified: an fp32 rebuild under this design reproduced `findings/430`'s
`5.99e-4`/`1.45e-4` bit-for-bit before any quantized scheme was trusted.

## A costly false trail during validation -- worth recording precisely

Chasing an apparent ~15x-worse-than-expected parity gap on the first
quantized scheme led through several ruled-out hypotheses (streaming vs.
batched evaluation order, stale Python reference file, stale compiled
object) before the real cause surfaced: `Core/Src/ssm_weights.c` was a
*different scheme's* export (`weight_int8_proj_perchannel`, generated later
and copied in by mistake) rather than the one believed to be under test
(`weight_int8_proj_pertensor`). Every diagnostic along the way was correct
and the arithmetic was never wrong -- the comparison itself was between the
wrong two things. Resolved by reading the actual compiled `.c` file's raw
`q`/`scale` values directly rather than trusting which folder was
*supposed* to have been copied where.

Consequence adopted for the rest of the deployment matrix: before trusting
any parity or footprint number, confirm which scheme is actually sitting in
`Core/Src` (grep the scale array's element count and whether `A_log_q` is
present, which together identify weight-mode and granularity unambiguously)
and confirm the `.map`/`.elf` postdates that exact source file's
modification time. Both checks caught real staleness more than once during
this work and are cheap enough to be routine, not exceptional.

<!-- Claude comment: the RXFAIL race is a good example of a fix that looked
complete (moving the init call earlier resolved the symptom) but wasn't
addressing the design property that caused it. The per-access rewrite was
motivated purely by the memory-doubling problem; the race disappearing
was a side effect nobody was solving for when the rewrite started. Worth
remembering when a bug's proximate fix works: ask whether the design that
allowed it is still there, not just whether the symptom is gone. -->