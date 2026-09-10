# Per-channel and full-weight int8 do not beat per-tensor projections-only

**Feeds:** `06_phase_5_quantization_and_head.md` (rung 1 scheme choice);
`findings/460` (which established projections/per-tensor as viable and is
extended, not replaced, here). Settles the granularity and weight-mode axes
before the export-script and C changes.

## Result

Extending the `findings/460` weight-int8 check across two axes -- weight mode
(`projections` vs `all`, which adds A_log, D, and biases) and granularity
(`per-tensor` vs `per-channel`) -- shows all four combinations preserve AUC on
the case-1 fold to within the same noise band. Per-channel reconstructs the
weights more faithfully but does not improve, and slightly worsens, the
anomaly-detection AUC. The recommendation is unchanged and now positively
supported against the alternatives: `projections` + `per-tensor`, the cheapest
and most conservative rung.

Scope: `f4cd557b7e3b`, case 1, run `16662b29beb3`. One fold.

## The A_log naming point (settled)

Under `--weight-mode all`, the quantized recurrence tensors report as
`blocks.N.A_log`, not `blocks.N.A`. The checkpoint stores the log-space
parameter; the C export materializes `A = -exp(A_log)` itself. So this pass
quantizes `A_log` (SNR 47-50 dB, well behaved), which is a different -- and
smoother -- tensor than the `-exp(A_log)` the C port stores. Consequence for
the C stage: to reproduce a validated `all`-mode result on device, the export
script would quantize `A_log` and materialize after dequant, not quantize the
final `A`. Not relevant to the recommended scheme, which leaves A_log fp32.

## Finding 1: per-channel halves embedding error, as theory predicts

Per-tensor to per-channel, mean embedding absolute error:

| weight mode | per-tensor | per-channel |
| :--- | ---: | ---: |
| projections | 2.22e-2 | 1.01e-2 |
| all | 2.25e-2 | 8.77e-3 |

Per-channel SNR is uniformly higher (per-tensor 36-52 dB, per-channel
43-52 dB). By embedding-reconstruction fidelity, per-channel is the better
scheme, and `all` vs `projections` barely moves reconstruction either way.

## Finding 2: better reconstruction does not mean better AUC -- the opposite, slightly

The AUC/pAUC deltas invert the fidelity ranking:

| combination | primary head (knn_full, mean) dAUC | overall dAUC sign |
| :--- | ---: | :--- |
| projections / per-tensor | +0.0012 | all 12 positive |
| all / per-tensor | +0.0003 | 11 of 12 positive |
| projections / per-channel | -0.0013 | mostly negative |
| all / per-channel | -0.0009 | mostly negative |

The two per-tensor schemes drift AUC uniformly positive; both per-channel
schemes drift it mostly negative (and pAUC broadly negative). Every delta, in
all four combinations, stays within +/-0.01 AUC on the primary head.

## Interpretation: all four are a wash, and the deltas are not signal

None of these deltas is real accuracy signal -- all sit inside the fold's own
seed noise band (`SEED_SD = 0.0132`, `findings/210`/`findings/250`). The
consistent *sign* split between per-tensor (positive) and per-channel
(negative) is the informative part, and it has a mechanism:

- Per-tensor int8 shares one scale across a tensor, so its rounding error is
  correlated/structured. That structure mildly smooths the embedding geometry
  in a way that happens to help distance-based separation -- the same
  incidental-regularizer effect `findings/460` noted.
- Per-channel gives each output channel its own tight scale, removing that
  structure. The reconstruction is more faithful but the residual error is
  finer-grained and uncorrelated, so the incidental smoothing goes away and
  AUC settles back to just below fp32.

So the axes do not trade accuracy -- they trade *which side of fp32 the noise
lands on*. The real finding is that the scheme choice does not matter for
accuracy on this fold.

## Consequence: choose on cost, which points at the cheapest rung

With accuracy a wash, the decision falls to cost, which these Python scripts
do not measure but which now decides:

- **Per-tensor** stores one fp32 scale per tensor (13 total). Trivial flash
  cost; simplest C dequant (one scale per array).
- **Per-channel** stores one fp32 scale per output channel (e.g. 128 for a
  `[128, 64]` projection), across every tensor. More flash; the C dequant
  must index the scale by output channel inside the loop.
- **`all` vs `projections`** is also a wash on accuracy, so quantizing A_log/D
  buys nothing -- and leaving them fp32 keeps the recurrence params exact,
  sidestepping the `findings/150` section 6 half-life risk for free.

Recommended scheme, unchanged from `findings/460` and now supported against
all three alternatives: **projections + per-tensor**. Cheapest, simplest in C,
and most conservative on the one part of the model `findings/150` flagged.

## Open items

- One fold. The per-tensor-positive / per-channel-negative sign split would be
  worth confirming on a second fold before the final writeup -- if per-tensor
  went reliably positive across folds a reviewer might reasonably ask why, and
  "incidental regularization within noise" is the answer to have ready. Not
  blocking the C port.
- Activation quantization still untouched; it is a separate effort (needs
  `ranges.json` and the `ssm_block.py` inline-quant path), and is where
  `findings/150`'s dynamic-range concerns actually bite.

<!-- Claude comment: the counterintuitive core here -- more faithful weight
reconstruction giving slightly worse AUC -- is a genuinely useful thing for
the paper to state plainly, because a reader's default assumption is "per-
channel is strictly better." The defensible claim is narrow: on this fold,
granularity and weight-mode are accuracy-neutral within seed noise, so the
deployment choice is made on footprint and implementation cost, not accuracy.
That is a stronger and more honest position than picking per-channel because
it "should" be better. -->