# EVT threshold collapse on true-int8 heads, fixed by switching to parametric fitting

**Feeds:** `findings/140_thresholds_and_secondary_metrics.md` sec 6-7 (the
EVT collapse-risk warning and the shape-parameter diagnostic this reused
directly, at a new operating point neither section tested); `findings/542`
(the backbone this head sits on top of); `mcu/export_reference_heads.py`
(the script whose default threshold method this changed).

## Result

`export_reference_heads.py`'s first export for the `true_int8` scheme used
`evt_threshold` (EVT/GPD tail fitting) at `target_far=0.01` and produced
secondary metrics far below what `findings/140` established for the fp32
baseline on the same fold and heads -- recall as low as 0.072
(`euclidean`, `h=int16`). Switching only the threshold-estimation method to
`parametric_threshold` (already implemented in `src/eval/thresholds.py`,
and already `findings/140` sec 9's actual deployment recommendation --
EVT was never the recommended default, a choice this export script had
gotten wrong) recovered most of the loss, with AUC unchanged, confirming
the drop was a threshold-estimator artifact, not evidence of a problem in
the backbone or the embeddings themselves.

## Before / after (case 1, `n_test=528`, `target_far=0.01`)

| h-width | head | AUC | threshold method | threshold | P | R | A | F1 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| int8 | euclidean | 0.8968 | EVT | 3.706 | 0.931 | 0.205 | 0.595 | 0.335 |
| int8 | euclidean | 0.8968 | parametric | 3.027 | 0.942 | 0.371 | 0.674 | **0.533** |
| int8 | knn_16 | 0.9314 | EVT | 2.444 | 0.971 | 0.500 | 0.742 | 0.660 |
| int8 | knn_16 | 0.9314 | parametric | 2.249 | 0.968 | 0.576 | 0.778 | **0.722** |
| int16 | euclidean | 0.9465 | EVT | 3.860 | 0.826 | 0.072 | 0.528 | 0.132 |
| int16 | euclidean | 0.9465 | parametric | 2.746 | 0.947 | 0.470 | 0.722 | **0.628** |
| int16 | knn_16 | 0.9156 | EVT | 2.148 | 0.979 | 0.538 | 0.763 | 0.694 |
| int16 | knn_16 | 0.9156 | parametric | 1.915 | 0.984 | 0.712 | 0.850 | **0.826** |

AUC is identical within each h-width/head pair across both rows, as
expected -- switching the threshold method changes only where the decision
boundary sits on an unchanged score distribution; it cannot and does not
touch ranking quality. The entire F1 change comes from recall: precision
was already high under EVT (0.826-0.979) and stayed high or improved
slightly under parametric (0.942-0.984). This is precisely the shape of a
threshold placed too high, not a ranking problem -- consistent with
`findings/140` sec 3's original diagnosis of the pre-`calib_normal`
calibration failure ("the ranking was fine everywhere; the entire loss was
threshold placement"), now recurring for a different reason at a different
stage of the same project.

## Method

The trigger was a direct comparison against `findings/140`'s own numbers:
case-1 `mean/knn_clustered_16` at p95 (same-machine calibration) scored
F1=0.892 for the fp32 baseline; the first true-int8 EVT export scored
0.660-0.694 at the stricter p99 target. Some drop from a stricter
false-alarm budget is expected on its own, but not a drop of that size --
`findings/140` sec 6 shows the SAME heads gaining, not losing, F1 when
pushed from p95 to p99 via parametric fitting (`mean/knn_full`: 0.9534 ->
0.9624; `mean/knn_clustered_16`: 0.9436 -> 0.9596, both fp32).

`findings/140` sec 7 supplies a cheap, decisive diagnostic for exactly
this situation: the GPD shape parameter, with "shape >= 0.676 predicts
collapse with zero exceptions" across 48 configurations tested there. The
`evt_info` block from one of the two heads' true-int8 export showed
shape=0.9454 at `h=int16` -- squarely inside that zero-exception collapse
zone -- and shape=0.4915 at `h=int8`, inside the "unpredictable middle
zone" sec 7 also documents (not a guaranteed collapse, but not safe
either). Both values are consistent with the observed pattern: the
int16/euclidean combination showing the worst EVT recall (0.072) is the
one with the shape value furthest into the confirmed-collapse region.

The mechanism sec 6 already gives for this: EVT fits its GPD only to the
top `tail_fraction` (10%) of `calib_normal`, ~108 points here out of
~1,085. Parametric fitting uses the entire calibration sample via MLE +
AIC family selection across four candidate distributions. At an aggressive
quantile (p99), 108 points is a much less stable base to extrapolate from
than 1,085 -- exactly the effect sec 6/7 characterize, previously observed
at p999 (11/96 total collapses) and now observed at p99 for this
particular (quantized) score distribution, which sec 7 never tested.

## Consequence

`export_reference_heads.py`'s `--threshold-method` default is now
`parametric`, matching `findings/140` sec 9's actual recommendation rather
than the EVT choice this script started with -- EVT remains available via
`--threshold-method evt` for direct comparison, not as the default. Any
future quantized-scheme threshold export (the `combined`/
`weight_act_boundaries_ranges` scheme's own head export, not yet run,
included) should use this corrected default rather than rediscovering the
same collapse independently.

A real, smaller gap remains even after this fix:
`true_int8`'s best `knn_16` F1 (0.826, `h=int16`) is still below fp32's
case-1 p95 number (0.892), and further below fp32's own p99-equivalent
cross-fold mean (0.9596). Unlike the EVT-vs-parametric gap, this remaining
difference is not attributable to a threshold-estimation artifact -- AUC
itself is modestly lower for true-int8 (0.9156-0.9314) than fp32's typical
range, and a somewhat less separated score distribution costs recall at
any fixed low-FAR operating point even under an optimal threshold. Expected
consequence of quantization, not a new problem to chase.

## Open items

- **Which head (`euclidean` or `knn_16`) the two pasted `evt_info` blocks
  belong to was never explicitly confirmed** -- inferred from the
  correlation between shape magnitude and recall severity, not verified
  directly against the diagnostics JSON's key structure. Worth a five-second
  check next time this comes up, rather than relying on the inference here.
- **No case-1-specific fp32 p99-equivalent F1 number exists to compare
  against directly** -- `findings/140` sec 6's p99 numbers are a cross-fold
  mean, not broken out by case. The comparison above uses case-1's p95
  number as the closest available reference point, which understates how
  much of the remaining gap is the stricter FAR target versus quantization
  itself.
- **The `combined`/`weight_act_boundaries_ranges` scheme's own head export
  has not been run yet** -- unknown whether it would have shown the same
  EVT collapse risk (it likely has a smoother score distribution than
  true-int8's discrete arithmetic, similar to how `findings/542` found
  true-int8's discreteness amplifies input noise that fp32/fake-quant
  schemes absorb smoothly -- the same mechanism could plausibly make its
  `calib_normal` distribution better-behaved for EVT too, but this is an
  untested guess, not a finding).

<!-- Claude comment: the reusable lesson here isn't "EVT is bad" -- sec 6
already showed EVT is fine at p95, agreeing with every other method within
0.005 F1. It's that a threshold-estimation method's reliability is a
property of (sample size, quantile aggressiveness, score distribution
shape) jointly, and a method validated as safe on one scheme's score
distribution at one quantile doesn't transfer automatically to a new
scheme at a stricter quantile, even holding the underlying heads and fold
constant. This export script defaulted to EVT for a reason that sounded
right in isolation ("EVT is designed for aggressive tail quantiles") but
was wrong in context, because this project had already run the specific
experiment that says which method to prefer for these two heads. The fix
was cheap specifically because that prior experiment existed to consult --
the shape diagnostic turned a "why is recall bad" investigation into a
five-minute confirmation. -->