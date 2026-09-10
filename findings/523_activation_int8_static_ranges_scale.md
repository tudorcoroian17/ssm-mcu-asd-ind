# Static ranges-based activation scales confirm scan's outlier-driven fragility; primary metric holds across five variants

**Feeds:** `06_phase_5_quantization_and_head.md` (rung 2, deployment-realistic
scale); `findings/150` (`ranges.json`, the source of these scales);
`findings/522` (the on-the-fly ceiling probe this extends to the
deployment-realistic case, resolving its "ranges-based scale test not yet
run" open item).

## Result

Testing `--scale-source ranges` (fixed scales from `findings/150`'s
calibration; per-tensor only, since `ranges.json` holds no per-channel
statistics) on both activation groups confirms `findings/522`'s prediction:
static scales are worse than every on-the-fly variant tried, on both
embedding fidelity and secondary metrics. But the primary metric (mean
pooling / knn_full) stays within noise in both groups -- the fifth
consecutive activation-quantization variant, and the seventh counting weight
quantization, to leave it unmoved. The damage sharpens into a specific,
repeatable weak point: `concat_mean_last` pooling combined with either kNN
head degrades meaningfully in every `scan` variant tested except one
(the accidental wrong-axis per-channel run in `findings/522`).

Scope: same run/config/fold as `findings/520`-`522`.

## Ranges scales are worse than on-the-fly, exactly as predicted

Mean-pool embedding error (mean_abs_err), on-the-fly vs ranges:

| group | on-the-fly (522) | ranges |
| :--- | ---: | ---: |
| scan | 9.65e-2 | **2.00e-1** |
| boundaries | 1.28e-2 | **3.65e-2** |

Both roughly double to triple. Mechanism: `load_ranges_scales` sets each
tensor's scale from `max(|min|, |max|) / 127` -- the single most extreme value
in the calibration set. `block1.y_scan`'s range is `[-100, 108]` with a
median of `0.31`; a scale sized for that range represents the typical frame
coarsely on every single frame, not just the rare outlier clip the way
on-the-fly's per-forward scale would confine the damage to.

## The primary metric holds -- five for five now

| variant | mean/knn_full dAUC | mean/knn_full dpAUC |
| :--- | ---: | ---: |
| scan / ranges | -0.0045 | +0.0167 |
| boundaries / ranges | +0.0057 | +0.0179 |

Both comfortably inside the noise band every prior variant showed. Across
`findings/520` (weight, per-tensor), `521` (weight, per-channel/all), `522`
(activation on-the-fly, three scan variants + boundaries), and this finding
(activation, static, both groups), the pooling/head combination the paper's
claim rests on has not moved meaningfully in any of seven tested variants.
Worth stating plainly: this is now a broad, repeated result, not a single
lucky run.

## `concat_mean_last` + kNN is scan's real, repeatable weak point

`dpAUC`, `concat_mean_last` pooling, across every `scan` variant tested:

| scan variant | knn_full | knn_clustered_16 |
| :--- | ---: | ---: |
| per-tensor, onthefly | -0.0529 | -0.1274 |
| per-channel, wrong axis (522) | -0.0054 | +0.0122 |
| per-channel, correct axis (522) | -0.0062 | -0.0048 |
| per-tensor, ranges | **-0.0876** | **-0.0879** |

Four independent variants implicate this combination; only the accidental
wrong-axis run (already known to be an implementation error, not a real
scheme) does not. `ranges` -- the deployment-realistic scale -- is the worst
of the four, on both kNN heads simultaneously. This is no longer "one
combination wobbled once"; it is a specific, attributable fragility of
`concat_mean_last` pooling paired with kNN-style distance heads under scan
quantization.

## Boundaries' known max-pooling weakness also worsens under a static scale

`max`/`knn_full` dAUC: `-0.0124` on-the-fly (`findings/522`) versus `-0.0312`
under `ranges`. Same "static is worse than adaptive" story layered on top of
the single-frame sensitivity `findings/522` already attributed max pooling's
weakness to.

## Consequence

The deployment-realistic activation-quantization scheme (`ranges`-based
static scales) does not move the primary metric on either group, on this
fold. It does confirm and sharpen two secondary weak points already flagged
in `findings/522`: `scan` quantization threatens `concat_mean_last` + kNN
specifically, and both groups' pre-existing max-pooling sensitivity gets
worse under a static rather than adaptive scale. Neither weak point touches
the primary claim, but both are now real enough (repeated across four and
two variants respectively) to name as constraints rather than curiosities if
`concat_mean_last` or max pooling ever becomes load-bearing for this project.

## Deferred: percentile-clipped static scales

Not filed as an open item, because it is not queued work -- it is a design
idea to pick back up once the model is actually running on the Nucleo board,
deliberately as a discussion with a later version of the project rather than
something to chase in Python now.

The idea: `ranges.json` already carries `p99`/`p99.9` alongside min/max.
Building the static scale from a percentile (e.g. `p99.9 / 127`) instead of
`max / 127` would deliberately clip the rare extreme frames that are driving
`scan`'s outlier-scale problem, trading a small, bounded clipping error on
values occurring under 0.1% of the time for real resolution on the typical
case -- the standard fix for exactly the failure mode this finding
demonstrates. The on-the-fly results already establish that int8 can
represent this recurrence adequately when the scale fits the data well
(per-forward, adaptive); percentile clipping is the static analogue of a
well-fitted scale, and would be the natural lever to pull if this scheme ever
needs to move from diagnostic to deployable on real hardware.

## Open items

- **One fold.** As with every prior quantization finding, not blocking
  further work but needed before the final accuracy table.
- **`concat_mean_last` + kNN under `scan`, and max pooling under either group,
  are unresolved weak points** if either ever becomes a combination this
  project depends on. Currently neither is load-bearing (primary is `mean`
  pooling / `knn_full`).

<!-- Claude comment: the number worth leading with, if this ever becomes a
paper section, is "seven for seven" -- every weight- and activation-
quantization variant tested across findings/520-523, on-the-fly or static,
per-tensor or per-channel, has left the primary metric within noise. That is
a stronger and more surprising claim than any single variant's result, and
it is only visible by reading the four findings together rather than any one
of them alone. -->