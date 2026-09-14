# Deployment setup: `true_int8_h16_qabar_euclidean_int8`

Combo 24 of the case 2 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 2
- Model hash: `d610f3c01dd7`
- Backbone: true int8 arithmetic, h storage = int16
- Head: euclidean head (int32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.327602** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=0.989, R=0.672, A=0.832, F1=0.800.

Ranking quality (threshold-independent): AUC=0.8731, pAUC=0.8617.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0341339825) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.292141 | 1433 | 0.935 | 0.755 | 0.851 | 0.835 |  |
| `percentile_same_machine_99.0` | 1.327602 | 1513 | 0.989 | 0.672 | 0.832 | 0.800 | **<- active** |
| `percentile_same_machine_99.5` | 1.341002 | 1543 | 0.994 | 0.638 | 0.817 | 0.777 |  |
| `evt_p95_equiv` | 1.293324 | 1436 | 0.934 | 0.747 | 0.847 | 0.830 |  |
| `parametric_p95_equiv` | 1.294365 | 1438 | 0.934 | 0.747 | 0.847 | 0.830 |  |
| `kde_p95_equiv` | 1.299529 | 1449 | 0.938 | 0.736 | 0.843 | 0.825 |  |
| `evt_p99_equiv` | 1.327860 | 1513 | 0.989 | 0.672 | 0.832 | 0.800 |  |
| `parametric_p99_equiv` | 1.326718 | 1511 | 0.989 | 0.672 | 0.832 | 0.800 |  |
| `kde_p99_equiv` | 1.336434 | 1533 | 0.994 | 0.649 | 0.823 | 0.785 |  |
| `evt_p999_equiv` | 1.361034 | 1590 | 1.000 | 0.615 | 0.808 | 0.762 |  |
| `parametric_p999_equiv` | 1.357966 | 1583 | 1.000 | 0.619 | 0.809 | 0.765 |  |
| `kde_p999_equiv` | 1.373035 | 1618 | 1.000 | 0.604 | 0.802 | 0.753 |  |
| `mad` | 1.485863 | 1895 | 1.000 | 0.392 | 0.696 | 0.564 |  |
| `iqr` | 1.582818 | 2150 | 1.000 | 0.215 | 0.608 | 0.354 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
