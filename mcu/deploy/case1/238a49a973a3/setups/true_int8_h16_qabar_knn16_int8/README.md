# Deployment setup: `true_int8_h16_qabar_knn16_int8`

Combo 28 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: true int8 arithmetic, h storage = int16
- Head: knn_clustered_16 head (int32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **2.366976** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 1 test split: P=1.000, R=0.447, A=0.723, F1=0.618.

Ranking quality (threshold-independent): AUC=0.8421, pAUC=0.8192.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0329585263) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.886141 | 3275 | 0.946 | 0.663 | 0.812 | 0.780 |  |
| `percentile_same_machine_99.0` | 2.366976 | 5158 | 1.000 | 0.447 | 0.723 | 0.618 | **<- active** |
| `percentile_same_machine_99.5` | 2.414494 | 5367 | 1.000 | 0.413 | 0.706 | 0.584 |  |
| `evt_p95_equiv` | 1.910999 | 3362 | 0.949 | 0.640 | 0.803 | 0.765 |  |
| `parametric_p95_equiv` | 1.939900 | 3464 | 0.960 | 0.636 | 0.805 | 0.765 |  |
| `kde_p95_equiv` | 1.897298 | 3314 | 0.951 | 0.655 | 0.811 | 0.776 |  |
| `evt_p99_equiv` | 2.270004 | 4744 | 0.977 | 0.477 | 0.733 | 0.641 |  |
| `parametric_p99_equiv` | 2.084807 | 4001 | 0.972 | 0.534 | 0.759 | 0.689 |  |
| `kde_p99_equiv` | 2.365536 | 5151 | 1.000 | 0.447 | 0.723 | 0.618 |  |
| `evt_p999_equiv` | 2.913121 | 7812 | 1.000 | 0.148 | 0.574 | 0.257 |  |
| `parametric_p999_equiv` | 2.231350 | 4584 | 0.977 | 0.485 | 0.737 | 0.648 |  |
| `kde_p999_equiv` | 2.519682 | 5845 | 1.000 | 0.348 | 0.674 | 0.517 |  |
| `mad` | 1.998870 | 3678 | 0.975 | 0.598 | 0.792 | 0.742 |  |
| `iqr` | 2.160669 | 4298 | 0.971 | 0.504 | 0.744 | 0.663 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
