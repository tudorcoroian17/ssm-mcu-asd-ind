# Deployment setup: `true_int8_h16_qab_euclidean_int8`

Combo 20 of the case 2 deployment matrix.

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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.328676** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=0.973, R=0.675, A=0.828, F1=0.797.

Ranking quality (threshold-independent): AUC=0.8715, pAUC=0.8531.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0341339825) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.299290 | 1449 | 0.930 | 0.755 | 0.849 | 0.833 |  |
| `percentile_same_machine_99.0` | 1.328676 | 1515 | 0.973 | 0.675 | 0.828 | 0.797 | **<- active** |
| `percentile_same_machine_99.5` | 1.340497 | 1542 | 0.983 | 0.649 | 0.819 | 0.782 |  |
| `evt_p95_equiv` | 1.299279 | 1449 | 0.930 | 0.755 | 0.849 | 0.833 |  |
| `parametric_p95_equiv` | 1.299197 | 1449 | 0.930 | 0.755 | 0.849 | 0.833 |  |
| `kde_p95_equiv` | 1.305160 | 1462 | 0.933 | 0.736 | 0.842 | 0.823 |  |
| `evt_p99_equiv` | 1.335642 | 1531 | 0.978 | 0.657 | 0.821 | 0.786 |  |
| `parametric_p99_equiv` | 1.332216 | 1523 | 0.978 | 0.664 | 0.825 | 0.791 |  |
| `kde_p99_equiv` | 1.342485 | 1547 | 0.988 | 0.645 | 0.819 | 0.781 |  |
| `evt_p999_equiv` | 1.379665 | 1634 | 1.000 | 0.589 | 0.794 | 0.741 |  |
| `parametric_p999_equiv` | 1.364120 | 1597 | 1.000 | 0.615 | 0.808 | 0.762 |  |
| `kde_p999_equiv` | 1.394542 | 1669 | 1.000 | 0.574 | 0.787 | 0.729 |  |
| `mad` | 1.470381 | 1856 | 1.000 | 0.430 | 0.715 | 0.602 |  |
| `iqr` | 1.570313 | 2116 | 1.000 | 0.234 | 0.617 | 0.379 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
