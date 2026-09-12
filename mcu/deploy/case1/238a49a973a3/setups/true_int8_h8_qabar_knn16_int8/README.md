# Deployment setup: `true_int8_h8_qabar_knn16_int8`

Combo 20 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **2.519723** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 1 test split: P=0.991, R=0.436, A=0.716, F1=0.605.

Ranking quality (threshold-independent): AUC=0.8586, pAUC=0.8075.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0329585263) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.207395 | 4486 | 0.948 | 0.621 | 0.794 | 0.751 |  |
| `percentile_same_machine_99.0` | 2.519723 | 5845 | 0.991 | 0.436 | 0.716 | 0.605 | **<- active** |
| `percentile_same_machine_99.5` | 2.582647 | 6140 | 0.990 | 0.390 | 0.693 | 0.560 |  |
| `evt_p95_equiv` | 2.232509 | 4588 | 0.953 | 0.610 | 0.790 | 0.744 |  |
| `parametric_p95_equiv` | 2.076767 | 3970 | 0.903 | 0.705 | 0.814 | 0.791 |  |
| `kde_p95_equiv` | 2.215878 | 4520 | 0.953 | 0.621 | 0.795 | 0.752 |  |
| `evt_p99_equiv` | 2.536265 | 5922 | 0.991 | 0.424 | 0.710 | 0.594 |  |
| `parametric_p99_equiv` | 2.275756 | 4768 | 0.957 | 0.591 | 0.782 | 0.731 |  |
| `kde_p99_equiv` | 2.537608 | 5928 | 0.991 | 0.424 | 0.710 | 0.594 |  |
| `evt_p999_equiv` | 2.649045 | 6460 | 1.000 | 0.345 | 0.672 | 0.513 |  |
| `parametric_p999_equiv` | 2.521551 | 5853 | 0.991 | 0.436 | 0.716 | 0.605 |  |
| `kde_p999_equiv` | 2.690292 | 6663 | 1.000 | 0.311 | 0.655 | 0.474 |  |
| `mad` | 2.162625 | 4306 | 0.925 | 0.655 | 0.801 | 0.767 |  |
| `iqr` | 2.341633 | 5048 | 0.973 | 0.549 | 0.767 | 0.702 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
