# Deployment setup: `true_int8_h8_qab_knn16_fp32`

Combo 15 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
- Backbone: true int8 arithmetic, h storage = int8
- Head: knn_clustered_16 head (fp32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.492695**.

At this threshold on the case 1 test split: P=0.988, R=0.644, A=0.818, F1=0.780.

Ranking quality (threshold-independent): AUC=0.8920, pAUC=0.8587.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.374269 | 0.918 | 0.807 | 0.867 | 0.859 |  |
| `percentile_same_machine_99.0` | 1.492695 | 0.988 | 0.644 | 0.818 | 0.780 | **<- active** |
| `percentile_same_machine_99.5` | 1.681419 | 0.986 | 0.530 | 0.761 | 0.690 |  |
| `evt_p95_equiv` | 1.371733 | 0.918 | 0.811 | 0.869 | 0.861 |  |
| `parametric_p95_equiv` | 1.391721 | 0.934 | 0.803 | 0.873 | 0.864 |  |
| `kde_p95_equiv` | 1.380520 | 0.926 | 0.807 | 0.871 | 0.862 |  |
| `evt_p99_equiv` | 1.524339 | 0.988 | 0.610 | 0.801 | 0.754 |  |
| `parametric_p99_equiv` | 1.471598 | 0.978 | 0.663 | 0.824 | 0.790 |  |
| `kde_p99_equiv` | 1.510498 | 0.988 | 0.621 | 0.807 | 0.763 |  |
| `evt_p999_equiv` | 2.115759 | 0.979 | 0.178 | 0.587 | 0.301 |  |
| `parametric_p999_equiv` | 1.566594 | 0.987 | 0.587 | 0.790 | 0.736 |  |
| `kde_p999_equiv` | 2.134944 | 0.978 | 0.167 | 0.581 | 0.285 |  |
| `mad` | 1.533357 | 0.988 | 0.606 | 0.799 | 0.751 |  |
| `iqr` | 1.653484 | 0.986 | 0.549 | 0.771 | 0.706 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
