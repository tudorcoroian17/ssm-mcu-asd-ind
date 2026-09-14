# Deployment setup: `true_int8_h8_qab_knn16_fp32`

Combo 13 of the case 4 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 4
- Model hash: `8de65745cf2e`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.380966**.

At this threshold on the case 4 test split: P=0.989, R=0.992, A=0.991, F1=0.991.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.269100 | 0.970 | 0.992 | 0.981 | 0.981 |  |
| `percentile_same_machine_99.0` | 1.380966 | 0.989 | 0.992 | 0.991 | 0.991 | **<- active** |
| `percentile_same_machine_99.5` | 1.435840 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `evt_p95_equiv` | 1.269996 | 0.970 | 0.992 | 0.981 | 0.981 |  |
| `parametric_p95_equiv` | 1.273637 | 0.970 | 0.992 | 0.981 | 0.981 |  |
| `kde_p95_equiv` | 1.275338 | 0.970 | 0.992 | 0.981 | 0.981 |  |
| `evt_p99_equiv` | 1.378974 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `parametric_p99_equiv` | 1.366961 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `kde_p99_equiv` | 1.386214 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `evt_p999_equiv` | 1.553739 | 1.000 | 0.977 | 0.989 | 0.989 |  |
| `parametric_p999_equiv` | 1.479720 | 1.000 | 0.981 | 0.991 | 0.990 |  |
| `kde_p999_equiv` | 1.555804 | 1.000 | 0.977 | 0.989 | 0.989 |  |
| `mad` | 1.482482 | 1.000 | 0.981 | 0.991 | 0.990 |  |
| `iqr` | 1.617939 | 1.000 | 0.947 | 0.974 | 0.973 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
