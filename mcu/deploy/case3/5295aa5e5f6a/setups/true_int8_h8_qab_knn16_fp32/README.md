# Deployment setup: `true_int8_h8_qab_knn16_fp32`

Combo 13 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.049972**.

At this threshold on the case 3 test split: P=0.970, R=0.974, A=0.972, F1=0.972.

Ranking quality (threshold-independent): AUC=0.9959, pAUC=0.9838.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 0.941891 | 0.942 | 0.989 | 0.964 | 0.965 |  |
| `percentile_same_machine_99.0` | 1.049972 | 0.970 | 0.974 | 0.972 | 0.972 | **<- active** |
| `percentile_same_machine_99.5` | 1.093134 | 0.981 | 0.970 | 0.975 | 0.975 |  |
| `evt_p95_equiv` | 0.939649 | 0.942 | 0.989 | 0.964 | 0.965 |  |
| `parametric_p95_equiv` | 0.949824 | 0.949 | 0.989 | 0.968 | 0.969 |  |
| `kde_p95_equiv` | 0.947669 | 0.949 | 0.989 | 0.968 | 0.969 |  |
| `evt_p99_equiv` | 1.079453 | 0.970 | 0.974 | 0.972 | 0.972 |  |
| `parametric_p99_equiv` | 1.029631 | 0.966 | 0.974 | 0.970 | 0.970 |  |
| `kde_p99_equiv` | 1.056884 | 0.970 | 0.974 | 0.972 | 0.972 |  |
| `evt_p999_equiv` | 1.395228 | 1.000 | 0.336 | 0.668 | 0.503 |  |
| `parametric_p999_equiv` | 1.127085 | 0.984 | 0.928 | 0.957 | 0.955 |  |
| `kde_p999_equiv` | 1.966950 | 1.000 | 0.113 | 0.557 | 0.203 |  |
| `mad` | 1.073288 | 0.970 | 0.974 | 0.972 | 0.972 |  |
| `iqr` | 1.175277 | 0.996 | 0.838 | 0.917 | 0.910 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
