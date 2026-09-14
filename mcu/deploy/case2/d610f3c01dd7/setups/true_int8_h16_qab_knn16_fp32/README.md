# Deployment setup: `true_int8_h16_qab_knn16_fp32`

Combo 21 of the case 2 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 2
- Model hash: `d610f3c01dd7`
- Backbone: true int8 arithmetic, h storage = int16
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **0.897841**.

At this threshold on the case 2 test split: P=0.978, R=0.989, A=0.983, F1=0.983.

Ranking quality (threshold-independent): AUC=0.9958, pAUC=0.9956.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 0.863505 | 0.956 | 0.992 | 0.974 | 0.974 |  |
| `percentile_same_machine_99.0` | 0.897841 | 0.978 | 0.989 | 0.983 | 0.983 | **<- active** |
| `percentile_same_machine_99.5` | 0.909128 | 0.992 | 0.985 | 0.989 | 0.989 |  |
| `evt_p95_equiv` | 0.863039 | 0.956 | 0.992 | 0.974 | 0.974 |  |
| `parametric_p95_equiv` | 0.866099 | 0.960 | 0.992 | 0.975 | 0.976 |  |
| `kde_p95_equiv` | 0.867470 | 0.960 | 0.992 | 0.975 | 0.976 |  |
| `evt_p99_equiv` | 0.902624 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `parametric_p99_equiv` | 0.913958 | 1.000 | 0.977 | 0.989 | 0.989 |  |
| `kde_p99_equiv` | 0.906524 | 0.992 | 0.989 | 0.991 | 0.991 |  |
| `evt_p999_equiv` | 0.944349 | 1.000 | 0.966 | 0.983 | 0.983 |  |
| `parametric_p999_equiv` | 0.968486 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `kde_p999_equiv` | 0.955159 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `mad` | 0.997635 | 1.000 | 0.955 | 0.977 | 0.977 |  |
| `iqr` | 1.094936 | 1.000 | 0.811 | 0.906 | 0.896 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
