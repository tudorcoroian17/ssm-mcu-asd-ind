# Deployment setup: `true_int8_h8_qab_knn16_int8`

Combo 14 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.156069** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 3 test split: P=0.978, R=0.989, A=0.983, F1=0.983.

Ranking quality (threshold-independent): AUC=0.9906, pAUC=0.9934.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.029621847) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.109587 | 1403 | 0.936 | 0.989 | 0.960 | 0.961 |  |
| `percentile_same_machine_99.0` | 1.156069 | 1523 | 0.978 | 0.989 | 0.983 | 0.983 | **<- active** |
| `percentile_same_machine_99.5` | 1.172997 | 1568 | 0.992 | 0.989 | 0.991 | 0.991 |  |
| `evt_p95_equiv` | 1.106131 | 1394 | 0.926 | 0.989 | 0.955 | 0.956 |  |
| `parametric_p95_equiv` | 1.183578 | 1597 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `kde_p95_equiv` | 1.123657 | 1439 | 0.953 | 0.989 | 0.970 | 0.970 |  |
| `evt_p99_equiv` | 1.180679 | 1589 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `parametric_p99_equiv` | 1.312628 | 1964 | 0.996 | 0.970 | 0.983 | 0.983 |  |
| `kde_p99_equiv` | 1.181471 | 1591 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `evt_p999_equiv` | 1.358967 | 2105 | 0.996 | 0.958 | 0.977 | 0.977 |  |
| `parametric_p999_equiv` | 1.467713 | 2455 | 1.000 | 0.857 | 0.928 | 0.923 |  |
| `kde_p999_equiv` | 2.755189 | 8651 | 1.000 | 0.098 | 0.549 | 0.179 |  |
| `mad` | 1.450618 | 2398 | 1.000 | 0.894 | 0.947 | 0.944 |  |
| `iqr` | 1.654918 | 3121 | 1.000 | 0.294 | 0.647 | 0.455 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
