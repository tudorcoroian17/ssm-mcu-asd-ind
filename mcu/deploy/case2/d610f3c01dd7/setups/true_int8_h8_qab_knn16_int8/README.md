# Deployment setup: `true_int8_h8_qab_knn16_int8`

Combo 14 of the case 2 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 2
- Model hash: `d610f3c01dd7`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **0.868654** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 2 test split: P=1.000, R=0.977, A=0.989, F1=0.989.

Ranking quality (threshold-independent): AUC=0.9961, pAUC=0.9974.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0341339825) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 0.823047 | 581 | 0.964 | 0.996 | 0.979 | 0.980 |  |
| `percentile_same_machine_99.0` | 0.868654 | 648 | 1.000 | 0.977 | 0.989 | 0.989 | **<- active** |
| `percentile_same_machine_99.5` | 0.896567 | 690 | 1.000 | 0.970 | 0.985 | 0.985 |  |
| `evt_p95_equiv` | 0.823065 | 581 | 0.964 | 0.996 | 0.979 | 0.980 |  |
| `parametric_p95_equiv` | 0.825791 | 585 | 0.971 | 0.996 | 0.983 | 0.983 |  |
| `kde_p95_equiv` | 0.826083 | 586 | 0.971 | 0.996 | 0.983 | 0.983 |  |
| `evt_p99_equiv` | 0.865530 | 643 | 1.000 | 0.981 | 0.991 | 0.990 |  |
| `parametric_p99_equiv` | 0.873539 | 655 | 1.000 | 0.977 | 0.989 | 0.989 |  |
| `kde_p99_equiv` | 0.872488 | 653 | 1.000 | 0.977 | 0.989 | 0.989 |  |
| `evt_p999_equiv` | 0.922601 | 731 | 1.000 | 0.958 | 0.979 | 0.979 |  |
| `parametric_p999_equiv` | 0.930349 | 743 | 1.000 | 0.958 | 0.979 | 0.979 |  |
| `kde_p999_equiv` | 0.919982 | 726 | 1.000 | 0.958 | 0.979 | 0.979 |  |
| `mad` | 0.936910 | 753 | 1.000 | 0.958 | 0.979 | 0.979 |  |
| `iqr` | 1.010591 | 877 | 1.000 | 0.909 | 0.955 | 0.953 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
