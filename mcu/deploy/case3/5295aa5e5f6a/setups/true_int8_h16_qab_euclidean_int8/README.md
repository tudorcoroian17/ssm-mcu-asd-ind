# Deployment setup: `true_int8_h16_qab_euclidean_int8`

Combo 20 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.464732** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.984, R=0.468, A=0.730, F1=0.634.

Ranking quality (threshold-independent): AUC=0.8379, pAUC=0.7589.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0347989075) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.280721 | 4295 | 0.928 | 0.585 | 0.770 | 0.718 |  |
| `percentile_same_machine_99.0` | 2.464732 | 5017 | 0.984 | 0.468 | 0.730 | 0.634 | **<- active** |
| `percentile_same_machine_99.5` | 8.355776 | 57656 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.261962 | 4225 | 0.924 | 0.592 | 0.772 | 0.722 |  |
| `parametric_p95_equiv` | 2.603930 | 5599 | 0.984 | 0.460 | 0.726 | 0.627 |  |
| `kde_p95_equiv` | 2.370900 | 4642 | 0.973 | 0.540 | 0.762 | 0.694 |  |
| `evt_p99_equiv` | 2.805220 | 6498 | 0.977 | 0.325 | 0.658 | 0.487 |  |
| `parametric_p99_equiv` | 3.056741 | 7716 | 0.973 | 0.268 | 0.630 | 0.420 |  |
| `kde_p99_equiv` | 2.658505 | 5836 | 0.983 | 0.445 | 0.719 | 0.613 |  |
| `evt_p999_equiv` | 8.485435 | 59459 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.658523 | 11053 | 0.957 | 0.166 | 0.579 | 0.283 |  |
| `kde_p999_equiv` | 10.389089 | 89130 | 1.000 | 0.008 | 0.504 | 0.015 |  |
| `mad` | 2.981468 | 7341 | 0.973 | 0.275 | 0.634 | 0.429 |  |
| `iqr` | 3.436135 | 9750 | 0.966 | 0.211 | 0.602 | 0.347 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
