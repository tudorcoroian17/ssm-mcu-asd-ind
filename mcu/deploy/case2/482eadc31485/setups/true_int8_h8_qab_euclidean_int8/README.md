# Deployment setup: `true_int8_h8_qab_euclidean_int8`

Combo 12 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.448726** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=0.994, R=0.581, A=0.789, F1=0.733.

Ranking quality (threshold-independent): AUC=0.9373, pAUC=0.8663.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.029606408) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.350638 | 2081 | 0.937 | 0.785 | 0.866 | 0.854 |  |
| `percentile_same_machine_99.0` | 1.448726 | 2394 | 0.994 | 0.581 | 0.789 | 0.733 | **<- active** |
| `percentile_same_machine_99.5` | 1.473287 | 2476 | 0.993 | 0.551 | 0.774 | 0.709 |  |
| `evt_p95_equiv` | 1.348331 | 2074 | 0.933 | 0.789 | 0.866 | 0.855 |  |
| `parametric_p95_equiv` | 1.341139 | 2052 | 0.922 | 0.800 | 0.866 | 0.857 |  |
| `kde_p95_equiv` | 1.353936 | 2091 | 0.937 | 0.781 | 0.864 | 0.852 |  |
| `evt_p99_equiv` | 1.449433 | 2397 | 0.994 | 0.577 | 0.787 | 0.730 |  |
| `parametric_p99_equiv` | 1.440441 | 2367 | 0.987 | 0.592 | 0.792 | 0.741 |  |
| `kde_p99_equiv` | 1.457704 | 2424 | 0.993 | 0.570 | 0.783 | 0.724 |  |
| `evt_p999_equiv` | 1.522053 | 2643 | 0.992 | 0.472 | 0.734 | 0.639 |  |
| `parametric_p999_equiv` | 1.554056 | 2755 | 0.991 | 0.411 | 0.704 | 0.581 |  |
| `kde_p999_equiv` | 1.544080 | 2720 | 0.991 | 0.438 | 0.717 | 0.607 |  |
| `mad` | 1.589249 | 2881 | 0.990 | 0.366 | 0.681 | 0.534 |  |
| `iqr` | 1.754280 | 3511 | 1.000 | 0.162 | 0.581 | 0.279 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
