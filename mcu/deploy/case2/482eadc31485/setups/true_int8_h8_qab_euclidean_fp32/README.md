# Deployment setup: `true_int8_h8_qab_euclidean_fp32`

Combo 11 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
- Backbone: true int8 arithmetic, h storage = int8
- Head: euclidean head (fp32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.448726**.

At this threshold on the case 2 test split: P=0.994, R=0.581, A=0.789, F1=0.733.

Ranking quality (threshold-independent): AUC=0.9373, pAUC=0.8663.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.350638 | 0.937 | 0.785 | 0.866 | 0.854 |  |
| `percentile_same_machine_99.0` | 1.448726 | 0.994 | 0.581 | 0.789 | 0.733 | **<- active** |
| `percentile_same_machine_99.5` | 1.473287 | 0.993 | 0.551 | 0.774 | 0.709 |  |
| `evt_p95_equiv` | 1.348331 | 0.933 | 0.789 | 0.866 | 0.855 |  |
| `parametric_p95_equiv` | 1.341139 | 0.922 | 0.800 | 0.866 | 0.857 |  |
| `kde_p95_equiv` | 1.353936 | 0.937 | 0.781 | 0.864 | 0.852 |  |
| `evt_p99_equiv` | 1.449433 | 0.994 | 0.577 | 0.787 | 0.730 |  |
| `parametric_p99_equiv` | 1.440441 | 0.987 | 0.592 | 0.792 | 0.741 |  |
| `kde_p99_equiv` | 1.457704 | 0.993 | 0.570 | 0.783 | 0.724 |  |
| `evt_p999_equiv` | 1.522053 | 0.992 | 0.472 | 0.734 | 0.639 |  |
| `parametric_p999_equiv` | 1.554056 | 0.991 | 0.411 | 0.704 | 0.581 |  |
| `kde_p999_equiv` | 1.544080 | 0.991 | 0.438 | 0.717 | 0.607 |  |
| `mad` | 1.589249 | 0.990 | 0.366 | 0.681 | 0.534 |  |
| `iqr` | 1.754280 | 1.000 | 0.162 | 0.581 | 0.279 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
