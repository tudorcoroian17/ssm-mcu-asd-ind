# Deployment setup: `w_all_pertensor_qab_euclidean_fp32`

Combo 5 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
- Backbone: weight-only int8 (all tensors, per-tensor)
- Head: euclidean head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.316728**.

At this threshold on the case 2 test split: P=0.996, R=0.962, A=0.979, F1=0.979.

Ranking quality (threshold-independent): AUC=0.9850, pAUC=0.9803.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.282285 | 0.934 | 0.962 | 0.947 | 0.948 |  |
| `percentile_same_machine_99.0` | 1.316728 | 0.996 | 0.962 | 0.979 | 0.979 | **<- active** |
| `percentile_same_machine_99.5` | 1.328612 | 0.996 | 0.958 | 0.977 | 0.977 |  |
| `evt_p95_equiv` | 1.281777 | 0.934 | 0.962 | 0.947 | 0.948 |  |
| `parametric_p95_equiv` | 1.291346 | 0.955 | 0.962 | 0.958 | 0.959 |  |
| `kde_p95_equiv` | 1.286211 | 0.948 | 0.962 | 0.955 | 0.955 |  |
| `evt_p99_equiv` | 1.317437 | 0.996 | 0.962 | 0.979 | 0.979 |  |
| `parametric_p99_equiv` | 1.342721 | 0.996 | 0.958 | 0.977 | 0.977 |  |
| `kde_p99_equiv` | 1.324109 | 0.996 | 0.958 | 0.977 | 0.977 |  |
| `evt_p999_equiv` | 1.347181 | 0.996 | 0.958 | 0.977 | 0.977 |  |
| `parametric_p999_equiv` | 1.401024 | 1.000 | 0.936 | 0.968 | 0.967 |  |
| `kde_p999_equiv` | 1.358678 | 1.000 | 0.958 | 0.979 | 0.979 |  |
| `mad` | 1.449662 | 1.000 | 0.894 | 0.947 | 0.944 |  |
| `iqr` | 1.545609 | 1.000 | 0.842 | 0.921 | 0.914 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
