# Deployment setup: `w_all_pertensor_qab_knn16_fp32`

Combo 6 of the case 4 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 4
- Model hash: `3660d1d10006`
- Backbone: weight-only int8 (all tensors, per-tensor)
- Head: knn_clustered_16 head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **0.948851**.

At this threshold on the case 4 test split: P=0.989, R=0.992, A=0.991, F1=0.991.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 0.928152 | 0.963 | 0.992 | 0.977 | 0.978 |  |
| `percentile_same_machine_99.0` | 0.948851 | 0.989 | 0.992 | 0.991 | 0.991 | **<- active** |
| `percentile_same_machine_99.5` | 0.956677 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `evt_p95_equiv` | 0.927535 | 0.963 | 0.992 | 0.977 | 0.978 |  |
| `parametric_p95_equiv` | 0.930308 | 0.970 | 0.992 | 0.981 | 0.981 |  |
| `kde_p95_equiv` | 0.931408 | 0.970 | 0.992 | 0.981 | 0.981 |  |
| `evt_p99_equiv` | 0.947477 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `parametric_p99_equiv` | 0.945957 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `kde_p99_equiv` | 0.953385 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `evt_p999_equiv` | 0.969443 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `parametric_p999_equiv` | 0.960955 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `kde_p999_equiv` | 0.976665 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `mad` | 0.984897 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `iqr` | 1.020146 | 1.000 | 0.992 | 0.996 | 0.996 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
