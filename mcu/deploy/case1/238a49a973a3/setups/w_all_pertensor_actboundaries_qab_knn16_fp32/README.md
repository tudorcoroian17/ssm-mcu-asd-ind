# Deployment setup: `w_all_pertensor_actboundaries_qab_knn16_fp32`

Combo 10 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.479710**.

At this threshold on the case 1 test split: P=0.983, R=0.443, A=0.718, F1=0.611.

Ranking quality (threshold-independent): AUC=0.8257, pAUC=0.7949.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.969523 | 0.937 | 0.617 | 0.788 | 0.744 |  |
| `percentile_same_machine_99.0` | 2.479710 | 0.983 | 0.443 | 0.718 | 0.611 | **<- active** |
| `percentile_same_machine_99.5` | 2.552021 | 1.000 | 0.383 | 0.691 | 0.553 |  |
| `evt_p95_equiv` | 2.036952 | 0.963 | 0.587 | 0.782 | 0.729 |  |
| `parametric_p95_equiv` | 2.024572 | 0.952 | 0.595 | 0.782 | 0.732 |  |
| `kde_p95_equiv` | 1.989114 | 0.942 | 0.610 | 0.786 | 0.740 |  |
| `evt_p99_equiv` | 2.410403 | 0.976 | 0.466 | 0.727 | 0.631 |  |
| `parametric_p99_equiv` | 2.187182 | 0.971 | 0.504 | 0.744 | 0.663 |  |
| `kde_p99_equiv` | 2.500259 | 0.991 | 0.428 | 0.712 | 0.598 |  |
| `evt_p999_equiv` | 2.749069 | 1.000 | 0.265 | 0.633 | 0.419 |  |
| `parametric_p999_equiv` | 2.352463 | 0.969 | 0.477 | 0.731 | 0.640 |  |
| `kde_p999_equiv` | 2.663138 | 1.000 | 0.345 | 0.672 | 0.513 |  |
| `mad` | 2.008695 | 0.952 | 0.606 | 0.788 | 0.741 |  |
| `iqr` | 2.174721 | 0.971 | 0.511 | 0.748 | 0.670 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
