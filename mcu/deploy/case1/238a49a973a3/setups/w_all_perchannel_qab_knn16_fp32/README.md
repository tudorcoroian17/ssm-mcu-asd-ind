# Deployment setup: `w_all_perchannel_qab_knn16_fp32`

Combo 2 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.461262**.

At this threshold on the case 1 test split: P=0.984, R=0.462, A=0.727, F1=0.629.

Ranking quality (threshold-independent): AUC=0.8429, pAUC=0.8163.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.936059 | 0.945 | 0.655 | 0.809 | 0.774 |  |
| `percentile_same_machine_99.0` | 2.461262 | 0.984 | 0.462 | 0.727 | 0.629 | **<- active** |
| `percentile_same_machine_99.5` | 2.517714 | 0.992 | 0.443 | 0.720 | 0.613 |  |
| `evt_p95_equiv` | 1.985659 | 0.960 | 0.629 | 0.801 | 0.760 |  |
| `parametric_p95_equiv` | 2.003756 | 0.964 | 0.617 | 0.797 | 0.753 |  |
| `kde_p95_equiv` | 1.955933 | 0.945 | 0.648 | 0.805 | 0.769 |  |
| `evt_p99_equiv` | 2.373351 | 0.977 | 0.477 | 0.733 | 0.641 |  |
| `parametric_p99_equiv` | 2.163623 | 0.972 | 0.534 | 0.759 | 0.689 |  |
| `kde_p99_equiv` | 2.472991 | 0.992 | 0.462 | 0.729 | 0.630 |  |
| `evt_p999_equiv` | 2.919491 | 1.000 | 0.220 | 0.610 | 0.360 |  |
| `parametric_p999_equiv` | 2.326039 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `kde_p999_equiv` | 2.636317 | 1.000 | 0.367 | 0.684 | 0.537 |  |
| `mad` | 1.994252 | 0.965 | 0.629 | 0.803 | 0.761 |  |
| `iqr` | 2.155517 | 0.973 | 0.538 | 0.761 | 0.693 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
