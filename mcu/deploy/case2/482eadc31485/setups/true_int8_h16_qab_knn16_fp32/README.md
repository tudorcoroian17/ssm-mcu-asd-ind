# Deployment setup: `true_int8_h16_qab_knn16_fp32`

Combo 21 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.261379**.

At this threshold on the case 2 test split: P=0.987, R=0.830, A=0.909, F1=0.902.

Ranking quality (threshold-independent): AUC=0.9493, pAUC=0.9316.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.224244 | 0.928 | 0.875 | 0.904 | 0.901 |  |
| `percentile_same_machine_99.0` | 1.261379 | 0.987 | 0.830 | 0.909 | 0.902 | **<- active** |
| `percentile_same_machine_99.5` | 1.273911 | 0.995 | 0.823 | 0.909 | 0.901 |  |
| `evt_p95_equiv` | 1.222443 | 0.929 | 0.883 | 0.908 | 0.905 |  |
| `parametric_p95_equiv` | 1.236560 | 0.962 | 0.860 | 0.913 | 0.908 |  |
| `kde_p95_equiv` | 1.228828 | 0.943 | 0.868 | 0.908 | 0.904 |  |
| `evt_p99_equiv` | 1.263711 | 0.991 | 0.830 | 0.911 | 0.903 |  |
| `parametric_p99_equiv` | 1.291368 | 0.995 | 0.804 | 0.900 | 0.889 |  |
| `kde_p99_equiv` | 1.271637 | 0.995 | 0.823 | 0.909 | 0.901 |  |
| `evt_p999_equiv` | 1.321751 | 1.000 | 0.762 | 0.881 | 0.865 |  |
| `parametric_p999_equiv` | 1.353642 | 1.000 | 0.740 | 0.870 | 0.850 |  |
| `kde_p999_equiv` | 1.331612 | 1.000 | 0.758 | 0.879 | 0.863 |  |
| `mad` | 1.394018 | 1.000 | 0.687 | 0.843 | 0.814 |  |
| `iqr` | 1.489907 | 1.000 | 0.498 | 0.749 | 0.665 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
