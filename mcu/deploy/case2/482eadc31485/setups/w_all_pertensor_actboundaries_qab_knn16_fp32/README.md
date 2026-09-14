# Deployment setup: `w_all_pertensor_actboundaries_qab_knn16_fp32`

Combo 10 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.387987**.

At this threshold on the case 2 test split: P=0.996, R=0.845, A=0.921, F1=0.914.

Ranking quality (threshold-independent): AUC=0.9306, pAUC=0.9381.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.335220 | 0.971 | 0.875 | 0.925 | 0.921 |  |
| `percentile_same_machine_99.0` | 1.387987 | 0.996 | 0.845 | 0.921 | 0.914 | **<- active** |
| `percentile_same_machine_99.5` | 1.405959 | 1.000 | 0.838 | 0.919 | 0.912 |  |
| `evt_p95_equiv` | 1.339794 | 0.979 | 0.875 | 0.928 | 0.924 |  |
| `parametric_p95_equiv` | 1.353416 | 0.987 | 0.868 | 0.928 | 0.924 |  |
| `kde_p95_equiv` | 1.344944 | 0.979 | 0.872 | 0.926 | 0.922 |  |
| `evt_p99_equiv` | 1.383217 | 0.996 | 0.849 | 0.923 | 0.916 |  |
| `parametric_p99_equiv` | 1.414983 | 1.000 | 0.823 | 0.911 | 0.903 |  |
| `kde_p99_equiv` | 1.393857 | 1.000 | 0.842 | 0.921 | 0.914 |  |
| `evt_p999_equiv` | 1.425799 | 1.000 | 0.819 | 0.909 | 0.900 |  |
| `parametric_p999_equiv` | 1.484959 | 1.000 | 0.747 | 0.874 | 0.855 |  |
| `kde_p999_equiv` | 1.436559 | 1.000 | 0.804 | 0.902 | 0.891 |  |
| `mad` | 1.564071 | 1.000 | 0.619 | 0.809 | 0.765 |  |
| `iqr` | 1.686851 | 1.000 | 0.415 | 0.708 | 0.587 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
