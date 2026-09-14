# Deployment setup: `w_all_pertensor_actboundaries_qab_euclidean_fp32`

Combo 9 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.373113**.

At this threshold on the case 2 test split: P=1.000, R=0.894, A=0.947, F1=0.944.

Ranking quality (threshold-independent): AUC=0.9710, pAUC=0.9553.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.329833 | 0.964 | 0.913 | 0.940 | 0.938 |  |
| `percentile_same_machine_99.0` | 1.373113 | 1.000 | 0.894 | 0.947 | 0.944 | **<- active** |
| `percentile_same_machine_99.5` | 1.383729 | 1.000 | 0.883 | 0.942 | 0.938 |  |
| `evt_p95_equiv` | 1.332005 | 0.972 | 0.913 | 0.943 | 0.942 |  |
| `parametric_p95_equiv` | 1.339072 | 0.984 | 0.906 | 0.945 | 0.943 |  |
| `kde_p95_equiv` | 1.336396 | 0.980 | 0.909 | 0.945 | 0.943 |  |
| `evt_p99_equiv` | 1.367561 | 0.996 | 0.898 | 0.947 | 0.944 |  |
| `parametric_p99_equiv` | 1.395673 | 1.000 | 0.872 | 0.936 | 0.931 |  |
| `kde_p99_equiv` | 1.377104 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `evt_p999_equiv` | 1.406188 | 1.000 | 0.857 | 0.928 | 0.923 |  |
| `parametric_p999_equiv` | 1.459949 | 1.000 | 0.838 | 0.919 | 0.912 |  |
| `kde_p999_equiv` | 1.416690 | 1.000 | 0.857 | 0.928 | 0.923 |  |
| `mad` | 1.530182 | 1.000 | 0.796 | 0.898 | 0.887 |  |
| `iqr` | 1.644736 | 1.000 | 0.649 | 0.825 | 0.787 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
