# Deployment setup: `true_int8_h16_qab_euclidean_fp32`

Combo 19 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
- Backbone: true int8 arithmetic, h storage = int16
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.362422**.

At this threshold on the case 2 test split: P=0.992, R=0.891, A=0.942, F1=0.938.

Ranking quality (threshold-independent): AUC=0.9573, pAUC=0.9547.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.324253 | 0.949 | 0.917 | 0.934 | 0.933 |  |
| `percentile_same_machine_99.0` | 1.362422 | 0.992 | 0.891 | 0.942 | 0.938 | **<- active** |
| `percentile_same_machine_99.5` | 1.379495 | 0.996 | 0.887 | 0.942 | 0.938 |  |
| `evt_p95_equiv` | 1.325137 | 0.949 | 0.917 | 0.934 | 0.933 |  |
| `parametric_p95_equiv` | 1.336465 | 0.976 | 0.909 | 0.943 | 0.941 |  |
| `kde_p95_equiv` | 1.329862 | 0.964 | 0.909 | 0.938 | 0.936 |  |
| `evt_p99_equiv` | 1.362255 | 0.992 | 0.891 | 0.942 | 0.938 |  |
| `parametric_p99_equiv` | 1.386701 | 0.996 | 0.887 | 0.942 | 0.938 |  |
| `kde_p99_equiv` | 1.370287 | 0.992 | 0.887 | 0.940 | 0.936 |  |
| `evt_p999_equiv` | 1.405211 | 1.000 | 0.887 | 0.943 | 0.940 |  |
| `parametric_p999_equiv` | 1.443679 | 1.000 | 0.857 | 0.928 | 0.923 |  |
| `kde_p999_equiv` | 1.415907 | 1.000 | 0.883 | 0.942 | 0.938 |  |
| `mad` | 1.475396 | 1.000 | 0.845 | 0.923 | 0.916 |  |
| `iqr` | 1.563073 | 1.000 | 0.713 | 0.857 | 0.833 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
