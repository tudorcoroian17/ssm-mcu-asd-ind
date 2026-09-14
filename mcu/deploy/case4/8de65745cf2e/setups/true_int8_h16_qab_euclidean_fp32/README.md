# Deployment setup: `true_int8_h16_qab_euclidean_fp32`

Combo 19 of the case 4 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 4
- Model hash: `8de65745cf2e`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.384400**.

At this threshold on the case 4 test split: P=0.981, R=1.000, A=0.991, F1=0.991.

Ranking quality (threshold-independent): AUC=1.0000, pAUC=1.0000.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.337306 | 0.927 | 1.000 | 0.960 | 0.962 |  |
| `percentile_same_machine_99.0` | 1.384400 | 0.981 | 1.000 | 0.991 | 0.991 | **<- active** |
| `percentile_same_machine_99.5` | 1.388471 | 0.985 | 1.000 | 0.992 | 0.993 |  |
| `evt_p95_equiv` | 1.338181 | 0.927 | 1.000 | 0.960 | 0.962 |  |
| `parametric_p95_equiv` | 1.341428 | 0.930 | 1.000 | 0.962 | 0.964 |  |
| `kde_p95_equiv` | 1.341201 | 0.930 | 1.000 | 0.962 | 0.964 |  |
| `evt_p99_equiv` | 1.377642 | 0.978 | 1.000 | 0.989 | 0.989 |  |
| `parametric_p99_equiv` | 1.384156 | 0.981 | 1.000 | 0.991 | 0.991 |  |
| `kde_p99_equiv` | 1.385969 | 0.981 | 1.000 | 0.991 | 0.991 |  |
| `evt_p999_equiv` | 1.419879 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `parametric_p999_equiv` | 1.432539 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `kde_p999_equiv` | 1.425855 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `mad` | 1.458260 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `iqr` | 1.530479 | 1.000 | 1.000 | 1.000 | 1.000 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
