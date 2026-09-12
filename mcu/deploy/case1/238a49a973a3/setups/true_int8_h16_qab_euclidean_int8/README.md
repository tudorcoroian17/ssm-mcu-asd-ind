# Deployment setup: `true_int8_h16_qab_euclidean_int8`

Combo 22 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: true int8 arithmetic, h storage = int16
- Head: euclidean head (int32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **4.239431** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8786, pAUC=0.6821.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0329585263) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.596056 | 6204 | 0.908 | 0.451 | 0.703 | 0.603 |  |
| `percentile_same_machine_99.0` | 4.239431 | 16545 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 9.538887 | 83764 | 0.800 | 0.045 | 0.517 | 0.086 |  |
| `evt_p95_equiv` | 2.671872 | 6572 | 0.918 | 0.383 | 0.674 | 0.540 |  |
| `parametric_p95_equiv` | 2.519309 | 5843 | 0.903 | 0.492 | 0.720 | 0.637 |  |
| `kde_p95_equiv` | 2.696152 | 6692 | 0.914 | 0.364 | 0.665 | 0.520 |  |
| `evt_p99_equiv` | 4.583785 | 19343 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.997711 | 8273 | 0.903 | 0.246 | 0.610 | 0.387 |  |
| `kde_p99_equiv` | 8.875286 | 72515 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 14.003488 | 180525 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.642728 | 12216 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 9.854139 | 89393 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.161258 | 4300 | 0.872 | 0.697 | 0.797 | 0.775 |  |
| `iqr` | 2.383744 | 5231 | 0.863 | 0.595 | 0.750 | 0.704 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
