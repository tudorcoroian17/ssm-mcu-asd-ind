# Deployment setup: `true_int8_h16_qab_euclidean_int8`

Combo 22 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **3.730269** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.9254, pAUC=0.8117.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0339353216) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.692986 | 2489 | 0.946 | 0.803 | 0.879 | 0.869 |  |
| `percentile_same_machine_99.0` | 3.730269 | 12083 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 10.985322 | 104790 | 0.833 | 0.038 | 0.515 | 0.072 |  |
| `evt_p95_equiv` | 1.720512 | 2570 | 0.953 | 0.777 | 0.869 | 0.856 |  |
| `parametric_p95_equiv` | 2.022816 | 3553 | 0.948 | 0.481 | 0.727 | 0.638 |  |
| `kde_p95_equiv` | 1.903099 | 3145 | 0.957 | 0.591 | 0.782 | 0.731 |  |
| `evt_p99_equiv` | 3.396567 | 10018 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.398828 | 4997 | 0.925 | 0.186 | 0.585 | 0.309 |  |
| `kde_p99_equiv` | 10.251797 | 91263 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 36.255489 | 1141415 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 2.903977 | 7323 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 11.321511 | 111302 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.757856 | 2683 | 0.961 | 0.739 | 0.854 | 0.835 |  |
| `iqr` | 1.919793 | 3200 | 0.957 | 0.583 | 0.778 | 0.725 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
