# Deployment setup: `true_int8_h8_qab_euclidean_fp32`

Combo 11 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.766728**.

At this threshold on the case 3 test split: P=0.960, R=0.272, A=0.630, F1=0.424.

Ranking quality (threshold-independent): AUC=0.7732, pAUC=0.6811.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.381544 | 0.921 | 0.442 | 0.702 | 0.597 |  |
| `percentile_same_machine_99.0` | 2.766728 | 0.960 | 0.272 | 0.630 | 0.424 | **<- active** |
| `percentile_same_machine_99.5` | 10.109355 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.381233 | 0.921 | 0.442 | 0.702 | 0.597 |  |
| `parametric_p95_equiv` | 2.733347 | 0.962 | 0.283 | 0.636 | 0.437 |  |
| `kde_p95_equiv` | 2.489393 | 0.930 | 0.400 | 0.685 | 0.559 |  |
| `evt_p99_equiv` | 3.277730 | 0.959 | 0.177 | 0.585 | 0.299 |  |
| `parametric_p99_equiv` | 3.378098 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `kde_p99_equiv` | 2.958662 | 0.970 | 0.242 | 0.617 | 0.387 |  |
| `evt_p999_equiv` | 9.868543 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 4.283211 | 0.926 | 0.094 | 0.543 | 0.171 |  |
| `kde_p999_equiv` | 11.617423 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 3.370841 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `iqr` | 4.023130 | 0.931 | 0.102 | 0.547 | 0.184 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
