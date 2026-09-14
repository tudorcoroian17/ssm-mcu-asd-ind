# Deployment setup: `true_int8_h16_qabar_euclidean_int8`

Combo 24 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.768772** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.957, R=0.253, A=0.621, F1=0.400.

Ranking quality (threshold-independent): AUC=0.7353, pAUC=0.6657.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.029621847) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.410282 | 6621 | 0.909 | 0.415 | 0.687 | 0.570 |  |
| `percentile_same_machine_99.0` | 2.768772 | 8737 | 0.957 | 0.253 | 0.621 | 0.400 | **<- active** |
| `percentile_same_machine_99.5` | 10.358150 | 122276 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.388623 | 6502 | 0.903 | 0.423 | 0.689 | 0.576 |  |
| `parametric_p95_equiv` | 2.760273 | 8683 | 0.957 | 0.253 | 0.621 | 0.400 |  |
| `kde_p95_equiv` | 2.502865 | 7139 | 0.928 | 0.340 | 0.657 | 0.497 |  |
| `evt_p99_equiv` | 3.293151 | 12359 | 0.953 | 0.155 | 0.574 | 0.266 |  |
| `parametric_p99_equiv` | 3.395058 | 13136 | 0.949 | 0.140 | 0.566 | 0.243 |  |
| `kde_p99_equiv` | 2.968629 | 10044 | 0.964 | 0.204 | 0.598 | 0.336 |  |
| `evt_p999_equiv` | 10.327787 | 121560 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 4.281642 | 20893 | 0.926 | 0.094 | 0.543 | 0.171 |  |
| `kde_p999_equiv` | 11.905272 | 161530 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 3.297827 | 12395 | 0.953 | 0.155 | 0.574 | 0.266 |  |
| `iqr` | 3.948786 | 17771 | 0.931 | 0.102 | 0.547 | 0.184 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
