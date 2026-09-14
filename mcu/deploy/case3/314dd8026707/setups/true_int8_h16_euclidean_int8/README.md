# Deployment setup: `true_int8_h16_euclidean_int8`

Combo 16 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.651355** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.958, R=0.260, A=0.625, F1=0.409.

Ranking quality (threshold-independent): AUC=0.8480, pAUC=0.7008.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0335061081) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.183838 | 4248 | 0.935 | 0.487 | 0.726 | 0.640 |  |
| `percentile_same_machine_99.0` | 2.651355 | 6262 | 0.958 | 0.260 | 0.625 | 0.409 | **<- active** |
| `percentile_same_machine_99.5` | 5.655834 | 28493 | 0.750 | 0.023 | 0.508 | 0.044 |  |
| `evt_p95_equiv` | 2.181794 | 4240 | 0.928 | 0.487 | 0.725 | 0.639 |  |
| `parametric_p95_equiv` | 2.359739 | 4960 | 0.934 | 0.374 | 0.674 | 0.534 |  |
| `kde_p95_equiv` | 2.245270 | 4490 | 0.933 | 0.475 | 0.721 | 0.630 |  |
| `evt_p99_equiv` | 3.018648 | 8117 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `parametric_p99_equiv` | 2.961663 | 7813 | 0.957 | 0.170 | 0.581 | 0.288 |  |
| `kde_p99_equiv` | 2.742603 | 6700 | 0.949 | 0.211 | 0.600 | 0.346 |  |
| `evt_p999_equiv` | 7.761528 | 53659 | 0.667 | 0.015 | 0.504 | 0.030 |  |
| `parametric_p999_equiv` | 3.820642 | 13002 | 0.935 | 0.109 | 0.551 | 0.196 |  |
| `kde_p999_equiv` | 8.706445 | 67520 | 1.000 | 0.004 | 0.502 | 0.008 |  |
| `mad` | 3.126776 | 8709 | 0.947 | 0.136 | 0.564 | 0.238 |  |
| `iqr` | 3.710185 | 12261 | 0.935 | 0.109 | 0.551 | 0.196 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
