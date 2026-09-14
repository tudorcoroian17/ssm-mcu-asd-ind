# Deployment setup: `true_int8_h16_euclidean_fp32`

Combo 15 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.651355**.

At this threshold on the case 3 test split: P=0.958, R=0.260, A=0.625, F1=0.409.

Ranking quality (threshold-independent): AUC=0.8480, pAUC=0.7008.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.183838 | 0.935 | 0.487 | 0.726 | 0.640 |  |
| `percentile_same_machine_99.0` | 2.651355 | 0.958 | 0.260 | 0.625 | 0.409 | **<- active** |
| `percentile_same_machine_99.5` | 5.655834 | 0.750 | 0.023 | 0.508 | 0.044 |  |
| `evt_p95_equiv` | 2.181794 | 0.928 | 0.487 | 0.725 | 0.639 |  |
| `parametric_p95_equiv` | 2.359739 | 0.934 | 0.374 | 0.674 | 0.534 |  |
| `kde_p95_equiv` | 2.245270 | 0.933 | 0.475 | 0.721 | 0.630 |  |
| `evt_p99_equiv` | 3.018648 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `parametric_p99_equiv` | 2.961663 | 0.957 | 0.170 | 0.581 | 0.288 |  |
| `kde_p99_equiv` | 2.742603 | 0.949 | 0.211 | 0.600 | 0.346 |  |
| `evt_p999_equiv` | 7.761528 | 0.667 | 0.015 | 0.504 | 0.030 |  |
| `parametric_p999_equiv` | 3.820642 | 0.935 | 0.109 | 0.551 | 0.196 |  |
| `kde_p999_equiv` | 8.706445 | 1.000 | 0.004 | 0.502 | 0.008 |  |
| `mad` | 3.126776 | 0.947 | 0.136 | 0.564 | 0.238 |  |
| `iqr` | 3.710185 | 0.935 | 0.109 | 0.551 | 0.196 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
