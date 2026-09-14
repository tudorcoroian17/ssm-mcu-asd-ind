# Deployment setup: `true_int8_h8_knn16_int8`

Combo 14 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: true int8 arithmetic, h storage = int8
- Head: knn_clustered_16 head (int32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **2.092469** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 2 test split: P=0.940, R=0.177, A=0.583, F1=0.298.

Ranking quality (threshold-independent): AUC=0.7545, pAUC=0.6486.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0305022799) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.950109 | 4087 | 0.894 | 0.351 | 0.655 | 0.504 |  |
| `percentile_same_machine_99.0` | 2.092469 | 4706 | 0.940 | 0.177 | 0.583 | 0.298 | **<- active** |
| `percentile_same_machine_99.5` | 2.154066 | 4987 | 0.943 | 0.125 | 0.558 | 0.220 |  |
| `evt_p95_equiv` | 1.953868 | 4103 | 0.891 | 0.340 | 0.649 | 0.492 |  |
| `parametric_p95_equiv` | 1.975055 | 4193 | 0.910 | 0.306 | 0.638 | 0.458 |  |
| `kde_p95_equiv` | 1.966516 | 4157 | 0.883 | 0.313 | 0.636 | 0.462 |  |
| `evt_p99_equiv` | 2.102731 | 4752 | 0.959 | 0.177 | 0.585 | 0.299 |  |
| `parametric_p99_equiv` | 2.132662 | 4889 | 0.950 | 0.143 | 0.568 | 0.249 |  |
| `kde_p99_equiv` | 2.116630 | 4815 | 0.956 | 0.162 | 0.577 | 0.277 |  |
| `evt_p999_equiv` | 2.263016 | 5504 | 0.833 | 0.038 | 0.515 | 0.072 |  |
| `parametric_p999_equiv` | 2.313196 | 5751 | 0.778 | 0.026 | 0.509 | 0.051 |  |
| `kde_p999_equiv` | 2.286489 | 5619 | 0.800 | 0.030 | 0.511 | 0.058 |  |
| `mad` | 2.497201 | 6703 | 1.000 | 0.008 | 0.504 | 0.015 |  |
| `iqr` | 2.806066 | 8463 | 1.000 | 0.004 | 0.502 | 0.008 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
