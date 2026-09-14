# Deployment setup: `true_int8_h16_knn16_int8`

Combo 18 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: true int8 arithmetic, h storage = int16
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.919837** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 2 test split: P=1.000, R=0.574, A=0.787, F1=0.729.

Ranking quality (threshold-independent): AUC=0.7963, pAUC=0.8187.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0305022799) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.868680 | 3753 | 0.983 | 0.638 | 0.813 | 0.773 |  |
| `percentile_same_machine_99.0` | 1.919837 | 3962 | 1.000 | 0.574 | 0.787 | 0.729 | **<- active** |
| `percentile_same_machine_99.5` | 1.936192 | 4029 | 1.000 | 0.551 | 0.775 | 0.710 |  |
| `evt_p95_equiv` | 1.867471 | 3748 | 0.983 | 0.638 | 0.813 | 0.773 |  |
| `parametric_p95_equiv` | 1.861552 | 3725 | 0.971 | 0.638 | 0.809 | 0.770 |  |
| `kde_p95_equiv` | 1.877127 | 3787 | 0.982 | 0.623 | 0.806 | 0.762 |  |
| `evt_p99_equiv` | 1.922720 | 3973 | 1.000 | 0.570 | 0.785 | 0.726 |  |
| `parametric_p99_equiv` | 1.920400 | 3964 | 1.000 | 0.570 | 0.785 | 0.726 |  |
| `kde_p99_equiv` | 1.935200 | 4025 | 1.000 | 0.555 | 0.777 | 0.714 |  |
| `evt_p999_equiv` | 1.966001 | 4154 | 1.000 | 0.521 | 0.760 | 0.685 |  |
| `parametric_p999_equiv` | 1.977596 | 4203 | 1.000 | 0.502 | 0.751 | 0.668 |  |
| `kde_p999_equiv` | 1.991100 | 4261 | 1.000 | 0.475 | 0.738 | 0.645 |  |
| `mad` | 2.182358 | 5119 | 1.000 | 0.223 | 0.611 | 0.364 |  |
| `iqr` | 2.356718 | 5970 | 1.000 | 0.079 | 0.540 | 0.147 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
