# Deployment setup: `true_int8_h8_knn16_int8`

Combo 14 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.802419** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 1 test split: P=0.976, R=0.458, A=0.723, F1=0.624.

Ranking quality (threshold-independent): AUC=0.8347, pAUC=0.7930.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0296475155) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.541989 | 2705 | 0.927 | 0.625 | 0.788 | 0.747 |  |
| `percentile_same_machine_99.0` | 1.802419 | 3696 | 0.976 | 0.458 | 0.723 | 0.624 | **<- active** |
| `percentile_same_machine_99.5` | 2.024164 | 4661 | 0.988 | 0.303 | 0.650 | 0.464 |  |
| `evt_p95_equiv` | 1.541254 | 2703 | 0.927 | 0.625 | 0.788 | 0.747 |  |
| `parametric_p95_equiv` | 1.517837 | 2621 | 0.916 | 0.663 | 0.801 | 0.769 |  |
| `kde_p95_equiv` | 1.546338 | 2720 | 0.927 | 0.625 | 0.788 | 0.747 |  |
| `evt_p99_equiv` | 1.834035 | 3827 | 0.983 | 0.436 | 0.714 | 0.604 |  |
| `parametric_p99_equiv` | 1.630118 | 3023 | 0.968 | 0.572 | 0.777 | 0.719 |  |
| `kde_p99_equiv` | 1.815013 | 3748 | 0.984 | 0.455 | 0.723 | 0.622 |  |
| `evt_p999_equiv` | 2.410284 | 6609 | 0.976 | 0.152 | 0.574 | 0.262 |  |
| `parametric_p999_equiv` | 1.765876 | 3548 | 0.970 | 0.489 | 0.737 | 0.650 |  |
| `kde_p999_equiv` | 2.431869 | 6728 | 0.975 | 0.148 | 0.572 | 0.257 |  |
| `mad` | 1.623051 | 2997 | 0.968 | 0.580 | 0.780 | 0.725 |  |
| `iqr` | 1.766387 | 3550 | 0.970 | 0.489 | 0.737 | 0.650 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
