# Deployment setup: `true_int8_h16_knn16_int8`

Combo 18 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **2.219186** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 1 test split: P=0.986, R=0.515, A=0.754, F1=0.677.

Ranking quality (threshold-independent): AUC=0.9156, pAUC=0.8843.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0342950821) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.782093 | 2700 | 0.947 | 0.814 | 0.884 | 0.876 |  |
| `percentile_same_machine_99.0` | 2.219186 | 4187 | 0.986 | 0.515 | 0.754 | 0.677 | **<- active** |
| `percentile_same_machine_99.5` | 2.302302 | 4507 | 0.985 | 0.508 | 0.750 | 0.670 |  |
| `evt_p95_equiv` | 1.820613 | 2818 | 0.953 | 0.773 | 0.867 | 0.854 |  |
| `parametric_p95_equiv` | 1.751199 | 2607 | 0.944 | 0.826 | 0.888 | 0.881 |  |
| `kde_p95_equiv` | 1.797272 | 2746 | 0.951 | 0.803 | 0.881 | 0.871 |  |
| `evt_p99_equiv` | 2.148275 | 3924 | 0.979 | 0.538 | 0.763 | 0.694 |  |
| `parametric_p99_equiv` | 1.914782 | 3117 | 0.984 | 0.712 | 0.850 | 0.826 |  |
| `kde_p99_equiv` | 2.232143 | 4236 | 0.986 | 0.515 | 0.754 | 0.677 |  |
| `evt_p999_equiv` | 2.545686 | 5510 | 1.000 | 0.413 | 0.706 | 0.584 |  |
| `parametric_p999_equiv` | 2.102637 | 3759 | 0.980 | 0.557 | 0.773 | 0.710 |  |
| `kde_p999_equiv` | 2.389938 | 4856 | 1.000 | 0.485 | 0.742 | 0.653 |  |
| `mad` | 1.758381 | 2629 | 0.943 | 0.822 | 0.886 | 0.879 |  |
| `iqr` | 1.942320 | 3208 | 0.984 | 0.693 | 0.841 | 0.813 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
