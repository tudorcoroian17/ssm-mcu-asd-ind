# Deployment setup: `true_int8_h16_euclidean_int8`

Combo 16 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **3.708059** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 1 test split: P=0.846, R=0.083, A=0.534, F1=0.152.

Ranking quality (threshold-independent): AUC=0.9465, pAUC=0.7659.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0342950821) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.335197 | 4636 | 0.939 | 0.640 | 0.799 | 0.761 |  |
| `percentile_same_machine_99.0` | 3.708059 | 11690 | 0.846 | 0.083 | 0.534 | 0.152 | **<- active** |
| `percentile_same_machine_99.5` | 7.931917 | 53492 | 0.818 | 0.034 | 0.513 | 0.065 |  |
| `evt_p95_equiv` | 2.385586 | 4839 | 0.949 | 0.636 | 0.801 | 0.762 |  |
| `parametric_p95_equiv` | 2.334694 | 4634 | 0.939 | 0.640 | 0.799 | 0.761 |  |
| `kde_p95_equiv` | 2.443439 | 5076 | 0.952 | 0.595 | 0.782 | 0.732 |  |
| `evt_p99_equiv` | 3.860404 | 12671 | 0.826 | 0.072 | 0.528 | 0.132 |  |
| `parametric_p99_equiv` | 2.745642 | 6409 | 0.947 | 0.470 | 0.722 | 0.628 |  |
| `kde_p99_equiv` | 7.366091 | 46133 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 18.612660 | 294546 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.292839 | 9219 | 0.939 | 0.235 | 0.610 | 0.376 |  |
| `kde_p999_equiv` | 8.172484 | 56786 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.973276 | 3311 | 0.878 | 0.924 | 0.898 | 0.900 |  |
| `iqr` | 2.210246 | 4154 | 0.888 | 0.750 | 0.828 | 0.813 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
