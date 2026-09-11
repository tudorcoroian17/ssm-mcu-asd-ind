# Deployment setup: `true_int8_h16_euclidean_int8`

Combo 16 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **4.153618** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8906, pAUC=0.7422.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0296475155) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.183226 | 5423 | 0.941 | 0.670 | 0.814 | 0.783 |  |
| `percentile_same_machine_99.0` | 4.153618 | 19628 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 6.870059 | 53696 | 0.714 | 0.038 | 0.511 | 0.072 |  |
| `evt_p95_equiv` | 2.232368 | 5670 | 0.944 | 0.644 | 0.803 | 0.766 |  |
| `parametric_p95_equiv` | 2.368554 | 6382 | 0.951 | 0.587 | 0.778 | 0.726 |  |
| `kde_p95_equiv` | 2.271487 | 5870 | 0.949 | 0.633 | 0.799 | 0.759 |  |
| `evt_p99_equiv` | 4.466625 | 22698 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.723066 | 8436 | 0.907 | 0.295 | 0.633 | 0.446 |  |
| `kde_p99_equiv` | 6.401086 | 46616 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 38.366952 | 1674704 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.183878 | 11533 | 0.812 | 0.098 | 0.538 | 0.176 |  |
| `kde_p999_equiv` | 7.097329 | 57308 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.080420 | 4924 | 0.914 | 0.723 | 0.828 | 0.808 |  |
| `iqr` | 2.261664 | 5819 | 0.949 | 0.633 | 0.799 | 0.759 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
