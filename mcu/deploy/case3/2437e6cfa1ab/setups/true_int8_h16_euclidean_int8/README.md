# Deployment setup: `true_int8_h16_euclidean_int8`

Combo 16 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.990196** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.989, R=0.989, A=0.989, F1=0.989.

Ranking quality (threshold-independent): AUC=0.9927, pAUC=0.9617.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0315041016) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.736547 | 3038 | 0.957 | 1.000 | 0.977 | 0.978 |  |
| `percentile_same_machine_99.0` | 1.990196 | 3991 | 0.989 | 0.989 | 0.989 | 0.989 | **<- active** |
| `percentile_same_machine_99.5` | 3.842993 | 14880 | 0.939 | 0.117 | 0.555 | 0.208 |  |
| `evt_p95_equiv` | 1.737043 | 3040 | 0.957 | 1.000 | 0.977 | 0.978 |  |
| `parametric_p95_equiv` | 1.916445 | 3700 | 0.989 | 1.000 | 0.994 | 0.994 |  |
| `kde_p95_equiv` | 1.779144 | 3189 | 0.974 | 1.000 | 0.987 | 0.987 |  |
| `evt_p99_equiv` | 2.103535 | 4458 | 0.989 | 0.981 | 0.985 | 0.985 |  |
| `parametric_p99_equiv` | 2.136887 | 4601 | 0.989 | 0.977 | 0.983 | 0.983 |  |
| `kde_p99_equiv` | 2.029913 | 4152 | 0.989 | 0.981 | 0.985 | 0.985 |  |
| `evt_p999_equiv` | 4.967131 | 24859 | 1.000 | 0.038 | 0.519 | 0.073 |  |
| `parametric_p999_equiv` | 2.414257 | 5873 | 0.990 | 0.781 | 0.887 | 0.873 |  |
| `kde_p999_equiv` | 4.898775 | 24179 | 1.000 | 0.038 | 0.519 | 0.073 |  |
| `mad` | 2.086650 | 4387 | 0.989 | 0.981 | 0.985 | 0.985 |  |
| `iqr` | 2.372222 | 5670 | 0.991 | 0.860 | 0.926 | 0.921 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
