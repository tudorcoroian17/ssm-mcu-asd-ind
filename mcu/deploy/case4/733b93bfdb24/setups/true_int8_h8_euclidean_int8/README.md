# Deployment setup: `true_int8_h8_euclidean_int8`

Combo 12 of the case 4 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 4
- Model hash: `733b93bfdb24`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.830791** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 4 test split: P=0.989, R=1.000, A=0.994, F1=0.994.

Ranking quality (threshold-independent): AUC=1.0000, pAUC=1.0000.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0386070679) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.714282 | 1972 | 0.946 | 1.000 | 0.972 | 0.972 |  |
| `percentile_same_machine_99.0` | 1.830791 | 2249 | 0.989 | 1.000 | 0.994 | 0.994 | **<- active** |
| `percentile_same_machine_99.5` | 1.883914 | 2381 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `evt_p95_equiv` | 1.711726 | 1966 | 0.943 | 1.000 | 0.970 | 0.971 |  |
| `parametric_p95_equiv` | 1.717771 | 1980 | 0.946 | 1.000 | 0.972 | 0.972 |  |
| `kde_p95_equiv` | 1.723591 | 1993 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `evt_p99_equiv` | 1.846973 | 2289 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `parametric_p99_equiv` | 1.848672 | 2293 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `kde_p99_equiv` | 1.849394 | 2295 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `evt_p999_equiv` | 2.052688 | 2827 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `parametric_p999_equiv` | 2.007291 | 2703 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `kde_p999_equiv` | 2.028730 | 2761 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `mad` | 1.998424 | 2679 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `iqr` | 2.204168 | 3260 | 1.000 | 0.996 | 0.998 | 0.998 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
