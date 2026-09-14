# Deployment setup: `true_int8_h8_knn16_int8`

Combo 14 of the case 4 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 4
- Model hash: `733b93bfdb24`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.831369** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 4 test split: P=0.989, R=0.989, A=0.989, F1=0.989.

Ranking quality (threshold-independent): AUC=0.9929, pAUC=0.9953.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0386070679) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.720448 | 1986 | 0.949 | 0.992 | 0.970 | 0.970 |  |
| `percentile_same_machine_99.0` | 1.831369 | 2250 | 0.989 | 0.989 | 0.989 | 0.989 | **<- active** |
| `percentile_same_machine_99.5` | 1.882711 | 2378 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `evt_p95_equiv` | 1.715467 | 1974 | 0.949 | 0.992 | 0.970 | 0.970 |  |
| `parametric_p95_equiv` | 1.719776 | 1984 | 0.949 | 0.992 | 0.970 | 0.970 |  |
| `kde_p95_equiv` | 1.727686 | 2003 | 0.949 | 0.992 | 0.970 | 0.970 |  |
| `evt_p99_equiv` | 1.848289 | 2292 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `parametric_p99_equiv` | 1.847287 | 2289 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `kde_p99_equiv` | 1.845241 | 2284 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `evt_p999_equiv` | 2.015060 | 2724 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `parametric_p999_equiv` | 2.001484 | 2688 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `kde_p999_equiv` | 2.004083 | 2695 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `mad` | 2.003215 | 2692 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `iqr` | 2.209725 | 3276 | 1.000 | 0.958 | 0.979 | 0.979 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
