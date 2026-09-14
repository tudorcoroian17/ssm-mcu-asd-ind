# Deployment setup: `true_int8_h16_knn16_int8`

Combo 18 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.270546** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 3 test split: P=0.974, R=0.989, A=0.981, F1=0.981.

Ranking quality (threshold-independent): AUC=0.9888, pAUC=0.9940.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0315041016) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.206320 | 1466 | 0.936 | 0.989 | 0.960 | 0.961 |  |
| `percentile_same_machine_99.0` | 1.270546 | 1626 | 0.974 | 0.989 | 0.981 | 0.981 | **<- active** |
| `percentile_same_machine_99.5` | 1.300417 | 1704 | 0.981 | 0.989 | 0.985 | 0.985 |  |
| `evt_p95_equiv` | 1.206239 | 1466 | 0.936 | 0.989 | 0.960 | 0.961 |  |
| `parametric_p95_equiv` | 1.260506 | 1601 | 0.974 | 0.989 | 0.981 | 0.981 |  |
| `kde_p95_equiv` | 1.217148 | 1493 | 0.953 | 0.989 | 0.970 | 0.970 |  |
| `evt_p99_equiv` | 1.281118 | 1654 | 0.978 | 0.989 | 0.983 | 0.983 |  |
| `parametric_p99_equiv` | 1.320101 | 1756 | 0.992 | 0.989 | 0.991 | 0.991 |  |
| `kde_p99_equiv` | 1.282526 | 1657 | 0.978 | 0.989 | 0.983 | 0.983 |  |
| `evt_p999_equiv` | 1.435380 | 2076 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `parametric_p999_equiv` | 1.378875 | 1916 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `kde_p999_equiv` | 1.776655 | 3180 | 1.000 | 0.970 | 0.985 | 0.985 |  |
| `mad` | 1.381024 | 1922 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `iqr` | 1.488248 | 2232 | 1.000 | 0.989 | 0.994 | 0.994 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
