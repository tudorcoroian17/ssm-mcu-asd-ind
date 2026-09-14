# Deployment setup: `true_int8_h16_knn16_int8`

Combo 18 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.535734** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 4 test split: P=0.988, R=0.966, A=0.977, F1=0.977.

Ranking quality (threshold-independent): AUC=0.9911, pAUC=0.9891.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0351456207) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.385656 | 1554 | 0.913 | 0.992 | 0.949 | 0.951 |  |
| `percentile_same_machine_99.0` | 1.535734 | 1909 | 0.988 | 0.966 | 0.977 | 0.977 | **<- active** |
| `percentile_same_machine_99.5` | 1.561527 | 1974 | 0.996 | 0.962 | 0.979 | 0.979 |  |
| `evt_p95_equiv` | 1.388221 | 1560 | 0.916 | 0.992 | 0.951 | 0.953 |  |
| `parametric_p95_equiv` | 1.364118 | 1506 | 0.910 | 0.992 | 0.947 | 0.949 |  |
| `kde_p95_equiv` | 1.391249 | 1567 | 0.916 | 0.992 | 0.951 | 0.953 |  |
| `evt_p99_equiv` | 1.524503 | 1882 | 0.981 | 0.966 | 0.974 | 0.973 |  |
| `parametric_p99_equiv` | 1.505828 | 1836 | 0.966 | 0.974 | 0.970 | 0.970 |  |
| `kde_p99_equiv` | 1.540418 | 1921 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `evt_p999_equiv` | 1.637334 | 2170 | 1.000 | 0.909 | 0.955 | 0.953 |  |
| `parametric_p999_equiv` | 1.682241 | 2291 | 1.000 | 0.887 | 0.943 | 0.940 |  |
| `kde_p999_equiv` | 1.642245 | 2183 | 1.000 | 0.909 | 0.955 | 0.953 |  |
| `mad` | 1.627709 | 2145 | 1.000 | 0.913 | 0.957 | 0.955 |  |
| `iqr` | 1.832257 | 2718 | 1.000 | 0.725 | 0.862 | 0.840 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
