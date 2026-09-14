# Deployment setup: `true_int8_h8_knn16_int8`

Combo 14 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.946231** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 4 test split: P=0.972, R=0.917, A=0.945, F1=0.944.

Ranking quality (threshold-independent): AUC=0.9894, pAUC=0.9704.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0351456207) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.765348 | 2523 | 0.932 | 0.977 | 0.953 | 0.954 |  |
| `percentile_same_machine_99.0` | 1.946231 | 3067 | 0.972 | 0.917 | 0.945 | 0.944 | **<- active** |
| `percentile_same_machine_99.5` | 2.041279 | 3373 | 0.996 | 0.842 | 0.919 | 0.912 |  |
| `evt_p95_equiv` | 1.761409 | 2512 | 0.925 | 0.977 | 0.949 | 0.950 |  |
| `parametric_p95_equiv` | 1.788733 | 2590 | 0.942 | 0.977 | 0.958 | 0.959 |  |
| `kde_p95_equiv` | 1.776583 | 2555 | 0.938 | 0.977 | 0.957 | 0.957 |  |
| `evt_p99_equiv` | 1.972837 | 3151 | 0.972 | 0.906 | 0.940 | 0.938 |  |
| `parametric_p99_equiv` | 1.977915 | 3167 | 0.972 | 0.906 | 0.940 | 0.938 |  |
| `kde_p99_equiv` | 1.967189 | 3133 | 0.972 | 0.906 | 0.940 | 0.938 |  |
| `evt_p999_equiv` | 2.387823 | 4616 | 1.000 | 0.592 | 0.796 | 0.744 |  |
| `parametric_p999_equiv` | 2.213850 | 3968 | 1.000 | 0.702 | 0.851 | 0.825 |  |
| `kde_p999_equiv` | 2.621343 | 5563 | 1.000 | 0.442 | 0.721 | 0.613 |  |
| `mad` | 2.102967 | 3580 | 0.995 | 0.770 | 0.883 | 0.868 |  |
| `iqr` | 2.354160 | 4487 | 1.000 | 0.630 | 0.815 | 0.773 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
