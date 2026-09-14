# Deployment setup: `w_all_pertensor_actboundaries_knn16_fp32`

Combo 10 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
- Head: knn_clustered_16 head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.214427**.

At this threshold on the case 4 test split: P=0.970, R=0.992, A=0.981, F1=0.981.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.181505 | 0.920 | 0.992 | 0.953 | 0.955 |  |
| `percentile_same_machine_99.0` | 1.214427 | 0.970 | 0.992 | 0.981 | 0.981 | **<- active** |
| `percentile_same_machine_99.5` | 1.234338 | 0.985 | 0.992 | 0.989 | 0.989 |  |
| `evt_p95_equiv` | 1.180408 | 0.920 | 0.992 | 0.953 | 0.955 |  |
| `parametric_p95_equiv` | 1.177138 | 0.920 | 0.992 | 0.953 | 0.955 |  |
| `kde_p95_equiv` | 1.185377 | 0.926 | 0.992 | 0.957 | 0.958 |  |
| `evt_p99_equiv` | 1.220045 | 0.974 | 0.992 | 0.983 | 0.983 |  |
| `parametric_p99_equiv` | 1.210193 | 0.967 | 0.992 | 0.979 | 0.980 |  |
| `kde_p99_equiv` | 1.226376 | 0.974 | 0.992 | 0.983 | 0.983 |  |
| `evt_p999_equiv` | 1.255854 | 0.992 | 0.992 | 0.992 | 0.992 |  |
| `parametric_p999_equiv` | 1.242213 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `kde_p999_equiv` | 1.268346 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `mad` | 1.320282 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `iqr` | 1.403264 | 1.000 | 0.989 | 0.994 | 0.994 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
