# Deployment setup: `full_fp32_knn16_fp32`

Combo 20 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
- Backbone: full fp32 (no quantization)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.261948**.

At this threshold on the case 4 test split: P=0.974, R=0.992, A=0.983, F1=0.983.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.229698 | 0.926 | 0.992 | 0.957 | 0.958 |  |
| `percentile_same_machine_99.0` | 1.261948 | 0.974 | 0.992 | 0.983 | 0.983 | **<- active** |
| `percentile_same_machine_99.5` | 1.277745 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `evt_p95_equiv` | 1.226751 | 0.920 | 0.992 | 0.953 | 0.955 |  |
| `parametric_p95_equiv` | 1.224077 | 0.913 | 0.992 | 0.949 | 0.951 |  |
| `kde_p95_equiv` | 1.233837 | 0.926 | 0.992 | 0.957 | 0.958 |  |
| `evt_p99_equiv` | 1.270249 | 0.981 | 0.992 | 0.987 | 0.987 |  |
| `parametric_p99_equiv` | 1.263267 | 0.978 | 0.992 | 0.985 | 0.985 |  |
| `kde_p99_equiv` | 1.275535 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `evt_p999_equiv` | 1.310696 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `parametric_p999_equiv` | 1.301370 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `kde_p999_equiv` | 1.326318 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `mad` | 1.395581 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `iqr` | 1.517324 | 1.000 | 0.958 | 0.979 | 0.979 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
