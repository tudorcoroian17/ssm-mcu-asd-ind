# Deployment setup: `true_int8_h8_knn16_fp32`

Combo 13 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
- Backbone: true int8 arithmetic, h storage = int8
- Head: knn_clustered_16 head (fp32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.426822**.

At this threshold on the case 3 test split: P=0.996, R=0.996, A=0.996, F1=0.996.

Ranking quality (threshold-independent): AUC=0.9962, pAUC=0.9980.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.314843 | 0.960 | 0.996 | 0.977 | 0.978 |  |
| `percentile_same_machine_99.0` | 1.426822 | 0.996 | 0.996 | 0.996 | 0.996 | **<- active** |
| `percentile_same_machine_99.5` | 1.472024 | 0.996 | 0.996 | 0.996 | 0.996 |  |
| `evt_p95_equiv` | 1.316207 | 0.960 | 0.996 | 0.977 | 0.978 |  |
| `parametric_p95_equiv` | 1.322680 | 0.967 | 0.996 | 0.981 | 0.981 |  |
| `kde_p95_equiv` | 1.321946 | 0.967 | 0.996 | 0.981 | 0.981 |  |
| `evt_p99_equiv` | 1.442775 | 0.996 | 0.996 | 0.996 | 0.996 |  |
| `parametric_p99_equiv` | 1.405822 | 0.996 | 0.996 | 0.996 | 0.996 |  |
| `kde_p99_equiv` | 1.435704 | 0.996 | 0.996 | 0.996 | 0.996 |  |
| `evt_p999_equiv` | 1.693426 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `parametric_p999_equiv` | 1.503002 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `kde_p999_equiv` | 1.664909 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `mad` | 1.473375 | 0.996 | 0.996 | 0.996 | 0.996 |  |
| `iqr` | 1.595858 | 1.000 | 0.992 | 0.996 | 0.996 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
