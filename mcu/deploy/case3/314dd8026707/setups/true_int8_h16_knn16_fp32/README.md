# Deployment setup: `true_int8_h16_knn16_fp32`

Combo 17 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
- Backbone: true int8 arithmetic, h storage = int16
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.361458**.

At this threshold on the case 3 test split: P=0.983, R=0.891, A=0.938, F1=0.935.

Ranking quality (threshold-independent): AUC=0.9834, pAUC=0.9557.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.297852 | 0.938 | 0.962 | 0.949 | 0.950 |  |
| `percentile_same_machine_99.0` | 1.361458 | 0.983 | 0.891 | 0.938 | 0.935 | **<- active** |
| `percentile_same_machine_99.5` | 1.408375 | 0.995 | 0.796 | 0.896 | 0.885 |  |
| `evt_p95_equiv` | 1.293076 | 0.935 | 0.974 | 0.953 | 0.954 |  |
| `parametric_p95_equiv` | 1.373988 | 0.983 | 0.853 | 0.919 | 0.913 |  |
| `kde_p95_equiv` | 1.314177 | 0.954 | 0.947 | 0.951 | 0.951 |  |
| `evt_p99_equiv` | 1.393569 | 0.991 | 0.819 | 0.906 | 0.897 |  |
| `parametric_p99_equiv` | 1.539146 | 0.992 | 0.475 | 0.736 | 0.643 |  |
| `kde_p99_equiv` | 1.395659 | 0.995 | 0.815 | 0.906 | 0.896 |  |
| `evt_p999_equiv` | 1.636381 | 1.000 | 0.264 | 0.632 | 0.418 |  |
| `parametric_p999_equiv` | 1.738891 | 1.000 | 0.230 | 0.615 | 0.374 |  |
| `kde_p999_equiv` | 3.184521 | 1.000 | 0.057 | 0.528 | 0.107 |  |
| `mad` | 1.811108 | 1.000 | 0.230 | 0.615 | 0.374 |  |
| `iqr` | 2.147530 | 1.000 | 0.121 | 0.560 | 0.215 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
