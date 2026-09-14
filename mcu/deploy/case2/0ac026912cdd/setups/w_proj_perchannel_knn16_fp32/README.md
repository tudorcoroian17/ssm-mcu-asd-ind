# Deployment setup: `w_proj_perchannel_knn16_fp32`

Combo 6 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
- Backbone: weight-only int8 (projections tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.269373**.

At this threshold on the case 2 test split: P=0.992, R=0.974, A=0.983, F1=0.983.

Ranking quality (threshold-independent): AUC=0.9797, pAUC=0.9858.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.237490 | 0.974 | 0.974 | 0.974 | 0.974 |  |
| `percentile_same_machine_99.0` | 1.269373 | 0.992 | 0.974 | 0.983 | 0.983 | **<- active** |
| `percentile_same_machine_99.5` | 1.284432 | 0.996 | 0.966 | 0.981 | 0.981 |  |
| `evt_p95_equiv` | 1.239156 | 0.974 | 0.974 | 0.974 | 0.974 |  |
| `parametric_p95_equiv` | 1.243550 | 0.977 | 0.974 | 0.975 | 0.975 |  |
| `kde_p95_equiv` | 1.243138 | 0.977 | 0.974 | 0.975 | 0.975 |  |
| `evt_p99_equiv` | 1.269102 | 0.992 | 0.974 | 0.983 | 0.983 |  |
| `parametric_p99_equiv` | 1.292657 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `kde_p99_equiv` | 1.277458 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `evt_p999_equiv` | 1.297709 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `parametric_p999_equiv` | 1.349201 | 1.000 | 0.958 | 0.979 | 0.979 |  |
| `kde_p999_equiv` | 1.308485 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `mad` | 1.397413 | 1.000 | 0.947 | 0.974 | 0.973 |  |
| `iqr` | 1.499543 | 1.000 | 0.917 | 0.958 | 0.957 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
