# Deployment setup: `w_all_perchannel_knn16_fp32`

Combo 2 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.268076**.

At this threshold on the case 2 test split: P=0.992, R=0.966, A=0.979, F1=0.979.

Ranking quality (threshold-independent): AUC=0.9773, pAUC=0.9856.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.234024 | 0.966 | 0.974 | 0.970 | 0.970 |  |
| `percentile_same_machine_99.0` | 1.268076 | 0.992 | 0.966 | 0.979 | 0.979 | **<- active** |
| `percentile_same_machine_99.5` | 1.279572 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `evt_p95_equiv` | 1.235798 | 0.974 | 0.974 | 0.974 | 0.974 |  |
| `parametric_p95_equiv` | 1.238846 | 0.974 | 0.974 | 0.974 | 0.974 |  |
| `kde_p95_equiv` | 1.239366 | 0.974 | 0.974 | 0.974 | 0.974 |  |
| `evt_p99_equiv` | 1.267505 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `parametric_p99_equiv` | 1.290451 | 1.000 | 0.966 | 0.983 | 0.983 |  |
| `kde_p99_equiv` | 1.275003 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `evt_p999_equiv` | 1.294165 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `parametric_p999_equiv` | 1.350853 | 1.000 | 0.958 | 0.979 | 0.979 |  |
| `kde_p999_equiv` | 1.306788 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `mad` | 1.381868 | 1.000 | 0.951 | 0.975 | 0.975 |  |
| `iqr` | 1.483025 | 1.000 | 0.928 | 0.964 | 0.963 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
