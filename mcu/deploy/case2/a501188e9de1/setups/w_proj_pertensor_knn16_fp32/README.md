# Deployment setup: `w_proj_pertensor_knn16_fp32`

Combo 8 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: weight-only int8 (projections tensors, per-tensor)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.381985**.

At this threshold on the case 2 test split: P=0.991, R=0.872, A=0.932, F1=0.928.

Ranking quality (threshold-independent): AUC=0.9372, pAUC=0.9489.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.359926 | 0.971 | 0.898 | 0.936 | 0.933 |  |
| `percentile_same_machine_99.0` | 1.381985 | 0.991 | 0.872 | 0.932 | 0.928 | **<- active** |
| `percentile_same_machine_99.5` | 1.394275 | 0.991 | 0.864 | 0.928 | 0.923 |  |
| `evt_p95_equiv` | 1.357513 | 0.960 | 0.902 | 0.932 | 0.930 |  |
| `parametric_p95_equiv` | 1.367446 | 0.983 | 0.894 | 0.940 | 0.937 |  |
| `kde_p95_equiv` | 1.363824 | 0.983 | 0.894 | 0.940 | 0.937 |  |
| `evt_p99_equiv` | 1.387349 | 0.991 | 0.868 | 0.930 | 0.926 |  |
| `parametric_p99_equiv` | 1.419458 | 0.996 | 0.853 | 0.925 | 0.919 |  |
| `kde_p99_equiv` | 1.392860 | 0.991 | 0.864 | 0.928 | 0.923 |  |
| `evt_p999_equiv` | 1.426350 | 0.996 | 0.853 | 0.925 | 0.919 |  |
| `parametric_p999_equiv` | 1.479288 | 1.000 | 0.762 | 0.881 | 0.865 |  |
| `kde_p999_equiv` | 1.441759 | 1.000 | 0.842 | 0.921 | 0.914 |  |
| `mad` | 1.548884 | 1.000 | 0.630 | 0.815 | 0.773 |  |
| `iqr` | 1.670821 | 1.000 | 0.434 | 0.717 | 0.605 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
