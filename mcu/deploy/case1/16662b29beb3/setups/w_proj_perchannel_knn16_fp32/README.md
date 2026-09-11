# Deployment setup: `w_proj_perchannel_knn16_fp32`

Combo 6 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.328639**.

At this threshold on the case 1 test split: P=0.983, R=0.443, A=0.718, F1=0.611.

Ranking quality (threshold-independent): AUC=0.9106, pAUC=0.8538.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.730061 | 0.943 | 0.758 | 0.856 | 0.840 |  |
| `percentile_same_machine_99.0` | 2.328639 | 0.983 | 0.443 | 0.718 | 0.611 | **<- active** |
| `percentile_same_machine_99.5` | 2.400291 | 0.981 | 0.398 | 0.695 | 0.566 |  |
| `evt_p95_equiv` | 1.756687 | 0.961 | 0.742 | 0.856 | 0.838 |  |
| `parametric_p95_equiv` | 1.722353 | 0.939 | 0.761 | 0.856 | 0.841 |  |
| `kde_p95_equiv` | 1.744637 | 0.956 | 0.746 | 0.856 | 0.838 |  |
| `evt_p99_equiv` | 2.207691 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `parametric_p99_equiv` | 1.950825 | 0.976 | 0.617 | 0.801 | 0.756 |  |
| `kde_p99_equiv` | 2.334222 | 0.983 | 0.432 | 0.712 | 0.600 |  |
| `evt_p999_equiv` | 3.357407 | 1.000 | 0.038 | 0.519 | 0.073 |  |
| `parametric_p999_equiv` | 2.214693 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `kde_p999_equiv` | 2.508054 | 1.000 | 0.299 | 0.650 | 0.461 |  |
| `mad` | 1.466667 | 0.890 | 0.920 | 0.903 | 0.905 |  |
| `iqr` | 1.565192 | 0.883 | 0.856 | 0.871 | 0.869 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
