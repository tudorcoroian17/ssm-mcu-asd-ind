# Deployment setup: `w_proj_perchannel_knn16_fp32`

Combo 6 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.376782**.

At this threshold on the case 3 test split: P=0.973, R=0.823, A=0.900, F1=0.892.

Ranking quality (threshold-independent): AUC=0.9728, pAUC=0.9217.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.331507 | 0.948 | 0.894 | 0.923 | 0.920 |  |
| `percentile_same_machine_99.0` | 1.376782 | 0.973 | 0.823 | 0.900 | 0.892 | **<- active** |
| `percentile_same_machine_99.5` | 1.406353 | 0.985 | 0.736 | 0.862 | 0.842 |  |
| `evt_p95_equiv` | 1.329135 | 0.941 | 0.898 | 0.921 | 0.919 |  |
| `parametric_p95_equiv` | 1.370086 | 0.970 | 0.842 | 0.908 | 0.901 |  |
| `kde_p95_equiv` | 1.353340 | 0.962 | 0.864 | 0.915 | 0.911 |  |
| `evt_p99_equiv` | 1.403707 | 0.985 | 0.743 | 0.866 | 0.847 |  |
| `parametric_p99_equiv` | 1.488794 | 0.993 | 0.502 | 0.749 | 0.667 |  |
| `kde_p99_equiv` | 1.429771 | 0.989 | 0.691 | 0.842 | 0.813 |  |
| `evt_p999_equiv` | 1.539353 | 1.000 | 0.321 | 0.660 | 0.486 |  |
| `parametric_p999_equiv` | 1.610137 | 1.000 | 0.249 | 0.625 | 0.399 |  |
| `kde_p999_equiv` | 1.804952 | 1.000 | 0.204 | 0.602 | 0.339 |  |
| `mad` | 1.990637 | 1.000 | 0.147 | 0.574 | 0.257 |  |
| `iqr` | 2.332438 | 1.000 | 0.117 | 0.558 | 0.209 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
