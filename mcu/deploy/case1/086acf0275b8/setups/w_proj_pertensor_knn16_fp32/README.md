# Deployment setup: `w_proj_pertensor_knn16_fp32`

Combo 8 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.672728**.

At this threshold on the case 1 test split: P=0.988, R=0.625, A=0.809, F1=0.766.

Ranking quality (threshold-independent): AUC=0.8192, pAUC=0.8399.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.591014 | 0.927 | 0.720 | 0.831 | 0.810 |  |
| `percentile_same_machine_99.0` | 1.672728 | 0.988 | 0.625 | 0.809 | 0.766 | **<- active** |
| `percentile_same_machine_99.5` | 1.695281 | 0.987 | 0.595 | 0.794 | 0.742 |  |
| `evt_p95_equiv` | 1.592506 | 0.931 | 0.716 | 0.831 | 0.809 |  |
| `parametric_p95_equiv` | 1.610574 | 0.948 | 0.697 | 0.830 | 0.803 |  |
| `kde_p95_equiv` | 1.607457 | 0.949 | 0.701 | 0.831 | 0.806 |  |
| `evt_p99_equiv` | 1.667411 | 0.988 | 0.633 | 0.812 | 0.771 |  |
| `parametric_p99_equiv` | 1.672140 | 0.988 | 0.625 | 0.809 | 0.766 |  |
| `kde_p99_equiv` | 1.685007 | 0.988 | 0.610 | 0.801 | 0.754 |  |
| `evt_p999_equiv` | 1.774665 | 0.993 | 0.542 | 0.769 | 0.701 |  |
| `parametric_p999_equiv` | 1.732349 | 0.993 | 0.568 | 0.782 | 0.723 |  |
| `kde_p999_equiv` | 1.799832 | 1.000 | 0.523 | 0.761 | 0.687 |  |
| `mad` | 1.782333 | 1.000 | 0.534 | 0.767 | 0.696 |  |
| `iqr` | 1.906558 | 1.000 | 0.504 | 0.752 | 0.670 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
