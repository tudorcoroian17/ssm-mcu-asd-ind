# Deployment setup: `w_all_pertensor_knn16_fp32`

Combo 4 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
- Backbone: weight-only int8 (all tensors, per-tensor)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.668422**.

At this threshold on the case 1 test split: P=0.982, R=0.606, A=0.797, F1=0.749.

Ranking quality (threshold-independent): AUC=0.8197, pAUC=0.8379.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.583567 | 0.941 | 0.720 | 0.837 | 0.815 |  |
| `percentile_same_machine_99.0` | 1.668422 | 0.982 | 0.606 | 0.797 | 0.749 | **<- active** |
| `percentile_same_machine_99.5` | 1.729513 | 0.987 | 0.568 | 0.780 | 0.721 |  |
| `evt_p95_equiv` | 1.583422 | 0.941 | 0.720 | 0.837 | 0.815 |  |
| `parametric_p95_equiv` | 1.611820 | 0.953 | 0.686 | 0.826 | 0.797 |  |
| `kde_p95_equiv` | 1.599758 | 0.948 | 0.697 | 0.830 | 0.803 |  |
| `evt_p99_equiv` | 1.674256 | 0.981 | 0.602 | 0.795 | 0.746 |  |
| `parametric_p99_equiv` | 1.678356 | 0.981 | 0.598 | 0.794 | 0.744 |  |
| `kde_p99_equiv` | 1.691901 | 0.981 | 0.591 | 0.790 | 0.738 |  |
| `evt_p999_equiv` | 1.862061 | 1.000 | 0.511 | 0.756 | 0.677 |  |
| `parametric_p999_equiv` | 1.743611 | 0.987 | 0.557 | 0.775 | 0.712 |  |
| `kde_p999_equiv` | 1.880050 | 1.000 | 0.508 | 0.754 | 0.673 |  |
| `mad` | 1.781405 | 1.000 | 0.530 | 0.765 | 0.693 |  |
| `iqr` | 1.904509 | 1.000 | 0.500 | 0.750 | 0.667 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
