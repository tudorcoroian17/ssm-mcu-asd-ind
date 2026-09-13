# Deployment setup: `full_fp32_knn16_fp32`

Combo 20 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.693908**.

At this threshold on the case 1 test split: P=0.988, R=0.625, A=0.809, F1=0.766.

Ranking quality (threshold-independent): AUC=0.8191, pAUC=0.8373.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.612363 | 0.926 | 0.716 | 0.830 | 0.808 |  |
| `percentile_same_machine_99.0` | 1.693908 | 0.988 | 0.625 | 0.809 | 0.766 | **<- active** |
| `percentile_same_machine_99.5` | 1.729704 | 0.987 | 0.591 | 0.792 | 0.739 |  |
| `evt_p95_equiv` | 1.611531 | 0.927 | 0.720 | 0.831 | 0.810 |  |
| `parametric_p95_equiv` | 1.618107 | 0.935 | 0.708 | 0.830 | 0.806 |  |
| `kde_p95_equiv` | 1.622045 | 0.949 | 0.701 | 0.831 | 0.806 |  |
| `evt_p99_equiv` | 1.694532 | 0.988 | 0.617 | 0.805 | 0.760 |  |
| `parametric_p99_equiv` | 1.674379 | 0.983 | 0.640 | 0.814 | 0.775 |  |
| `kde_p99_equiv` | 1.705756 | 0.988 | 0.606 | 0.799 | 0.751 |  |
| `evt_p999_equiv` | 1.840611 | 1.000 | 0.511 | 0.756 | 0.677 |  |
| `parametric_p999_equiv` | 1.729234 | 0.987 | 0.591 | 0.792 | 0.739 |  |
| `kde_p999_equiv` | 1.866452 | 1.000 | 0.511 | 0.756 | 0.677 |  |
| `mad` | 1.813750 | 1.000 | 0.530 | 0.765 | 0.693 |  |
| `iqr` | 1.945951 | 1.000 | 0.500 | 0.750 | 0.667 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
