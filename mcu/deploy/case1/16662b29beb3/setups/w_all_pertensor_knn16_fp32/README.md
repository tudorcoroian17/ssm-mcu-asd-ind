# Deployment setup: `w_all_pertensor_knn16_fp32`

Combo 4 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.349115**.

At this threshold on the case 1 test split: P=0.983, R=0.443, A=0.718, F1=0.611.

Ranking quality (threshold-independent): AUC=0.9108, pAUC=0.8565.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.743148 | 0.948 | 0.761 | 0.860 | 0.845 |  |
| `percentile_same_machine_99.0` | 2.349115 | 0.983 | 0.443 | 0.718 | 0.611 | **<- active** |
| `percentile_same_machine_99.5` | 2.419949 | 0.981 | 0.398 | 0.695 | 0.566 |  |
| `evt_p95_equiv` | 1.776883 | 0.966 | 0.742 | 0.858 | 0.839 |  |
| `parametric_p95_equiv` | 1.655373 | 0.901 | 0.826 | 0.867 | 0.862 |  |
| `kde_p95_equiv` | 1.757365 | 0.957 | 0.750 | 0.858 | 0.841 |  |
| `evt_p99_equiv` | 2.229647 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `parametric_p99_equiv` | 1.880934 | 0.978 | 0.689 | 0.837 | 0.809 |  |
| `kde_p99_equiv` | 2.356533 | 0.983 | 0.436 | 0.714 | 0.604 |  |
| `evt_p999_equiv` | 3.295334 | 1.000 | 0.045 | 0.523 | 0.087 |  |
| `parametric_p999_equiv` | 2.170493 | 0.978 | 0.496 | 0.742 | 0.658 |  |
| `kde_p999_equiv` | 2.524458 | 1.000 | 0.303 | 0.652 | 0.465 |  |
| `mad` | 1.494301 | 0.891 | 0.924 | 0.905 | 0.907 |  |
| `iqr` | 1.599817 | 0.886 | 0.852 | 0.871 | 0.869 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
