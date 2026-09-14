# Deployment setup: `w_all_perchannel_knn16_fp32`

Combo 2 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.378586**.

At this threshold on the case 2 test split: P=0.992, R=0.883, A=0.938, F1=0.934.

Ranking quality (threshold-independent): AUC=0.9396, pAUC=0.9516.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.355694 | 0.972 | 0.902 | 0.938 | 0.935 |  |
| `percentile_same_machine_99.0` | 1.378586 | 0.992 | 0.883 | 0.938 | 0.934 | **<- active** |
| `percentile_same_machine_99.5` | 1.390863 | 0.991 | 0.868 | 0.930 | 0.926 |  |
| `evt_p95_equiv` | 1.353698 | 0.960 | 0.906 | 0.934 | 0.932 |  |
| `parametric_p95_equiv` | 1.363977 | 0.983 | 0.894 | 0.940 | 0.937 |  |
| `kde_p95_equiv` | 1.359927 | 0.979 | 0.898 | 0.940 | 0.937 |  |
| `evt_p99_equiv` | 1.383211 | 0.991 | 0.875 | 0.934 | 0.930 |  |
| `parametric_p99_equiv` | 1.415823 | 0.996 | 0.857 | 0.926 | 0.921 |  |
| `kde_p99_equiv` | 1.389089 | 0.991 | 0.872 | 0.932 | 0.928 |  |
| `evt_p999_equiv` | 1.422993 | 0.996 | 0.853 | 0.925 | 0.919 |  |
| `parametric_p999_equiv` | 1.475463 | 1.000 | 0.785 | 0.892 | 0.879 |  |
| `kde_p999_equiv` | 1.438657 | 1.000 | 0.853 | 0.926 | 0.921 |  |
| `mad` | 1.544978 | 1.000 | 0.660 | 0.830 | 0.795 |  |
| `iqr` | 1.669011 | 1.000 | 0.460 | 0.730 | 0.630 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
