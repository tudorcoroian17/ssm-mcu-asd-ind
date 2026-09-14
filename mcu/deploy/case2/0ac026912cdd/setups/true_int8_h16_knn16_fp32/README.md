# Deployment setup: `true_int8_h16_knn16_fp32`

Combo 17 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
- Backbone: true int8 arithmetic, h storage = int16
- Head: knn_clustered_16 head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.532445**.

At this threshold on the case 2 test split: P=0.987, R=0.883, A=0.936, F1=0.932.

Ranking quality (threshold-independent): AUC=0.9742, pAUC=0.9659.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.495068 | 0.947 | 0.947 | 0.947 | 0.947 |  |
| `percentile_same_machine_99.0` | 1.532445 | 0.987 | 0.883 | 0.936 | 0.932 | **<- active** |
| `percentile_same_machine_99.5` | 1.561987 | 1.000 | 0.838 | 0.919 | 0.912 |  |
| `evt_p95_equiv` | 1.493736 | 0.940 | 0.947 | 0.943 | 0.944 |  |
| `parametric_p95_equiv` | 1.493905 | 0.940 | 0.947 | 0.943 | 0.944 |  |
| `kde_p95_equiv` | 1.497857 | 0.958 | 0.943 | 0.951 | 0.951 |  |
| `evt_p99_equiv` | 1.537767 | 0.991 | 0.875 | 0.934 | 0.930 |  |
| `parametric_p99_equiv` | 1.549625 | 1.000 | 0.860 | 0.930 | 0.925 |  |
| `kde_p99_equiv` | 1.544769 | 1.000 | 0.868 | 0.934 | 0.929 |  |
| `evt_p999_equiv` | 1.579024 | 1.000 | 0.823 | 0.911 | 0.903 |  |
| `parametric_p999_equiv` | 1.614555 | 1.000 | 0.777 | 0.889 | 0.875 |  |
| `kde_p999_equiv` | 1.587773 | 1.000 | 0.800 | 0.900 | 0.889 |  |
| `mad` | 1.641506 | 1.000 | 0.725 | 0.862 | 0.840 |  |
| `iqr` | 1.739168 | 1.000 | 0.589 | 0.794 | 0.741 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
