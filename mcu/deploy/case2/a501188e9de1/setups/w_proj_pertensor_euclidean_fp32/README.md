# Deployment setup: `w_proj_pertensor_euclidean_fp32`

Combo 7 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: weight-only int8 (projections tensors, per-tensor)
- Head: euclidean head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.430832**.

At this threshold on the case 2 test split: P=0.988, R=0.909, A=0.949, F1=0.947.

Ranking quality (threshold-independent): AUC=0.9624, pAUC=0.9595.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.400932 | 0.957 | 0.913 | 0.936 | 0.934 |  |
| `percentile_same_machine_99.0` | 1.430832 | 0.988 | 0.909 | 0.949 | 0.947 | **<- active** |
| `percentile_same_machine_99.5` | 1.440311 | 0.992 | 0.902 | 0.947 | 0.945 |  |
| `evt_p95_equiv` | 1.398447 | 0.957 | 0.925 | 0.942 | 0.940 |  |
| `parametric_p95_equiv` | 1.414216 | 0.980 | 0.913 | 0.947 | 0.945 |  |
| `kde_p95_equiv` | 1.406990 | 0.976 | 0.913 | 0.945 | 0.943 |  |
| `evt_p99_equiv` | 1.435801 | 0.992 | 0.902 | 0.947 | 0.945 |  |
| `parametric_p99_equiv` | 1.477953 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `kde_p99_equiv` | 1.443266 | 0.992 | 0.902 | 0.947 | 0.945 |  |
| `evt_p999_equiv` | 1.481836 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `parametric_p999_equiv` | 1.550386 | 1.000 | 0.842 | 0.921 | 0.914 |  |
| `kde_p999_equiv` | 1.499309 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `mad` | 1.649169 | 1.000 | 0.725 | 0.862 | 0.840 |  |
| `iqr` | 1.802750 | 1.000 | 0.517 | 0.758 | 0.682 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
