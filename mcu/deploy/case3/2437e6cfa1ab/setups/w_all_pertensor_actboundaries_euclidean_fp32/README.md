# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.069480**.

At this threshold on the case 3 test split: P=0.991, R=0.875, A=0.934, F1=0.930.

Ranking quality (threshold-independent): AUC=0.9919, pAUC=0.9571.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.697632 | 0.953 | 0.996 | 0.974 | 0.974 |  |
| `percentile_same_machine_99.0` | 2.069480 | 0.991 | 0.875 | 0.934 | 0.930 | **<- active** |
| `percentile_same_machine_99.5` | 3.761814 | 0.938 | 0.113 | 0.553 | 0.202 |  |
| `evt_p95_equiv` | 1.692235 | 0.950 | 0.996 | 0.972 | 0.972 |  |
| `parametric_p95_equiv` | 1.813112 | 0.974 | 0.981 | 0.977 | 0.977 |  |
| `kde_p95_equiv` | 1.728028 | 0.960 | 0.985 | 0.972 | 0.972 |  |
| `evt_p99_equiv` | 2.166012 | 0.990 | 0.725 | 0.858 | 0.837 |  |
| `parametric_p99_equiv` | 2.042017 | 0.992 | 0.909 | 0.951 | 0.949 |  |
| `kde_p99_equiv` | 2.086972 | 0.991 | 0.857 | 0.925 | 0.919 |  |
| `evt_p999_equiv` | 5.258344 | 1.000 | 0.038 | 0.519 | 0.073 |  |
| `parametric_p999_equiv` | 2.333117 | 0.984 | 0.479 | 0.736 | 0.645 |  |
| `kde_p999_equiv` | 4.760658 | 1.000 | 0.057 | 0.528 | 0.107 |  |
| `mad` | 2.139264 | 0.990 | 0.770 | 0.881 | 0.866 |  |
| `iqr` | 2.408255 | 0.981 | 0.389 | 0.691 | 0.557 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
