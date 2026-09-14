# Deployment setup: `true_int8_h16_euclidean_fp32`

Combo 15 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
- Backbone: true int8 arithmetic, h storage = int16
- Head: euclidean head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.217859**.

At this threshold on the case 4 test split: P=0.983, R=0.664, A=0.826, F1=0.793.

Ranking quality (threshold-independent): AUC=0.9081, pAUC=0.8461.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.041558 | 0.879 | 0.743 | 0.821 | 0.806 |  |
| `percentile_same_machine_99.0` | 2.217859 | 0.983 | 0.664 | 0.826 | 0.793 | **<- active** |
| `percentile_same_machine_99.5` | 2.238007 | 0.994 | 0.649 | 0.823 | 0.785 |  |
| `evt_p95_equiv` | 2.048587 | 0.894 | 0.732 | 0.823 | 0.805 |  |
| `parametric_p95_equiv` | 2.046341 | 0.890 | 0.736 | 0.823 | 0.806 |  |
| `kde_p95_equiv` | 2.061817 | 0.898 | 0.728 | 0.823 | 0.804 |  |
| `evt_p99_equiv` | 2.200671 | 0.968 | 0.675 | 0.826 | 0.796 |  |
| `parametric_p99_equiv` | 2.180351 | 0.963 | 0.691 | 0.832 | 0.804 |  |
| `kde_p99_equiv` | 2.226904 | 0.994 | 0.660 | 0.828 | 0.794 |  |
| `evt_p999_equiv` | 2.324236 | 1.000 | 0.574 | 0.787 | 0.729 |  |
| `parametric_p999_equiv` | 2.314746 | 1.000 | 0.585 | 0.792 | 0.738 |  |
| `kde_p999_equiv` | 2.366973 | 1.000 | 0.528 | 0.764 | 0.691 |  |
| `mad` | 2.411903 | 1.000 | 0.506 | 0.753 | 0.672 |  |
| `iqr` | 2.669841 | 1.000 | 0.400 | 0.700 | 0.571 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
