# Deployment setup: `w_proj_pertensor_euclidean_fp32`

Combo 7 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **4.013941**.

At this threshold on the case 1 test split: P=0.846, R=0.083, A=0.534, F1=0.152.

Ranking quality (threshold-independent): AUC=0.8725, pAUC=0.7734.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.101883 | 0.939 | 0.701 | 0.828 | 0.803 |  |
| `percentile_same_machine_99.0` | 4.013941 | 0.846 | 0.083 | 0.534 | 0.152 | **<- active** |
| `percentile_same_machine_99.5` | 7.831523 | 1.000 | 0.042 | 0.521 | 0.080 |  |
| `evt_p95_equiv` | 2.140998 | 0.941 | 0.659 | 0.809 | 0.775 |  |
| `parametric_p95_equiv` | 2.450001 | 0.944 | 0.508 | 0.739 | 0.660 |  |
| `kde_p95_equiv` | 2.231812 | 0.949 | 0.633 | 0.799 | 0.759 |  |
| `evt_p99_equiv` | 4.090838 | 0.846 | 0.083 | 0.534 | 0.152 |  |
| `parametric_p99_equiv` | 2.775902 | 0.930 | 0.352 | 0.663 | 0.511 |  |
| `kde_p99_equiv` | 7.353201 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 41.321629 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.193002 | 0.938 | 0.231 | 0.608 | 0.371 |  |
| `kde_p999_equiv` | 8.039434 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.223435 | 0.949 | 0.633 | 0.799 | 0.759 |  |
| `iqr` | 2.380680 | 0.948 | 0.549 | 0.759 | 0.695 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
