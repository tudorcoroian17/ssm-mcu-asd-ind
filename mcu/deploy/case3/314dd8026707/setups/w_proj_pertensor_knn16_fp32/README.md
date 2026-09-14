# Deployment setup: `w_proj_pertensor_knn16_fp32`

Combo 8 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.390140**.

At this threshold on the case 3 test split: P=0.972, R=0.774, A=0.875, F1=0.861.

Ranking quality (threshold-independent): AUC=0.9706, pAUC=0.9126.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.333944 | 0.933 | 0.894 | 0.915 | 0.913 |  |
| `percentile_same_machine_99.0` | 1.390140 | 0.972 | 0.774 | 0.875 | 0.861 | **<- active** |
| `percentile_same_machine_99.5` | 1.411134 | 0.984 | 0.709 | 0.849 | 0.825 |  |
| `evt_p95_equiv` | 1.332370 | 0.933 | 0.894 | 0.915 | 0.913 |  |
| `parametric_p95_equiv` | 1.372404 | 0.965 | 0.826 | 0.898 | 0.890 |  |
| `kde_p95_equiv` | 1.356424 | 0.962 | 0.853 | 0.909 | 0.904 |  |
| `evt_p99_equiv` | 1.407591 | 0.984 | 0.717 | 0.853 | 0.830 |  |
| `parametric_p99_equiv` | 1.491085 | 0.992 | 0.449 | 0.723 | 0.618 |  |
| `kde_p99_equiv` | 1.434673 | 0.989 | 0.653 | 0.823 | 0.786 |  |
| `evt_p999_equiv` | 1.543662 | 1.000 | 0.321 | 0.660 | 0.486 |  |
| `parametric_p999_equiv` | 1.612384 | 1.000 | 0.245 | 0.623 | 0.394 |  |
| `kde_p999_equiv` | 1.764442 | 1.000 | 0.211 | 0.606 | 0.349 |  |
| `mad` | 1.994133 | 1.000 | 0.143 | 0.572 | 0.251 |  |
| `iqr` | 2.340557 | 1.000 | 0.117 | 0.558 | 0.209 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
