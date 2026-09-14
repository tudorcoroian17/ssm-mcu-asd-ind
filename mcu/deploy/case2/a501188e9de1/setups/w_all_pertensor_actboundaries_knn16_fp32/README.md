# Deployment setup: `w_all_pertensor_actboundaries_knn16_fp32`

Combo 10 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.652760**.

At this threshold on the case 2 test split: P=0.995, R=0.819, A=0.908, F1=0.899.

Ranking quality (threshold-independent): AUC=0.9417, pAUC=0.9236.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.609418 | 0.974 | 0.849 | 0.913 | 0.907 |  |
| `percentile_same_machine_99.0` | 1.652760 | 0.995 | 0.819 | 0.908 | 0.899 | **<- active** |
| `percentile_same_machine_99.5` | 1.662961 | 0.995 | 0.804 | 0.900 | 0.889 |  |
| `evt_p95_equiv` | 1.610106 | 0.974 | 0.849 | 0.913 | 0.907 |  |
| `parametric_p95_equiv` | 1.623276 | 0.987 | 0.838 | 0.913 | 0.906 |  |
| `kde_p95_equiv` | 1.616099 | 0.982 | 0.842 | 0.913 | 0.907 |  |
| `evt_p99_equiv` | 1.651598 | 0.995 | 0.823 | 0.909 | 0.901 |  |
| `parametric_p99_equiv` | 1.682747 | 0.995 | 0.762 | 0.879 | 0.863 |  |
| `kde_p99_equiv` | 1.659843 | 0.995 | 0.804 | 0.900 | 0.889 |  |
| `evt_p999_equiv` | 1.697184 | 1.000 | 0.721 | 0.860 | 0.838 |  |
| `parametric_p999_equiv` | 1.750181 | 1.000 | 0.630 | 0.815 | 0.773 |  |
| `kde_p999_equiv` | 1.710550 | 1.000 | 0.702 | 0.851 | 0.825 |  |
| `mad` | 1.814561 | 1.000 | 0.536 | 0.768 | 0.698 |  |
| `iqr` | 1.925595 | 1.000 | 0.343 | 0.672 | 0.511 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
