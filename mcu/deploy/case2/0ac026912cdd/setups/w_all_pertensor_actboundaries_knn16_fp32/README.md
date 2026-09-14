# Deployment setup: `w_all_pertensor_actboundaries_knn16_fp32`

Combo 10 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.276028**.

At this threshold on the case 2 test split: P=0.985, R=0.974, A=0.979, F1=0.979.

Ranking quality (threshold-independent): AUC=0.9828, pAUC=0.9856.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.245597 | 0.963 | 0.974 | 0.968 | 0.968 |  |
| `percentile_same_machine_99.0` | 1.276028 | 0.985 | 0.974 | 0.979 | 0.979 | **<- active** |
| `percentile_same_machine_99.5` | 1.278822 | 0.989 | 0.974 | 0.981 | 0.981 |  |
| `evt_p95_equiv` | 1.246994 | 0.963 | 0.974 | 0.968 | 0.968 |  |
| `parametric_p95_equiv` | 1.250887 | 0.966 | 0.974 | 0.970 | 0.970 |  |
| `kde_p95_equiv` | 1.250808 | 0.966 | 0.974 | 0.970 | 0.970 |  |
| `evt_p99_equiv` | 1.277258 | 0.985 | 0.974 | 0.979 | 0.979 |  |
| `parametric_p99_equiv` | 1.301538 | 0.996 | 0.966 | 0.981 | 0.981 |  |
| `kde_p99_equiv` | 1.284269 | 0.989 | 0.974 | 0.981 | 0.981 |  |
| `evt_p999_equiv` | 1.304713 | 0.996 | 0.966 | 0.981 | 0.981 |  |
| `parametric_p999_equiv` | 1.360754 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `kde_p999_equiv` | 1.315947 | 1.000 | 0.966 | 0.983 | 0.983 |  |
| `mad` | 1.398942 | 1.000 | 0.943 | 0.972 | 0.971 |  |
| `iqr` | 1.502060 | 1.000 | 0.883 | 0.942 | 0.938 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
