# Deployment setup: `w_all_perchannel_knn16_fp32`

Combo 2 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.215013**.

At this threshold on the case 3 test split: P=0.989, R=0.989, A=0.989, F1=0.989.

Ranking quality (threshold-independent): AUC=0.9887, pAUC=0.9940.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.139819 | 0.953 | 0.989 | 0.970 | 0.970 |  |
| `percentile_same_machine_99.0` | 1.215013 | 0.989 | 0.989 | 0.989 | 0.989 | **<- active** |
| `percentile_same_machine_99.5` | 1.235256 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `evt_p95_equiv` | 1.135995 | 0.949 | 0.989 | 0.968 | 0.969 |  |
| `parametric_p95_equiv` | 1.207981 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `kde_p95_equiv` | 1.149159 | 0.956 | 0.989 | 0.972 | 0.972 |  |
| `evt_p99_equiv` | 1.229500 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `parametric_p99_equiv` | 1.310844 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `kde_p99_equiv` | 1.226149 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `evt_p999_equiv` | 1.459074 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `parametric_p999_equiv` | 1.428792 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `kde_p999_equiv` | 2.561905 | 1.000 | 0.151 | 0.575 | 0.262 |  |
| `mad` | 1.356282 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `iqr` | 1.500126 | 1.000 | 0.989 | 0.994 | 0.994 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
