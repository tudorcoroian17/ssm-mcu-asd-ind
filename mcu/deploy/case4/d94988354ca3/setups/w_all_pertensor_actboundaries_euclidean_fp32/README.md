# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.816719**.

At this threshold on the case 4 test split: P=0.971, R=0.898, A=0.936, F1=0.933.

Ranking quality (threshold-independent): AUC=0.9757, pAUC=0.9468.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.773006 | 0.914 | 0.917 | 0.915 | 0.915 |  |
| `percentile_same_machine_99.0` | 1.816719 | 0.971 | 0.898 | 0.936 | 0.933 | **<- active** |
| `percentile_same_machine_99.5` | 1.834131 | 0.983 | 0.891 | 0.938 | 0.935 |  |
| `evt_p95_equiv` | 1.773518 | 0.914 | 0.917 | 0.915 | 0.915 |  |
| `parametric_p95_equiv` | 1.778591 | 0.917 | 0.917 | 0.917 | 0.917 |  |
| `kde_p95_equiv` | 1.787258 | 0.934 | 0.906 | 0.921 | 0.920 |  |
| `evt_p99_equiv` | 1.823469 | 0.979 | 0.894 | 0.938 | 0.935 |  |
| `parametric_p99_equiv` | 1.839142 | 0.987 | 0.887 | 0.938 | 0.934 |  |
| `kde_p99_equiv` | 1.844665 | 0.987 | 0.883 | 0.936 | 0.932 |  |
| `evt_p999_equiv` | 1.868595 | 0.996 | 0.860 | 0.928 | 0.923 |  |
| `parametric_p999_equiv` | 1.898127 | 0.995 | 0.834 | 0.915 | 0.908 |  |
| `kde_p999_equiv` | 1.903353 | 0.995 | 0.830 | 0.913 | 0.905 |  |
| `mad` | 2.035279 | 1.000 | 0.740 | 0.870 | 0.850 |  |
| `iqr` | 2.197371 | 1.000 | 0.645 | 0.823 | 0.784 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
