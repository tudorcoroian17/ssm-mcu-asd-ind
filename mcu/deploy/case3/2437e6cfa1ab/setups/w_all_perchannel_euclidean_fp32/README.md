# Deployment setup: `w_all_perchannel_euclidean_fp32`

Combo 1 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.922504**.

At this threshold on the case 3 test split: P=0.989, R=0.981, A=0.985, F1=0.985.

Ranking quality (threshold-independent): AUC=0.9927, pAUC=0.9617.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.638472 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `percentile_same_machine_99.0` | 1.922504 | 0.989 | 0.981 | 0.985 | 0.985 | **<- active** |
| `percentile_same_machine_99.5` | 3.843431 | 0.935 | 0.109 | 0.551 | 0.196 |  |
| `evt_p95_equiv` | 1.633697 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `parametric_p95_equiv` | 1.743341 | 0.967 | 1.000 | 0.983 | 0.983 |  |
| `kde_p95_equiv` | 1.672131 | 0.957 | 1.000 | 0.977 | 0.978 |  |
| `evt_p99_equiv` | 2.070695 | 0.992 | 0.936 | 0.964 | 0.963 |  |
| `parametric_p99_equiv` | 1.948037 | 0.989 | 0.977 | 0.983 | 0.983 |  |
| `kde_p99_equiv` | 1.969400 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `evt_p999_equiv` | 5.463915 | 1.000 | 0.034 | 0.517 | 0.066 |  |
| `parametric_p999_equiv` | 2.206181 | 0.990 | 0.785 | 0.889 | 0.876 |  |
| `kde_p999_equiv` | 4.793717 | 1.000 | 0.045 | 0.523 | 0.087 |  |
| `mad` | 1.977096 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `iqr` | 2.233419 | 0.990 | 0.740 | 0.866 | 0.847 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
