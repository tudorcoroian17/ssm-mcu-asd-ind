# Deployment setup: `w_proj_perchannel_euclidean_fp32`

Combo 5 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: weight-only int8 (projections tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.438584**.

At this threshold on the case 2 test split: P=0.988, R=0.909, A=0.949, F1=0.947.

Ranking quality (threshold-independent): AUC=0.9634, pAUC=0.9619.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.407231 | 0.957 | 0.928 | 0.943 | 0.943 |  |
| `percentile_same_machine_99.0` | 1.438584 | 0.988 | 0.909 | 0.949 | 0.947 | **<- active** |
| `percentile_same_machine_99.5` | 1.447059 | 0.992 | 0.909 | 0.951 | 0.949 |  |
| `evt_p95_equiv` | 1.404717 | 0.957 | 0.928 | 0.943 | 0.943 |  |
| `parametric_p95_equiv` | 1.420501 | 0.980 | 0.913 | 0.947 | 0.945 |  |
| `kde_p95_equiv` | 1.413328 | 0.976 | 0.925 | 0.951 | 0.950 |  |
| `evt_p99_equiv` | 1.442028 | 0.992 | 0.909 | 0.951 | 0.949 |  |
| `parametric_p99_equiv` | 1.484991 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `kde_p99_equiv` | 1.449933 | 0.992 | 0.909 | 0.951 | 0.949 |  |
| `evt_p999_equiv` | 1.489045 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `parametric_p999_equiv` | 1.558285 | 1.000 | 0.845 | 0.923 | 0.916 |  |
| `kde_p999_equiv` | 1.507287 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `mad` | 1.658249 | 1.000 | 0.743 | 0.872 | 0.853 |  |
| `iqr` | 1.818616 | 1.000 | 0.521 | 0.760 | 0.685 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
