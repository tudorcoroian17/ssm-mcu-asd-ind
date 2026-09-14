# Deployment setup: `w_all_perchannel_euclidean_fp32`

Combo 1 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.864255**.

At this threshold on the case 4 test split: P=0.983, R=0.879, A=0.932, F1=0.928.

Ranking quality (threshold-independent): AUC=0.9680, pAUC=0.9433.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.819682 | 0.912 | 0.902 | 0.908 | 0.907 |  |
| `percentile_same_machine_99.0` | 1.864255 | 0.983 | 0.879 | 0.932 | 0.928 | **<- active** |
| `percentile_same_machine_99.5` | 1.875395 | 0.983 | 0.872 | 0.928 | 0.924 |  |
| `evt_p95_equiv` | 1.817334 | 0.909 | 0.906 | 0.908 | 0.907 |  |
| `parametric_p95_equiv` | 1.824966 | 0.926 | 0.902 | 0.915 | 0.914 |  |
| `kde_p95_equiv` | 1.833620 | 0.937 | 0.898 | 0.919 | 0.917 |  |
| `evt_p99_equiv` | 1.872653 | 0.983 | 0.875 | 0.930 | 0.926 |  |
| `parametric_p99_equiv` | 1.895706 | 0.996 | 0.857 | 0.926 | 0.921 |  |
| `kde_p99_equiv` | 1.896168 | 0.996 | 0.857 | 0.926 | 0.921 |  |
| `evt_p999_equiv` | 1.923537 | 0.996 | 0.845 | 0.921 | 0.914 |  |
| `parametric_p999_equiv` | 1.964922 | 1.000 | 0.826 | 0.913 | 0.905 |  |
| `kde_p999_equiv` | 1.960138 | 1.000 | 0.826 | 0.913 | 0.905 |  |
| `mad` | 2.145069 | 1.000 | 0.702 | 0.851 | 0.825 |  |
| `iqr` | 2.349693 | 1.000 | 0.555 | 0.777 | 0.714 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
