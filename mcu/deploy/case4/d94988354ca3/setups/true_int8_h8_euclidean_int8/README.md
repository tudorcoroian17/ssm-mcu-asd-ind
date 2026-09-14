# Deployment setup: `true_int8_h8_euclidean_int8`

Combo 12 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
- Backbone: true int8 arithmetic, h storage = int8
- Head: euclidean head (int32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.245622** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 4 test split: P=0.974, R=0.853, A=0.915, F1=0.909.

Ranking quality (threshold-independent): AUC=0.9838, pAUC=0.9509.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0351456207) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.082489 | 3511 | 0.930 | 0.947 | 0.938 | 0.938 |  |
| `percentile_same_machine_99.0` | 2.245622 | 4083 | 0.974 | 0.853 | 0.915 | 0.909 | **<- active** |
| `percentile_same_machine_99.5` | 2.313961 | 4335 | 0.995 | 0.811 | 0.904 | 0.894 |  |
| `evt_p95_equiv` | 2.078347 | 3497 | 0.930 | 0.947 | 0.938 | 0.938 |  |
| `parametric_p95_equiv` | 2.098952 | 3567 | 0.943 | 0.932 | 0.938 | 0.937 |  |
| `kde_p95_equiv` | 2.098074 | 3564 | 0.943 | 0.932 | 0.938 | 0.937 |  |
| `evt_p99_equiv` | 2.294029 | 4260 | 0.982 | 0.830 | 0.908 | 0.900 |  |
| `parametric_p99_equiv` | 2.276240 | 4195 | 0.978 | 0.845 | 0.913 | 0.907 |  |
| `kde_p99_equiv` | 2.275207 | 4191 | 0.978 | 0.849 | 0.915 | 0.909 |  |
| `evt_p999_equiv` | 2.651855 | 5693 | 1.000 | 0.623 | 0.811 | 0.767 |  |
| `parametric_p999_equiv` | 2.479502 | 4977 | 1.000 | 0.713 | 0.857 | 0.833 |  |
| `kde_p999_equiv` | 2.771227 | 6217 | 1.000 | 0.551 | 0.775 | 0.710 |  |
| `mad` | 2.488908 | 5015 | 1.000 | 0.702 | 0.851 | 0.825 |  |
| `iqr` | 2.741745 | 6086 | 1.000 | 0.585 | 0.792 | 0.738 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
