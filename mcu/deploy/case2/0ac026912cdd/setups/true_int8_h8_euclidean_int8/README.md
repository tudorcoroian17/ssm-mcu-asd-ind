# Deployment setup: `true_int8_h8_euclidean_int8`

Combo 12 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.020629** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=0.982, R=0.842, A=0.913, F1=0.907.

Ranking quality (threshold-independent): AUC=0.9711, pAUC=0.9412.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0324503718) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.939537 | 3572 | 0.955 | 0.891 | 0.925 | 0.922 |  |
| `percentile_same_machine_99.0` | 2.020629 | 3877 | 0.982 | 0.842 | 0.913 | 0.907 | **<- active** |
| `percentile_same_machine_99.5` | 2.071807 | 4076 | 0.995 | 0.811 | 0.904 | 0.894 |  |
| `evt_p95_equiv` | 1.938575 | 3569 | 0.955 | 0.891 | 0.925 | 0.922 |  |
| `parametric_p95_equiv` | 1.943649 | 3588 | 0.955 | 0.887 | 0.923 | 0.920 |  |
| `kde_p95_equiv` | 1.945886 | 3596 | 0.955 | 0.887 | 0.923 | 0.920 |  |
| `evt_p99_equiv` | 2.032515 | 3923 | 0.987 | 0.838 | 0.913 | 0.906 |  |
| `parametric_p99_equiv` | 2.032016 | 3921 | 0.987 | 0.838 | 0.913 | 0.906 |  |
| `kde_p99_equiv` | 2.039790 | 3951 | 0.987 | 0.834 | 0.911 | 0.904 |  |
| `evt_p999_equiv` | 2.159774 | 4430 | 1.000 | 0.732 | 0.866 | 0.845 |  |
| `parametric_p999_equiv` | 2.134164 | 4325 | 1.000 | 0.758 | 0.879 | 0.863 |  |
| `kde_p999_equiv` | 2.169658 | 4470 | 1.000 | 0.725 | 0.862 | 0.840 |  |
| `mad` | 2.165556 | 4453 | 1.000 | 0.728 | 0.864 | 0.843 |  |
| `iqr` | 2.303194 | 5038 | 1.000 | 0.543 | 0.772 | 0.704 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
