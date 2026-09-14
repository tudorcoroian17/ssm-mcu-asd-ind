# Deployment setup: `true_int8_h16_euclidean_int8`

Combo 16 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
- Backbone: true int8 arithmetic, h storage = int16
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.879577** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=0.983, R=0.894, A=0.940, F1=0.937.

Ranking quality (threshold-independent): AUC=0.9742, pAUC=0.9591.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0324503718) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.848088 | 3243 | 0.946 | 0.925 | 0.936 | 0.935 |  |
| `percentile_same_machine_99.0` | 1.879577 | 3355 | 0.983 | 0.894 | 0.940 | 0.937 | **<- active** |
| `percentile_same_machine_99.5` | 1.892609 | 3402 | 0.991 | 0.879 | 0.936 | 0.932 |  |
| `evt_p95_equiv` | 1.848848 | 3246 | 0.946 | 0.925 | 0.936 | 0.935 |  |
| `parametric_p95_equiv` | 1.851325 | 3255 | 0.957 | 0.925 | 0.942 | 0.940 |  |
| `kde_p95_equiv` | 1.853008 | 3261 | 0.961 | 0.925 | 0.943 | 0.942 |  |
| `evt_p99_equiv` | 1.887078 | 3382 | 0.987 | 0.891 | 0.940 | 0.937 |  |
| `parametric_p99_equiv` | 1.908634 | 3459 | 1.000 | 0.868 | 0.934 | 0.929 |  |
| `kde_p99_equiv` | 1.892783 | 3402 | 0.991 | 0.879 | 0.936 | 0.932 |  |
| `evt_p999_equiv` | 1.914937 | 3482 | 1.000 | 0.857 | 0.928 | 0.923 |  |
| `parametric_p999_equiv` | 1.973511 | 3699 | 1.000 | 0.800 | 0.900 | 0.889 |  |
| `kde_p999_equiv` | 1.931718 | 3544 | 1.000 | 0.842 | 0.921 | 0.914 |  |
| `mad` | 2.006582 | 3824 | 1.000 | 0.755 | 0.877 | 0.860 |  |
| `iqr` | 2.108851 | 4223 | 1.000 | 0.630 | 0.815 | 0.773 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
