# Deployment setup: `true_int8_h8_euclidean_int8`

Combo 12 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.084216** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.992, R=0.996, A=0.994, F1=0.994.

Ranking quality (threshold-independent): AUC=0.9926, pAUC=0.9612.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0315041016) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.964300 | 3888 | 0.953 | 1.000 | 0.975 | 0.976 |  |
| `percentile_same_machine_99.0` | 2.084216 | 4377 | 0.992 | 0.996 | 0.994 | 0.994 | **<- active** |
| `percentile_same_machine_99.5` | 4.777796 | 23000 | 0.889 | 0.060 | 0.526 | 0.113 |  |
| `evt_p95_equiv` | 1.957173 | 3859 | 0.953 | 1.000 | 0.975 | 0.976 |  |
| `parametric_p95_equiv` | 2.173037 | 4758 | 0.992 | 0.985 | 0.989 | 0.989 |  |
| `kde_p95_equiv` | 2.001092 | 4035 | 0.974 | 1.000 | 0.987 | 0.987 |  |
| `evt_p99_equiv` | 2.294050 | 5302 | 0.992 | 0.943 | 0.968 | 0.967 |  |
| `parametric_p99_equiv` | 2.439448 | 5996 | 0.991 | 0.849 | 0.921 | 0.915 |  |
| `kde_p99_equiv` | 2.166702 | 4730 | 0.992 | 0.985 | 0.989 | 0.989 |  |
| `evt_p999_equiv` | 4.383687 | 19362 | 0.923 | 0.091 | 0.542 | 0.165 |  |
| `parametric_p999_equiv` | 2.777077 | 7770 | 0.981 | 0.396 | 0.694 | 0.565 |  |
| `kde_p999_equiv` | 5.752339 | 33339 | 1.000 | 0.019 | 0.509 | 0.037 |  |
| `mad` | 2.381975 | 5717 | 0.992 | 0.906 | 0.949 | 0.947 |  |
| `iqr` | 2.699066 | 7340 | 0.984 | 0.475 | 0.734 | 0.641 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
