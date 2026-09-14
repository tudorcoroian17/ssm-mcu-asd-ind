# Deployment setup: `true_int8_h16_knn16_int8`

Combo 18 of the case 4 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 4
- Model hash: `733b93bfdb24`
- Backbone: true int8 arithmetic, h storage = int16
- Head: knn_clustered_16 head (int32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.975710** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 4 test split: P=0.992, R=0.966, A=0.979, F1=0.979.

Ranking quality (threshold-independent): AUC=0.9908, pAUC=0.9892.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0386070679) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.912115 | 2453 | 0.959 | 0.977 | 0.968 | 0.968 |  |
| `percentile_same_machine_99.0` | 1.975710 | 2619 | 0.992 | 0.966 | 0.979 | 0.979 | **<- active** |
| `percentile_same_machine_99.5` | 2.017809 | 2732 | 0.996 | 0.947 | 0.972 | 0.971 |  |
| `evt_p95_equiv` | 1.911274 | 2451 | 0.959 | 0.977 | 0.968 | 0.968 |  |
| `parametric_p95_equiv` | 1.903632 | 2431 | 0.956 | 0.981 | 0.968 | 0.968 |  |
| `kde_p95_equiv` | 1.917810 | 2468 | 0.963 | 0.977 | 0.970 | 0.970 |  |
| `evt_p99_equiv` | 1.990737 | 2659 | 0.992 | 0.962 | 0.977 | 0.977 |  |
| `parametric_p99_equiv` | 2.015122 | 2724 | 0.992 | 0.947 | 0.970 | 0.969 |  |
| `kde_p99_equiv` | 1.997019 | 2676 | 0.992 | 0.958 | 0.975 | 0.975 |  |
| `evt_p999_equiv` | 2.051985 | 2825 | 1.000 | 0.940 | 0.970 | 0.969 |  |
| `parametric_p999_equiv` | 2.147870 | 3095 | 1.000 | 0.887 | 0.943 | 0.940 |  |
| `kde_p999_equiv` | 2.074983 | 2889 | 1.000 | 0.928 | 0.964 | 0.963 |  |
| `mad` | 2.188315 | 3213 | 1.000 | 0.868 | 0.934 | 0.929 |  |
| `iqr` | 2.382343 | 3808 | 1.000 | 0.698 | 0.849 | 0.822 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
