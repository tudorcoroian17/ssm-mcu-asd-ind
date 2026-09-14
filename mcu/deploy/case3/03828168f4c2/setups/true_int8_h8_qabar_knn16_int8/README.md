# Deployment setup: `true_int8_h8_qabar_knn16_int8`

Combo 18 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.155249** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 3 test split: P=0.981, R=0.989, A=0.985, F1=0.985.

Ranking quality (threshold-independent): AUC=0.9908, pAUC=0.9937.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.029621847) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.118101 | 1425 | 0.929 | 0.989 | 0.957 | 0.958 |  |
| `percentile_same_machine_99.0` | 1.155249 | 1521 | 0.981 | 0.989 | 0.985 | 0.985 | **<- active** |
| `percentile_same_machine_99.5` | 1.181700 | 1591 | 0.992 | 0.989 | 0.991 | 0.991 |  |
| `evt_p95_equiv` | 1.113388 | 1413 | 0.923 | 0.989 | 0.953 | 0.954 |  |
| `parametric_p95_equiv` | 1.194669 | 1627 | 0.992 | 0.989 | 0.991 | 0.991 |  |
| `kde_p95_equiv` | 1.130883 | 1458 | 0.939 | 0.989 | 0.962 | 0.963 |  |
| `evt_p99_equiv` | 1.189415 | 1612 | 0.992 | 0.989 | 0.991 | 0.991 |  |
| `parametric_p99_equiv` | 1.329142 | 2013 | 0.996 | 0.974 | 0.985 | 0.985 |  |
| `kde_p99_equiv` | 1.189303 | 1612 | 0.992 | 0.989 | 0.991 | 0.991 |  |
| `evt_p999_equiv` | 1.364093 | 2121 | 1.000 | 0.962 | 0.981 | 0.981 |  |
| `parametric_p999_equiv` | 1.491071 | 2534 | 1.000 | 0.792 | 0.896 | 0.884 |  |
| `kde_p999_equiv` | 2.831193 | 9135 | 1.000 | 0.094 | 0.547 | 0.172 |  |
| `mad` | 1.450881 | 2399 | 1.000 | 0.875 | 0.938 | 0.934 |  |
| `iqr` | 1.638099 | 3058 | 1.000 | 0.302 | 0.651 | 0.464 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
