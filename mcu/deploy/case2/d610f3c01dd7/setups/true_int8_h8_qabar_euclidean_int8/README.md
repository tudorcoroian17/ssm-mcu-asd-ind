# Deployment setup: `true_int8_h8_qabar_euclidean_int8`

Combo 16 of the case 2 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 2
- Model hash: `d610f3c01dd7`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.259913** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=1.000, R=0.770, A=0.885, F1=0.870.

Ranking quality (threshold-independent): AUC=0.9214, pAUC=0.9185.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0341339825) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.224315 | 1287 | 0.982 | 0.830 | 0.908 | 0.900 |  |
| `percentile_same_machine_99.0` | 1.259913 | 1362 | 1.000 | 0.770 | 0.885 | 0.870 | **<- active** |
| `percentile_same_machine_99.5` | 1.269329 | 1383 | 1.000 | 0.755 | 0.877 | 0.860 |  |
| `evt_p95_equiv` | 1.221470 | 1281 | 0.978 | 0.838 | 0.909 | 0.902 |  |
| `parametric_p95_equiv` | 1.230401 | 1299 | 0.986 | 0.819 | 0.904 | 0.895 |  |
| `kde_p95_equiv` | 1.226273 | 1291 | 0.982 | 0.830 | 0.908 | 0.900 |  |
| `evt_p99_equiv` | 1.261330 | 1365 | 1.000 | 0.770 | 0.885 | 0.870 |  |
| `parametric_p99_equiv` | 1.280701 | 1408 | 1.000 | 0.736 | 0.868 | 0.848 |  |
| `kde_p99_equiv` | 1.264543 | 1372 | 1.000 | 0.770 | 0.885 | 0.870 |  |
| `evt_p999_equiv` | 1.293914 | 1437 | 1.000 | 0.698 | 0.849 | 0.822 |  |
| `parametric_p999_equiv` | 1.337801 | 1536 | 1.000 | 0.619 | 0.809 | 0.765 |  |
| `kde_p999_equiv` | 1.304980 | 1462 | 1.000 | 0.679 | 0.840 | 0.809 |  |
| `mad` | 1.362226 | 1593 | 1.000 | 0.570 | 0.785 | 0.726 |  |
| `iqr` | 1.443748 | 1789 | 1.000 | 0.442 | 0.721 | 0.613 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
