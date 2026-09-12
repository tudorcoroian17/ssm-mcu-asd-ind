# Deployment setup: `true_int8_h8_qabar_knn16_int8`

Combo 20 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.512042** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 1 test split: P=0.982, R=0.625, A=0.807, F1=0.764.

Ranking quality (threshold-independent): AUC=0.8858, pAUC=0.8519.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0339353216) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.380625 | 1655 | 0.925 | 0.792 | 0.864 | 0.853 |  |
| `percentile_same_machine_99.0` | 1.512042 | 1985 | 0.982 | 0.625 | 0.807 | 0.764 | **<- active** |
| `percentile_same_machine_99.5` | 1.659510 | 2391 | 0.986 | 0.542 | 0.767 | 0.699 |  |
| `evt_p95_equiv` | 1.382867 | 1661 | 0.928 | 0.784 | 0.862 | 0.850 |  |
| `parametric_p95_equiv` | 1.392805 | 1685 | 0.940 | 0.769 | 0.860 | 0.846 |  |
| `kde_p95_equiv` | 1.389892 | 1677 | 0.940 | 0.773 | 0.862 | 0.848 |  |
| `evt_p99_equiv` | 1.542088 | 2065 | 0.981 | 0.602 | 0.795 | 0.746 |  |
| `parametric_p99_equiv` | 1.474926 | 1889 | 0.972 | 0.655 | 0.818 | 0.783 |  |
| `kde_p99_equiv` | 1.518518 | 2002 | 0.982 | 0.617 | 0.803 | 0.758 |  |
| `evt_p999_equiv` | 1.945620 | 3287 | 0.978 | 0.341 | 0.667 | 0.506 |  |
| `parametric_p999_equiv` | 1.572743 | 2148 | 0.987 | 0.576 | 0.784 | 0.727 |  |
| `kde_p999_equiv` | 2.098003 | 3822 | 0.962 | 0.193 | 0.593 | 0.322 |  |
| `mad` | 1.557041 | 2105 | 0.981 | 0.591 | 0.790 | 0.738 |  |
| `iqr` | 1.682188 | 2457 | 0.986 | 0.538 | 0.765 | 0.696 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
