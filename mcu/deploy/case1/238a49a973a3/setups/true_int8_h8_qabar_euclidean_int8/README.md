# Deployment setup: `true_int8_h8_qabar_euclidean_int8`

Combo 18 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **4.025689** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8921, pAUC=0.6565.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0329585263) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.778137 | 7105 | 0.888 | 0.360 | 0.657 | 0.512 |  |
| `percentile_same_machine_99.0` | 4.025689 | 14919 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 7.855189 | 56804 | 0.667 | 0.015 | 0.504 | 0.030 |  |
| `evt_p95_equiv` | 2.817579 | 7308 | 0.907 | 0.333 | 0.650 | 0.488 |  |
| `parametric_p95_equiv` | 2.564423 | 6054 | 0.853 | 0.572 | 0.737 | 0.685 |  |
| `kde_p95_equiv` | 2.848339 | 7469 | 0.912 | 0.314 | 0.642 | 0.468 |  |
| `evt_p99_equiv` | 4.295463 | 16986 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 3.005537 | 8316 | 0.899 | 0.269 | 0.619 | 0.414 |  |
| `kde_p99_equiv` | 7.247778 | 48359 | 0.810 | 0.064 | 0.525 | 0.119 |  |
| `evt_p999_equiv` | 11.600028 | 123875 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.590775 | 11870 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 8.088637 | 60230 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.174087 | 4351 | 0.880 | 0.780 | 0.837 | 0.827 |  |
| `iqr` | 2.342041 | 5050 | 0.872 | 0.697 | 0.797 | 0.775 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
