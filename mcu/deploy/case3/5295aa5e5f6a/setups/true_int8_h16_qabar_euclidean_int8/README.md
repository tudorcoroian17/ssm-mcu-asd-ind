# Deployment setup: `true_int8_h16_qabar_euclidean_int8`

Combo 24 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.465450** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.984, R=0.468, A=0.730, F1=0.634.

Ranking quality (threshold-independent): AUC=0.8407, pAUC=0.7610.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0347989075) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.289547 | 4329 | 0.952 | 0.592 | 0.781 | 0.730 |  |
| `percentile_same_machine_99.0` | 2.465450 | 5020 | 0.984 | 0.468 | 0.730 | 0.634 | **<- active** |
| `percentile_same_machine_99.5` | 8.354892 | 57644 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.276430 | 4279 | 0.935 | 0.592 | 0.775 | 0.725 |  |
| `parametric_p95_equiv` | 2.608442 | 5619 | 0.984 | 0.460 | 0.726 | 0.627 |  |
| `kde_p95_equiv` | 2.378706 | 4673 | 0.973 | 0.547 | 0.766 | 0.700 |  |
| `evt_p99_equiv` | 2.834701 | 6636 | 0.976 | 0.306 | 0.649 | 0.466 |  |
| `parametric_p99_equiv` | 3.063431 | 7750 | 0.973 | 0.275 | 0.634 | 0.429 |  |
| `kde_p99_equiv` | 2.667692 | 5877 | 0.983 | 0.442 | 0.717 | 0.609 |  |
| `evt_p999_equiv` | 7.513861 | 46622 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.668401 | 11113 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `kde_p999_equiv` | 10.361415 | 88656 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 3.015240 | 7508 | 0.974 | 0.279 | 0.636 | 0.434 |  |
| `iqr` | 3.475075 | 9972 | 0.966 | 0.211 | 0.602 | 0.347 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
