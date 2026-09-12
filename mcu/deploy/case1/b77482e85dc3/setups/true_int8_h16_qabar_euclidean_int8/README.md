# Deployment setup: `true_int8_h16_qabar_euclidean_int8`

Combo 26 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **3.706431** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.9251, pAUC=0.8158.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0339353216) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.702452 | 2517 | 0.950 | 0.799 | 0.879 | 0.868 |  |
| `percentile_same_machine_99.0` | 3.706431 | 11929 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 10.989132 | 104863 | 0.833 | 0.038 | 0.515 | 0.072 |  |
| `evt_p95_equiv` | 1.724557 | 2583 | 0.953 | 0.765 | 0.864 | 0.849 |  |
| `parametric_p95_equiv` | 2.028233 | 3572 | 0.949 | 0.492 | 0.733 | 0.648 |  |
| `kde_p95_equiv` | 1.904072 | 3148 | 0.958 | 0.602 | 0.788 | 0.740 |  |
| `evt_p99_equiv` | 3.371213 | 9869 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.402918 | 5014 | 0.926 | 0.189 | 0.587 | 0.314 |  |
| `kde_p99_equiv` | 10.233356 | 90935 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 32.236548 | 902388 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 2.905767 | 7332 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 11.325804 | 111387 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.755546 | 2676 | 0.960 | 0.735 | 0.852 | 0.833 |  |
| `iqr` | 1.913224 | 3179 | 0.957 | 0.591 | 0.782 | 0.731 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
