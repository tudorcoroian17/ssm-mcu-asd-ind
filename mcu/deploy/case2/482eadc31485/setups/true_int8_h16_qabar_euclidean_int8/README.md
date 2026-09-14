# Deployment setup: `true_int8_h16_qabar_euclidean_int8`

Combo 24 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **1.356405** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=0.984, R=0.906, A=0.945, F1=0.943.

Ranking quality (threshold-independent): AUC=0.9597, pAUC=0.9536.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.029606408) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.324499 | 2001 | 0.921 | 0.921 | 0.921 | 0.921 |  |
| `percentile_same_machine_99.0` | 1.356405 | 2099 | 0.984 | 0.906 | 0.945 | 0.943 | **<- active** |
| `percentile_same_machine_99.5` | 1.373426 | 2152 | 0.992 | 0.887 | 0.940 | 0.936 |  |
| `evt_p95_equiv` | 1.324456 | 2001 | 0.921 | 0.921 | 0.921 | 0.921 |  |
| `parametric_p95_equiv` | 1.336792 | 2039 | 0.945 | 0.913 | 0.930 | 0.929 |  |
| `kde_p95_equiv` | 1.329843 | 2018 | 0.935 | 0.917 | 0.926 | 0.926 |  |
| `evt_p99_equiv` | 1.360450 | 2112 | 0.988 | 0.902 | 0.945 | 0.943 |  |
| `parametric_p99_equiv` | 1.386131 | 2192 | 0.992 | 0.887 | 0.940 | 0.936 |  |
| `kde_p99_equiv` | 1.365396 | 2127 | 0.992 | 0.898 | 0.945 | 0.943 |  |
| `evt_p999_equiv` | 1.403034 | 2246 | 1.000 | 0.887 | 0.943 | 0.940 |  |
| `parametric_p999_equiv` | 1.442079 | 2373 | 1.000 | 0.857 | 0.928 | 0.923 |  |
| `kde_p999_equiv` | 1.415892 | 2287 | 1.000 | 0.883 | 0.942 | 0.938 |  |
| `mad` | 1.477102 | 2489 | 1.000 | 0.838 | 0.919 | 0.912 |  |
| `iqr` | 1.564919 | 2794 | 1.000 | 0.713 | 0.857 | 0.833 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
