# Deployment setup: `w_all_perchannel_qabar_euclidean_fp32`

Combo 3 of the case 4 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 4
- Model hash: `3660d1d10006`
- Backbone: weight-only int8 (all tensors, per-channel)
- Head: euclidean head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.089769**.

At this threshold on the case 4 test split: P=0.993, R=1.000, A=0.996, F1=0.996.

Ranking quality (threshold-independent): AUC=1.0000, pAUC=1.0000.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.066369 | 0.920 | 1.000 | 0.957 | 0.958 |  |
| `percentile_same_machine_99.0` | 1.089769 | 0.993 | 1.000 | 0.996 | 0.996 | **<- active** |
| `percentile_same_machine_99.5` | 1.098376 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `evt_p95_equiv` | 1.067234 | 0.930 | 1.000 | 0.962 | 0.964 |  |
| `parametric_p95_equiv` | 1.068363 | 0.936 | 1.000 | 0.966 | 0.967 |  |
| `kde_p95_equiv` | 1.070208 | 0.946 | 1.000 | 0.972 | 0.972 |  |
| `evt_p99_equiv` | 1.091801 | 0.993 | 1.000 | 0.996 | 0.996 |  |
| `parametric_p99_equiv` | 1.100651 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `kde_p99_equiv` | 1.093997 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `evt_p999_equiv` | 1.120697 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `parametric_p999_equiv` | 1.138004 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `kde_p999_equiv` | 1.125345 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `mad` | 1.152926 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `iqr` | 1.215568 | 1.000 | 1.000 | 1.000 | 1.000 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
