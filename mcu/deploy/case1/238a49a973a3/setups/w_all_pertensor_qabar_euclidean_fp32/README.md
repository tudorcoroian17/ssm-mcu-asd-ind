# Deployment setup: `w_all_pertensor_qabar_euclidean_fp32`

Combo 7 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: weight-only int8 (all tensors, per-tensor)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **4.384393**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8763, pAUC=0.6710.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.737449 | 0.897 | 0.462 | 0.705 | 0.610 |  |
| `percentile_same_machine_99.0` | 4.384393 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 9.382439 | 0.769 | 0.038 | 0.513 | 0.072 |  |
| `evt_p95_equiv` | 2.816004 | 0.919 | 0.386 | 0.676 | 0.544 |  |
| `parametric_p95_equiv` | 2.578365 | 0.869 | 0.527 | 0.723 | 0.656 |  |
| `kde_p95_equiv` | 2.837035 | 0.914 | 0.364 | 0.665 | 0.520 |  |
| `evt_p99_equiv` | 4.793386 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 3.089934 | 0.893 | 0.254 | 0.612 | 0.395 |  |
| `kde_p99_equiv` | 8.690998 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 13.872407 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.784924 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 9.698125 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.204481 | 0.875 | 0.716 | 0.807 | 0.787 |  |
| `iqr` | 2.407267 | 0.856 | 0.606 | 0.752 | 0.710 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
