# Deployment setup: `w_all_perchannel_qabar_euclidean_fp32`

Combo 3 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.806403**.

At this threshold on the case 3 test split: P=0.959, R=0.264, A=0.626, F1=0.414.

Ranking quality (threshold-independent): AUC=0.7837, pAUC=0.6742.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.475754 | 0.919 | 0.430 | 0.696 | 0.586 |  |
| `percentile_same_machine_99.0` | 2.806403 | 0.959 | 0.264 | 0.626 | 0.414 | **<- active** |
| `percentile_same_machine_99.5` | 10.357801 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.453091 | 0.905 | 0.430 | 0.692 | 0.583 |  |
| `parametric_p95_equiv` | 2.826516 | 0.959 | 0.264 | 0.626 | 0.414 |  |
| `kde_p95_equiv` | 2.563892 | 0.936 | 0.389 | 0.681 | 0.549 |  |
| `evt_p99_equiv` | 3.384818 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `parametric_p99_equiv` | 3.485722 | 0.951 | 0.147 | 0.570 | 0.255 |  |
| `kde_p99_equiv` | 3.030517 | 0.967 | 0.223 | 0.608 | 0.362 |  |
| `evt_p999_equiv` | 10.124599 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 4.409006 | 0.926 | 0.094 | 0.543 | 0.171 |  |
| `kde_p999_equiv` | 11.879941 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 3.403541 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `iqr` | 4.029786 | 0.935 | 0.109 | 0.551 | 0.196 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
