# Deployment setup: `w_all_pertensor_qabar_euclidean_fp32`

Combo 7 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.373986**.

At this threshold on the case 3 test split: P=0.987, R=0.555, A=0.774, F1=0.710.

Ranking quality (threshold-independent): AUC=0.9163, pAUC=0.8020.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.201900 | 0.939 | 0.638 | 0.798 | 0.760 |  |
| `percentile_same_machine_99.0` | 2.373986 | 0.987 | 0.555 | 0.774 | 0.710 | **<- active** |
| `percentile_same_machine_99.5` | 8.485209 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.184036 | 0.935 | 0.653 | 0.804 | 0.769 |  |
| `parametric_p95_equiv` | 2.501000 | 0.985 | 0.487 | 0.740 | 0.652 |  |
| `kde_p95_equiv` | 2.291220 | 0.976 | 0.604 | 0.794 | 0.746 |  |
| `evt_p99_equiv` | 2.752597 | 0.982 | 0.408 | 0.700 | 0.576 |  |
| `parametric_p99_equiv` | 2.955914 | 0.974 | 0.287 | 0.640 | 0.443 |  |
| `kde_p99_equiv` | 2.588793 | 0.984 | 0.457 | 0.725 | 0.624 |  |
| `evt_p999_equiv` | 8.042839 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.564875 | 0.963 | 0.196 | 0.594 | 0.326 |  |
| `kde_p999_equiv` | 10.436555 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.965058 | 0.974 | 0.283 | 0.638 | 0.439 |  |
| `iqr` | 3.372438 | 0.968 | 0.226 | 0.609 | 0.367 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
