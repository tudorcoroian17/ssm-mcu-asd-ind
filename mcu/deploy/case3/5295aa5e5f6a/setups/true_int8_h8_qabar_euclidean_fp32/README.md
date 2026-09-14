# Deployment setup: `true_int8_h8_qabar_euclidean_fp32`

Combo 15 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
- Backbone: true int8 arithmetic, h storage = int8
- Head: euclidean head (fp32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.115695**.

At this threshold on the case 3 test split: P=0.975, R=0.449, A=0.719, F1=0.615.

Ranking quality (threshold-independent): AUC=0.7568, pAUC=0.7137.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.976733 | 0.908 | 0.487 | 0.719 | 0.634 |  |
| `percentile_same_machine_99.0` | 2.115695 | 0.975 | 0.449 | 0.719 | 0.615 | **<- active** |
| `percentile_same_machine_99.5` | 6.023961 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 1.969625 | 0.909 | 0.491 | 0.721 | 0.637 |  |
| `parametric_p95_equiv` | 2.286898 | 0.978 | 0.328 | 0.660 | 0.492 |  |
| `kde_p95_equiv` | 2.037207 | 0.939 | 0.468 | 0.719 | 0.625 |  |
| `evt_p99_equiv` | 2.414475 | 0.974 | 0.283 | 0.638 | 0.439 |  |
| `parametric_p99_equiv` | 2.723703 | 0.967 | 0.219 | 0.606 | 0.357 |  |
| `kde_p99_equiv` | 2.255786 | 0.980 | 0.362 | 0.677 | 0.529 |  |
| `evt_p999_equiv` | 5.541408 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.313221 | 0.931 | 0.102 | 0.547 | 0.184 |  |
| `kde_p999_equiv` | 7.180116 | 0.500 | 0.004 | 0.500 | 0.007 |  |
| `mad` | 2.668151 | 0.969 | 0.234 | 0.613 | 0.377 |  |
| `iqr` | 3.154760 | 0.935 | 0.109 | 0.551 | 0.196 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
