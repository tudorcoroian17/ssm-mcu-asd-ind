# Deployment setup: `true_int8_h16_qabar_knn16_fp32`

Combo 27 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: true int8 arithmetic, h storage = int16
- Head: knn_clustered_16 head (fp32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.366976**.

At this threshold on the case 1 test split: P=1.000, R=0.447, A=0.723, F1=0.618.

Ranking quality (threshold-independent): AUC=0.8421, pAUC=0.8192.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.886141 | 0.946 | 0.663 | 0.812 | 0.780 |  |
| `percentile_same_machine_99.0` | 2.366976 | 1.000 | 0.447 | 0.723 | 0.618 | **<- active** |
| `percentile_same_machine_99.5` | 2.414494 | 1.000 | 0.413 | 0.706 | 0.584 |  |
| `evt_p95_equiv` | 1.910999 | 0.949 | 0.640 | 0.803 | 0.765 |  |
| `parametric_p95_equiv` | 1.939900 | 0.960 | 0.636 | 0.805 | 0.765 |  |
| `kde_p95_equiv` | 1.897298 | 0.951 | 0.655 | 0.811 | 0.776 |  |
| `evt_p99_equiv` | 2.270004 | 0.977 | 0.477 | 0.733 | 0.641 |  |
| `parametric_p99_equiv` | 2.084807 | 0.972 | 0.534 | 0.759 | 0.689 |  |
| `kde_p99_equiv` | 2.365536 | 1.000 | 0.447 | 0.723 | 0.618 |  |
| `evt_p999_equiv` | 2.913121 | 1.000 | 0.148 | 0.574 | 0.257 |  |
| `parametric_p999_equiv` | 2.231350 | 0.977 | 0.485 | 0.737 | 0.648 |  |
| `kde_p999_equiv` | 2.519682 | 1.000 | 0.348 | 0.674 | 0.517 |  |
| `mad` | 1.998870 | 0.975 | 0.598 | 0.792 | 0.742 |  |
| `iqr` | 2.160669 | 0.971 | 0.504 | 0.744 | 0.663 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
