# Deployment setup: `w_all_pertensor_qabar_knn16_fp32`

Combo 8 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: weight-only int8 (all tensors, per-tensor)
- Head: knn_clustered_16 head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.494819**.

At this threshold on the case 1 test split: P=0.984, R=0.458, A=0.725, F1=0.625.

Ranking quality (threshold-independent): AUC=0.8332, pAUC=0.8054.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.960308 | 0.944 | 0.640 | 0.801 | 0.763 |  |
| `percentile_same_machine_99.0` | 2.494819 | 0.984 | 0.458 | 0.725 | 0.625 | **<- active** |
| `percentile_same_machine_99.5` | 2.555191 | 0.991 | 0.413 | 0.705 | 0.583 |  |
| `evt_p95_equiv` | 2.028977 | 0.964 | 0.606 | 0.792 | 0.744 |  |
| `parametric_p95_equiv` | 2.024230 | 0.964 | 0.606 | 0.792 | 0.744 |  |
| `kde_p95_equiv` | 1.981405 | 0.943 | 0.629 | 0.795 | 0.755 |  |
| `evt_p99_equiv` | 2.413021 | 0.976 | 0.466 | 0.727 | 0.631 |  |
| `parametric_p99_equiv` | 2.189444 | 0.971 | 0.515 | 0.750 | 0.673 |  |
| `kde_p99_equiv` | 2.508332 | 0.991 | 0.436 | 0.716 | 0.605 |  |
| `evt_p999_equiv` | 2.791704 | 1.000 | 0.261 | 0.631 | 0.414 |  |
| `parametric_p999_equiv` | 2.357568 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `kde_p999_equiv` | 2.671723 | 1.000 | 0.345 | 0.672 | 0.513 |  |
| `mad` | 2.011577 | 0.958 | 0.606 | 0.790 | 0.742 |  |
| `iqr` | 2.179152 | 0.972 | 0.527 | 0.756 | 0.683 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
