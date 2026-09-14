# Deployment setup: `true_int8_h8_qabar_knn16_fp32`

Combo 17 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.358761**.

At this threshold on the case 2 test split: P=0.962, R=0.483, A=0.732, F1=0.643.

Ranking quality (threshold-independent): AUC=0.8921, pAUC=0.8031.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.249969 | 0.920 | 0.694 | 0.817 | 0.791 |  |
| `percentile_same_machine_99.0` | 1.358761 | 0.962 | 0.483 | 0.732 | 0.643 | **<- active** |
| `percentile_same_machine_99.5` | 1.388387 | 0.983 | 0.434 | 0.713 | 0.602 |  |
| `evt_p95_equiv` | 1.252684 | 0.920 | 0.691 | 0.815 | 0.789 |  |
| `parametric_p95_equiv` | 1.252216 | 0.920 | 0.694 | 0.817 | 0.791 |  |
| `kde_p95_equiv` | 1.258750 | 0.919 | 0.687 | 0.813 | 0.786 |  |
| `evt_p99_equiv` | 1.349180 | 0.964 | 0.506 | 0.743 | 0.663 |  |
| `parametric_p99_equiv` | 1.353877 | 0.964 | 0.502 | 0.742 | 0.660 |  |
| `kde_p99_equiv` | 1.362340 | 0.962 | 0.472 | 0.726 | 0.633 |  |
| `evt_p999_equiv` | 1.438653 | 0.990 | 0.374 | 0.685 | 0.542 |  |
| `parametric_p999_equiv` | 1.474052 | 1.000 | 0.306 | 0.653 | 0.468 |  |
| `kde_p999_equiv` | 1.449628 | 0.990 | 0.358 | 0.677 | 0.526 |  |
| `mad` | 1.461695 | 1.000 | 0.328 | 0.664 | 0.494 |  |
| `iqr` | 1.603459 | 1.000 | 0.125 | 0.562 | 0.221 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
