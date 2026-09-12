# Deployment setup: `w_all_pertensor_actboundaries_qabar_knn16_fp32`

Combo 12 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.498835**.

At this threshold on the case 1 test split: P=0.982, R=0.424, A=0.708, F1=0.593.

Ranking quality (threshold-independent): AUC=0.8236, pAUC=0.7915.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.988369 | 0.942 | 0.610 | 0.786 | 0.740 |  |
| `percentile_same_machine_99.0` | 2.498835 | 0.982 | 0.424 | 0.708 | 0.593 | **<- active** |
| `percentile_same_machine_99.5` | 2.571013 | 1.000 | 0.367 | 0.684 | 0.537 |  |
| `evt_p95_equiv` | 2.054738 | 0.962 | 0.568 | 0.773 | 0.714 |  |
| `parametric_p95_equiv` | 2.031659 | 0.951 | 0.587 | 0.778 | 0.726 |  |
| `kde_p95_equiv` | 2.001949 | 0.941 | 0.606 | 0.784 | 0.737 |  |
| `evt_p99_equiv` | 2.426162 | 0.976 | 0.462 | 0.725 | 0.627 |  |
| `parametric_p99_equiv` | 2.197485 | 0.971 | 0.500 | 0.742 | 0.660 |  |
| `kde_p99_equiv` | 2.514127 | 0.991 | 0.409 | 0.703 | 0.579 |  |
| `evt_p999_equiv` | 2.725099 | 1.000 | 0.292 | 0.646 | 0.452 |  |
| `parametric_p999_equiv` | 2.366234 | 0.977 | 0.477 | 0.733 | 0.641 |  |
| `kde_p999_equiv` | 2.678134 | 1.000 | 0.326 | 0.663 | 0.491 |  |
| `mad` | 2.027392 | 0.952 | 0.595 | 0.782 | 0.732 |  |
| `iqr` | 2.202613 | 0.971 | 0.500 | 0.742 | 0.660 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
