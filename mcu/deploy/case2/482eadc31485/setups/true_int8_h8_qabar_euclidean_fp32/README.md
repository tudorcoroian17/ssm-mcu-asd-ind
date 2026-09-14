# Deployment setup: `true_int8_h8_qabar_euclidean_fp32`

Combo 15 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.492372**.

At this threshold on the case 2 test split: P=0.984, R=0.479, A=0.736, F1=0.645.

Ranking quality (threshold-independent): AUC=0.9054, pAUC=0.8138.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.371427 | 0.922 | 0.709 | 0.825 | 0.802 |  |
| `percentile_same_machine_99.0` | 1.492372 | 0.984 | 0.479 | 0.736 | 0.645 | **<- active** |
| `percentile_same_machine_99.5` | 1.530481 | 0.991 | 0.426 | 0.711 | 0.596 |  |
| `evt_p95_equiv` | 1.370313 | 0.922 | 0.709 | 0.825 | 0.802 |  |
| `parametric_p95_equiv` | 1.371702 | 0.922 | 0.709 | 0.825 | 0.802 |  |
| `kde_p95_equiv` | 1.377684 | 0.920 | 0.694 | 0.817 | 0.791 |  |
| `evt_p99_equiv` | 1.477960 | 0.973 | 0.536 | 0.760 | 0.691 |  |
| `parametric_p99_equiv` | 1.471856 | 0.966 | 0.543 | 0.762 | 0.696 |  |
| `kde_p99_equiv` | 1.494943 | 0.984 | 0.475 | 0.734 | 0.641 |  |
| `evt_p999_equiv` | 1.594823 | 1.000 | 0.340 | 0.670 | 0.507 |  |
| `parametric_p999_equiv` | 1.586422 | 1.000 | 0.351 | 0.675 | 0.520 |  |
| `kde_p999_equiv` | 1.602810 | 1.000 | 0.336 | 0.668 | 0.503 |  |
| `mad` | 1.607174 | 1.000 | 0.332 | 0.666 | 0.499 |  |
| `iqr` | 1.768929 | 1.000 | 0.109 | 0.555 | 0.197 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
