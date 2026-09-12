# Deployment setup: `true_int8_h16_qabar_euclidean_fp32`

Combo 25 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: true int8 arithmetic, h storage = int16
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **4.268182**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8776, pAUC=0.6784.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.604491 | 0.915 | 0.451 | 0.705 | 0.604 |  |
| `percentile_same_machine_99.0` | 4.268182 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 9.521031 | 0.786 | 0.042 | 0.515 | 0.079 |  |
| `evt_p95_equiv` | 2.675742 | 0.916 | 0.371 | 0.669 | 0.528 |  |
| `parametric_p95_equiv` | 2.517579 | 0.883 | 0.485 | 0.710 | 0.626 |  |
| `kde_p95_equiv` | 2.699977 | 0.922 | 0.356 | 0.663 | 0.514 |  |
| `evt_p99_equiv` | 4.590365 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 3.000179 | 0.903 | 0.246 | 0.610 | 0.387 |  |
| `kde_p99_equiv` | 8.831184 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 14.138703 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.651902 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 9.825560 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.168219 | 0.871 | 0.693 | 0.795 | 0.772 |  |
| `iqr` | 2.402318 | 0.860 | 0.580 | 0.742 | 0.692 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
