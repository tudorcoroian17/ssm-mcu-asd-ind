# Deployment setup: `w_all_pertensor_qabar_euclidean_fp32`

Combo 7 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.729507**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.9450, pAUC=0.8225.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.713516 | 0.955 | 0.811 | 0.886 | 0.877 |  |
| `percentile_same_machine_99.0` | 3.729507 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 11.009918 | 0.846 | 0.042 | 0.517 | 0.079 |  |
| `evt_p95_equiv` | 1.731399 | 0.955 | 0.795 | 0.879 | 0.868 |  |
| `parametric_p95_equiv` | 1.995839 | 0.953 | 0.538 | 0.756 | 0.688 |  |
| `kde_p95_equiv` | 1.898788 | 0.955 | 0.644 | 0.807 | 0.769 |  |
| `evt_p99_equiv` | 3.442220 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.380665 | 0.938 | 0.231 | 0.608 | 0.371 |  |
| `kde_p99_equiv` | 10.257885 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 36.159910 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 2.900871 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 11.365063 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.736072 | 0.959 | 0.795 | 0.881 | 0.870 |  |
| `iqr` | 1.901227 | 0.955 | 0.644 | 0.807 | 0.769 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
