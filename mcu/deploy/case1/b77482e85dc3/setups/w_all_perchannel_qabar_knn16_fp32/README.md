# Deployment setup: `w_all_perchannel_qabar_knn16_fp32`

Combo 4 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.388117**.

At this threshold on the case 1 test split: P=1.000, R=0.867, A=0.934, F1=0.929.

Ranking quality (threshold-independent): AUC=0.9287, pAUC=0.9484.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.260988 | 0.941 | 0.909 | 0.926 | 0.925 |  |
| `percentile_same_machine_99.0` | 1.388117 | 1.000 | 0.867 | 0.934 | 0.929 | **<- active** |
| `percentile_same_machine_99.5` | 1.421887 | 1.000 | 0.833 | 0.917 | 0.909 |  |
| `evt_p95_equiv` | 1.268298 | 0.952 | 0.902 | 0.928 | 0.926 |  |
| `parametric_p95_equiv` | 1.282753 | 0.952 | 0.902 | 0.928 | 0.926 |  |
| `kde_p95_equiv` | 1.276394 | 0.952 | 0.902 | 0.928 | 0.926 |  |
| `evt_p99_equiv` | 1.367008 | 0.987 | 0.879 | 0.934 | 0.930 |  |
| `parametric_p99_equiv` | 1.340118 | 0.967 | 0.886 | 0.928 | 0.925 |  |
| `kde_p99_equiv` | 1.387424 | 1.000 | 0.867 | 0.934 | 0.929 |  |
| `evt_p999_equiv` | 1.498732 | 1.000 | 0.814 | 0.907 | 0.898 |  |
| `parametric_p999_equiv` | 1.396558 | 1.000 | 0.856 | 0.928 | 0.922 |  |
| `kde_p999_equiv` | 1.494158 | 1.000 | 0.818 | 0.909 | 0.900 |  |
| `mad` | 1.446896 | 1.000 | 0.826 | 0.913 | 0.905 |  |
| `iqr` | 1.565777 | 1.000 | 0.769 | 0.884 | 0.869 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
