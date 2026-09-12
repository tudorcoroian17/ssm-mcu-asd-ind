# Deployment setup: `w_all_pertensor_actboundaries_qabar_knn16_fp32`

Combo 12 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.419884**.

At this threshold on the case 1 test split: P=0.996, R=0.856, A=0.926, F1=0.921.

Ranking quality (threshold-independent): AUC=0.9253, pAUC=0.9405.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.320038 | 0.940 | 0.898 | 0.920 | 0.919 |  |
| `percentile_same_machine_99.0` | 1.419884 | 0.996 | 0.856 | 0.926 | 0.921 | **<- active** |
| `percentile_same_machine_99.5` | 1.461264 | 1.000 | 0.830 | 0.915 | 0.907 |  |
| `evt_p95_equiv` | 1.319502 | 0.940 | 0.898 | 0.920 | 0.919 |  |
| `parametric_p95_equiv` | 1.339265 | 0.948 | 0.890 | 0.920 | 0.918 |  |
| `kde_p95_equiv` | 1.331862 | 0.944 | 0.894 | 0.920 | 0.918 |  |
| `evt_p99_equiv` | 1.406183 | 0.987 | 0.867 | 0.928 | 0.923 |  |
| `parametric_p99_equiv` | 1.393991 | 0.979 | 0.875 | 0.928 | 0.924 |  |
| `kde_p99_equiv` | 1.425172 | 0.996 | 0.856 | 0.926 | 0.921 |  |
| `evt_p999_equiv` | 1.536706 | 1.000 | 0.807 | 0.903 | 0.893 |  |
| `parametric_p999_equiv` | 1.447642 | 0.995 | 0.837 | 0.917 | 0.909 |  |
| `kde_p999_equiv` | 1.531579 | 1.000 | 0.814 | 0.907 | 0.898 |  |
| `mad` | 1.528214 | 1.000 | 0.822 | 0.911 | 0.902 |  |
| `iqr` | 1.645185 | 1.000 | 0.731 | 0.866 | 0.845 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
