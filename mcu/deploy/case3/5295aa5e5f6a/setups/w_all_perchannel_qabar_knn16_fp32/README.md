# Deployment setup: `w_all_perchannel_qabar_knn16_fp32`

Combo 4 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.264485**.

At this threshold on the case 3 test split: P=0.981, R=0.989, A=0.985, F1=0.985.

Ranking quality (threshold-independent): AUC=0.9885, pAUC=0.9924.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.190963 | 0.956 | 0.989 | 0.972 | 0.972 |  |
| `percentile_same_machine_99.0` | 1.264485 | 0.981 | 0.989 | 0.985 | 0.985 | **<- active** |
| `percentile_same_machine_99.5` | 1.290589 | 0.985 | 0.989 | 0.987 | 0.987 |  |
| `evt_p95_equiv` | 1.189526 | 0.953 | 0.989 | 0.970 | 0.970 |  |
| `parametric_p95_equiv` | 1.350167 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `kde_p95_equiv` | 1.211567 | 0.970 | 0.989 | 0.979 | 0.979 |  |
| `evt_p99_equiv` | 1.279330 | 0.981 | 0.989 | 0.985 | 0.985 |  |
| `parametric_p99_equiv` | 1.492486 | 0.996 | 0.974 | 0.985 | 0.985 |  |
| `kde_p99_equiv` | 1.287202 | 0.981 | 0.989 | 0.985 | 0.985 |  |
| `evt_p999_equiv` | 1.555580 | 0.996 | 0.955 | 0.975 | 0.975 |  |
| `parametric_p999_equiv` | 1.656247 | 1.000 | 0.872 | 0.936 | 0.931 |  |
| `kde_p999_equiv` | 4.827077 | 1.000 | 0.004 | 0.502 | 0.008 |  |
| `mad` | 1.398385 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `iqr` | 1.512503 | 0.996 | 0.974 | 0.985 | 0.985 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
