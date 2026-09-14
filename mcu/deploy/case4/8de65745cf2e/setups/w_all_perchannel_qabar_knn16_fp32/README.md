# Deployment setup: `w_all_perchannel_qabar_knn16_fp32`

Combo 4 of the case 4 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 4
- Model hash: `8de65745cf2e`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.125242**.

At this threshold on the case 4 test split: P=0.989, R=0.992, A=0.991, F1=0.991.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.090724 | 0.953 | 0.992 | 0.972 | 0.972 |  |
| `percentile_same_machine_99.0` | 1.125242 | 0.989 | 0.992 | 0.991 | 0.991 | **<- active** |
| `percentile_same_machine_99.5` | 1.134136 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `evt_p95_equiv` | 1.084725 | 0.946 | 0.992 | 0.968 | 0.969 |  |
| `parametric_p95_equiv` | 1.049702 | 0.913 | 0.992 | 0.949 | 0.951 |  |
| `kde_p95_equiv` | 1.092167 | 0.953 | 0.992 | 0.972 | 0.972 |  |
| `evt_p99_equiv` | 1.137784 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `parametric_p99_equiv` | 1.087241 | 0.946 | 0.992 | 0.968 | 0.969 |  |
| `kde_p99_equiv` | 1.130285 | 0.992 | 0.992 | 0.992 | 0.992 |  |
| `evt_p999_equiv` | 1.162921 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `parametric_p999_equiv` | 1.130916 | 0.992 | 0.992 | 0.992 | 0.992 |  |
| `kde_p999_equiv` | 1.163901 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `mad` | 1.078369 | 0.946 | 0.992 | 0.968 | 0.969 |  |
| `iqr` | 1.127167 | 0.992 | 0.992 | 0.992 | 0.992 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
