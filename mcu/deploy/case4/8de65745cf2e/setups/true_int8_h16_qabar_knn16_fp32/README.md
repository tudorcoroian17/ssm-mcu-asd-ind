# Deployment setup: `true_int8_h16_qabar_knn16_fp32`

Combo 25 of the case 4 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 4
- Model hash: `8de65745cf2e`
- Backbone: true int8 arithmetic, h storage = int16
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.183902**.

At this threshold on the case 4 test split: P=0.992, R=0.992, A=0.992, F1=0.992.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.157277 | 0.953 | 0.992 | 0.972 | 0.972 |  |
| `percentile_same_machine_99.0` | 1.183902 | 0.992 | 0.992 | 0.992 | 0.992 | **<- active** |
| `percentile_same_machine_99.5` | 1.194406 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `evt_p95_equiv` | 1.157298 | 0.953 | 0.992 | 0.972 | 0.972 |  |
| `parametric_p95_equiv` | 1.157554 | 0.953 | 0.992 | 0.972 | 0.972 |  |
| `kde_p95_equiv` | 1.160037 | 0.963 | 0.992 | 0.977 | 0.978 |  |
| `evt_p99_equiv` | 1.185668 | 0.992 | 0.992 | 0.992 | 0.992 |  |
| `parametric_p99_equiv` | 1.175857 | 0.978 | 0.992 | 0.985 | 0.985 |  |
| `kde_p99_equiv` | 1.189075 | 0.992 | 0.992 | 0.992 | 0.992 |  |
| `evt_p999_equiv` | 1.218748 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `parametric_p999_equiv` | 1.193380 | 0.996 | 0.992 | 0.994 | 0.994 |  |
| `kde_p999_equiv` | 1.223840 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `mad` | 1.238504 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `iqr` | 1.284090 | 1.000 | 0.992 | 0.996 | 0.996 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
