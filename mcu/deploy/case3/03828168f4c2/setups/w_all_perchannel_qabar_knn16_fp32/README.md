# Deployment setup: `w_all_perchannel_qabar_knn16_fp32`

Combo 4 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.116766**.

At this threshold on the case 3 test split: P=0.985, R=0.989, A=0.987, F1=0.987.

Ranking quality (threshold-independent): AUC=0.9887, pAUC=0.9940.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.082179 | 0.967 | 0.989 | 0.977 | 0.978 |  |
| `percentile_same_machine_99.0` | 1.116766 | 0.985 | 0.989 | 0.987 | 0.987 | **<- active** |
| `percentile_same_machine_99.5` | 1.135124 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `evt_p95_equiv` | 1.078003 | 0.960 | 0.989 | 0.974 | 0.974 |  |
| `parametric_p95_equiv` | 1.191485 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `kde_p95_equiv` | 1.099028 | 0.981 | 0.989 | 0.985 | 0.985 |  |
| `evt_p99_equiv` | 1.144238 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `parametric_p99_equiv` | 1.340349 | 1.000 | 0.970 | 0.985 | 0.985 |  |
| `kde_p99_equiv` | 1.157843 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `evt_p999_equiv` | 1.300604 | 1.000 | 0.977 | 0.989 | 0.989 |  |
| `parametric_p999_equiv` | 1.512114 | 1.000 | 0.872 | 0.936 | 0.931 |  |
| `kde_p999_equiv` | 2.737034 | 1.000 | 0.102 | 0.551 | 0.185 |  |
| `mad` | 1.437725 | 1.000 | 0.928 | 0.964 | 0.963 |  |
| `iqr` | 1.685446 | 1.000 | 0.355 | 0.677 | 0.524 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
