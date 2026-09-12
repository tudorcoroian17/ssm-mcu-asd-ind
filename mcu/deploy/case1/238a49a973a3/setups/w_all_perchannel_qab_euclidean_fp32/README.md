# Deployment setup: `w_all_perchannel_qab_euclidean_fp32`

Combo 1 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **4.391297**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8829, pAUC=0.6755.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.738717 | 0.905 | 0.470 | 0.710 | 0.618 |  |
| `percentile_same_machine_99.0` | 4.391297 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 9.290661 | 0.769 | 0.038 | 0.513 | 0.072 |  |
| `evt_p95_equiv` | 2.817259 | 0.923 | 0.409 | 0.688 | 0.567 |  |
| `parametric_p95_equiv` | 2.564022 | 0.873 | 0.545 | 0.733 | 0.671 |  |
| `kde_p95_equiv` | 2.839105 | 0.927 | 0.383 | 0.676 | 0.542 |  |
| `evt_p99_equiv` | 4.784312 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 3.074943 | 0.896 | 0.261 | 0.616 | 0.405 |  |
| `kde_p99_equiv` | 8.603237 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 13.941147 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.769581 | 0.846 | 0.083 | 0.534 | 0.152 |  |
| `kde_p999_equiv` | 9.604851 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.162787 | 0.874 | 0.735 | 0.814 | 0.798 |  |
| `iqr` | 2.365514 | 0.862 | 0.640 | 0.769 | 0.735 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
