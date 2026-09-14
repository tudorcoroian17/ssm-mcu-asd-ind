# Deployment setup: `w_all_pertensor_actboundaries_qab_euclidean_fp32`

Combo 9 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.300704**.

At this threshold on the case 3 test split: P=0.982, R=0.604, A=0.796, F1=0.748.

Ranking quality (threshold-independent): AUC=0.9380, pAUC=0.8204.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.133885 | 0.943 | 0.687 | 0.823 | 0.795 |  |
| `percentile_same_machine_99.0` | 2.300704 | 0.982 | 0.604 | 0.796 | 0.748 | **<- active** |
| `percentile_same_machine_99.5` | 8.571606 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.119667 | 0.943 | 0.691 | 0.825 | 0.797 |  |
| `parametric_p95_equiv` | 2.425875 | 0.986 | 0.517 | 0.755 | 0.678 |  |
| `kde_p95_equiv` | 2.228691 | 0.982 | 0.623 | 0.806 | 0.762 |  |
| `evt_p99_equiv` | 2.691929 | 0.982 | 0.404 | 0.698 | 0.572 |  |
| `parametric_p99_equiv` | 2.872135 | 0.975 | 0.294 | 0.643 | 0.452 |  |
| `kde_p99_equiv` | 2.524103 | 0.984 | 0.472 | 0.732 | 0.638 |  |
| `evt_p999_equiv` | 8.105871 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.470622 | 0.963 | 0.196 | 0.594 | 0.326 |  |
| `kde_p999_equiv` | 10.508489 | 1.000 | 0.004 | 0.502 | 0.008 |  |
| `mad` | 2.841940 | 0.976 | 0.309 | 0.651 | 0.470 |  |
| `iqr` | 3.252958 | 0.969 | 0.234 | 0.613 | 0.377 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
