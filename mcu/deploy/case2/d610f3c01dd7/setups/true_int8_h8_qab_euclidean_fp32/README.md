# Deployment setup: `true_int8_h8_qab_euclidean_fp32`

Combo 11 of the case 2 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 2
- Model hash: `d610f3c01dd7`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.255394**.

At this threshold on the case 2 test split: P=1.000, R=0.781, A=0.891, F1=0.877.

Ranking quality (threshold-independent): AUC=0.9281, pAUC=0.9219.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.211787 | 0.974 | 0.842 | 0.909 | 0.903 |  |
| `percentile_same_machine_99.0` | 1.255394 | 1.000 | 0.781 | 0.891 | 0.877 | **<- active** |
| `percentile_same_machine_99.5` | 1.281853 | 1.000 | 0.717 | 0.858 | 0.835 |  |
| `evt_p95_equiv` | 1.211619 | 0.974 | 0.842 | 0.909 | 0.903 |  |
| `parametric_p95_equiv` | 1.222783 | 0.987 | 0.830 | 0.909 | 0.902 |  |
| `kde_p95_equiv` | 1.215441 | 0.982 | 0.842 | 0.913 | 0.907 |  |
| `evt_p99_equiv` | 1.255922 | 1.000 | 0.777 | 0.889 | 0.875 |  |
| `parametric_p99_equiv` | 1.274203 | 1.000 | 0.740 | 0.870 | 0.850 |  |
| `kde_p99_equiv` | 1.262535 | 1.000 | 0.774 | 0.887 | 0.872 |  |
| `evt_p999_equiv` | 1.306254 | 1.000 | 0.675 | 0.838 | 0.806 |  |
| `parametric_p999_equiv` | 1.332593 | 1.000 | 0.611 | 0.806 | 0.759 |  |
| `kde_p999_equiv` | 1.312890 | 1.000 | 0.657 | 0.828 | 0.793 |  |
| `mad` | 1.352714 | 1.000 | 0.574 | 0.787 | 0.729 |  |
| `iqr` | 1.434838 | 1.000 | 0.423 | 0.711 | 0.594 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
