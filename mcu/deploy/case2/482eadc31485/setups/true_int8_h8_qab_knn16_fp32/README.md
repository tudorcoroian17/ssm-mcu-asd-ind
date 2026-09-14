# Deployment setup: `true_int8_h8_qab_knn16_fp32`

Combo 13 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.342002**.

At this threshold on the case 2 test split: P=0.993, R=0.517, A=0.757, F1=0.680.

Ranking quality (threshold-independent): AUC=0.9215, pAUC=0.8379.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.234847 | 0.915 | 0.770 | 0.849 | 0.836 |  |
| `percentile_same_machine_99.0` | 1.342002 | 0.993 | 0.517 | 0.757 | 0.680 | **<- active** |
| `percentile_same_machine_99.5` | 1.358773 | 0.992 | 0.494 | 0.745 | 0.660 |  |
| `evt_p95_equiv` | 1.239192 | 0.914 | 0.762 | 0.845 | 0.831 |  |
| `parametric_p95_equiv` | 1.236928 | 0.915 | 0.770 | 0.849 | 0.836 |  |
| `kde_p95_equiv` | 1.243733 | 0.916 | 0.740 | 0.836 | 0.818 |  |
| `evt_p99_equiv` | 1.335013 | 0.993 | 0.532 | 0.764 | 0.693 |  |
| `parametric_p99_equiv` | 1.329731 | 0.986 | 0.547 | 0.770 | 0.704 |  |
| `kde_p99_equiv` | 1.347784 | 0.993 | 0.509 | 0.753 | 0.673 |  |
| `evt_p999_equiv` | 1.423522 | 0.991 | 0.396 | 0.696 | 0.566 |  |
| `parametric_p999_equiv` | 1.435933 | 0.990 | 0.374 | 0.685 | 0.542 |  |
| `kde_p999_equiv` | 1.439121 | 0.990 | 0.362 | 0.679 | 0.530 |  |
| `mad` | 1.444785 | 0.989 | 0.351 | 0.674 | 0.518 |  |
| `iqr` | 1.599481 | 1.000 | 0.162 | 0.581 | 0.279 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
