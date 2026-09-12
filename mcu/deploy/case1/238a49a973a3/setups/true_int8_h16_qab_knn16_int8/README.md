# Deployment setup: `true_int8_h16_qab_knn16_int8`

Combo 24 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: true int8 arithmetic, h storage = int16
- Head: knn_clustered_16 head (int32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **2.341791** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 1 test split: P=1.000, R=0.477, A=0.739, F1=0.646.

Ranking quality (threshold-independent): AUC=0.8431, pAUC=0.8255.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0329585263) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.878919 | 3250 | 0.947 | 0.682 | 0.822 | 0.793 |  |
| `percentile_same_machine_99.0` | 2.341791 | 5048 | 1.000 | 0.477 | 0.739 | 0.646 | **<- active** |
| `percentile_same_machine_99.5` | 2.394799 | 5280 | 1.000 | 0.439 | 0.720 | 0.611 |  |
| `evt_p95_equiv` | 1.906331 | 3345 | 0.962 | 0.670 | 0.822 | 0.790 |  |
| `parametric_p95_equiv` | 1.939894 | 3464 | 0.961 | 0.648 | 0.811 | 0.774 |  |
| `kde_p95_equiv` | 1.893198 | 3300 | 0.952 | 0.674 | 0.820 | 0.789 |  |
| `evt_p99_equiv` | 2.257927 | 4693 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `parametric_p99_equiv` | 2.080587 | 3985 | 0.973 | 0.542 | 0.763 | 0.696 |  |
| `kde_p99_equiv` | 2.346850 | 5070 | 1.000 | 0.477 | 0.739 | 0.646 |  |
| `evt_p999_equiv` | 2.920523 | 7852 | 1.000 | 0.152 | 0.576 | 0.263 |  |
| `parametric_p999_equiv` | 2.222589 | 4548 | 0.977 | 0.492 | 0.741 | 0.655 |  |
| `kde_p999_equiv` | 2.499608 | 5752 | 1.000 | 0.371 | 0.686 | 0.541 |  |
| `mad` | 1.999390 | 3680 | 0.970 | 0.614 | 0.797 | 0.752 |  |
| `iqr` | 2.176053 | 4359 | 0.978 | 0.500 | 0.744 | 0.662 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
