# Deployment setup: `true_int8_h8_qab_knn16_int8`

Combo 16 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **2.413132** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 1 test split: P=0.978, R=0.511, A=0.750, F1=0.672.

Ranking quality (threshold-independent): AUC=0.8575, pAUC=0.8261.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0329585263) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.092099 | 4029 | 0.957 | 0.678 | 0.824 | 0.794 |  |
| `percentile_same_machine_99.0` | 2.413132 | 5361 | 0.978 | 0.511 | 0.750 | 0.672 | **<- active** |
| `percentile_same_machine_99.5` | 2.478969 | 5657 | 0.992 | 0.473 | 0.735 | 0.641 |  |
| `evt_p95_equiv` | 2.129677 | 4175 | 0.965 | 0.629 | 0.803 | 0.761 |  |
| `parametric_p95_equiv` | 2.031163 | 3798 | 0.917 | 0.712 | 0.824 | 0.802 |  |
| `kde_p95_equiv` | 2.093514 | 4035 | 0.957 | 0.678 | 0.824 | 0.794 |  |
| `evt_p99_equiv` | 2.389170 | 5255 | 0.971 | 0.511 | 0.748 | 0.670 |  |
| `parametric_p99_equiv` | 2.210304 | 4497 | 0.975 | 0.602 | 0.794 | 0.745 |  |
| `kde_p99_equiv` | 2.430982 | 5440 | 0.985 | 0.508 | 0.750 | 0.670 |  |
| `evt_p999_equiv` | 2.518987 | 5841 | 0.992 | 0.451 | 0.723 | 0.620 |  |
| `parametric_p999_equiv` | 2.429948 | 5436 | 0.985 | 0.508 | 0.750 | 0.670 |  |
| `kde_p999_equiv` | 2.553710 | 6004 | 1.000 | 0.432 | 0.716 | 0.603 |  |
| `mad` | 2.137788 | 4207 | 0.965 | 0.629 | 0.803 | 0.761 |  |
| `iqr` | 2.301374 | 4876 | 0.974 | 0.561 | 0.773 | 0.712 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
