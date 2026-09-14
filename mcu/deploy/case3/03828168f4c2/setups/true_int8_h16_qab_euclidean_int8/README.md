# Deployment setup: `true_int8_h16_qab_euclidean_int8`

Combo 20 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
- Backbone: true int8 arithmetic, h storage = int16
- Head: euclidean head (int32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.736556** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.958, R=0.257, A=0.623, F1=0.405.

Ranking quality (threshold-independent): AUC=0.7393, pAUC=0.6671.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.029621847) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.399019 | 6559 | 0.918 | 0.423 | 0.692 | 0.579 |  |
| `percentile_same_machine_99.0` | 2.736556 | 8535 | 0.958 | 0.257 | 0.623 | 0.405 | **<- active** |
| `percentile_same_machine_99.5` | 10.349579 | 122073 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.383603 | 6475 | 0.903 | 0.423 | 0.689 | 0.576 |  |
| `parametric_p95_equiv` | 2.755908 | 8656 | 0.957 | 0.253 | 0.621 | 0.400 |  |
| `kde_p95_equiv` | 2.498172 | 7112 | 0.931 | 0.358 | 0.666 | 0.518 |  |
| `evt_p99_equiv` | 3.284661 | 12296 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `parametric_p99_equiv` | 3.389914 | 13096 | 0.949 | 0.140 | 0.566 | 0.243 |  |
| `kde_p99_equiv` | 2.957726 | 9970 | 0.964 | 0.204 | 0.598 | 0.336 |  |
| `evt_p999_equiv` | 10.144592 | 117286 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 4.275473 | 20833 | 0.926 | 0.094 | 0.543 | 0.171 |  |
| `kde_p999_equiv` | 11.919291 | 161911 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 3.308562 | 12475 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `iqr` | 3.953868 | 17816 | 0.931 | 0.102 | 0.547 | 0.184 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
