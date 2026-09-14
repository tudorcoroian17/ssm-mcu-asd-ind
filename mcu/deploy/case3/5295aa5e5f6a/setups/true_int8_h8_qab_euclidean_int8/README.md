# Deployment setup: `true_int8_h8_qab_euclidean_int8`

Combo 12 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
- Backbone: true int8 arithmetic, h storage = int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.074964** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 3 test split: P=0.960, R=0.453, A=0.717, F1=0.615.

Ranking quality (threshold-independent): AUC=0.7563, pAUC=0.7203.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0347989075) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.950394 | 3141 | 0.928 | 0.487 | 0.725 | 0.639 |  |
| `percentile_same_machine_99.0` | 2.074964 | 3555 | 0.960 | 0.453 | 0.717 | 0.615 | **<- active** |
| `percentile_same_machine_99.5` | 5.882495 | 28575 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 1.929300 | 3074 | 0.910 | 0.498 | 0.725 | 0.644 |  |
| `parametric_p95_equiv` | 2.243599 | 4157 | 0.981 | 0.381 | 0.687 | 0.549 |  |
| `kde_p95_equiv` | 2.003221 | 3314 | 0.953 | 0.460 | 0.719 | 0.621 |  |
| `evt_p99_equiv` | 2.372250 | 4647 | 0.975 | 0.291 | 0.642 | 0.448 |  |
| `parametric_p99_equiv` | 2.663992 | 5860 | 0.968 | 0.226 | 0.609 | 0.367 |  |
| `kde_p99_equiv` | 2.214435 | 4049 | 0.981 | 0.400 | 0.696 | 0.568 |  |
| `evt_p999_equiv` | 5.845819 | 28220 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.229519 | 8613 | 0.931 | 0.102 | 0.547 | 0.184 |  |
| `kde_p999_equiv` | 7.298663 | 43990 | 1.000 | 0.004 | 0.502 | 0.008 |  |
| `mad` | 2.605606 | 5606 | 0.969 | 0.234 | 0.613 | 0.377 |  |
| `iqr` | 3.105653 | 7965 | 0.933 | 0.106 | 0.549 | 0.190 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
