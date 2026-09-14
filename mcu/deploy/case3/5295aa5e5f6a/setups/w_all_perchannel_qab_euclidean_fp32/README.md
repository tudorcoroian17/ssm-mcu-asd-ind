# Deployment setup: `w_all_perchannel_qab_euclidean_fp32`

Combo 1 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.382883**.

At this threshold on the case 3 test split: P=0.986, R=0.540, A=0.766, F1=0.698.

Ranking quality (threshold-independent): AUC=0.9092, pAUC=0.7936.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.208343 | 0.938 | 0.623 | 0.791 | 0.748 |  |
| `percentile_same_machine_99.0` | 2.382883 | 0.986 | 0.540 | 0.766 | 0.698 | **<- active** |
| `percentile_same_machine_99.5` | 8.472427 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.191855 | 0.939 | 0.634 | 0.796 | 0.757 |  |
| `parametric_p95_equiv` | 2.500738 | 0.984 | 0.479 | 0.736 | 0.645 |  |
| `kde_p95_equiv` | 2.296081 | 0.975 | 0.592 | 0.789 | 0.737 |  |
| `evt_p99_equiv` | 2.767626 | 0.981 | 0.385 | 0.689 | 0.553 |  |
| `parametric_p99_equiv` | 2.951093 | 0.974 | 0.283 | 0.638 | 0.439 |  |
| `kde_p99_equiv` | 2.595971 | 0.984 | 0.453 | 0.723 | 0.620 |  |
| `evt_p999_equiv` | 7.797786 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.552972 | 0.962 | 0.189 | 0.591 | 0.315 |  |
| `kde_p999_equiv` | 10.422626 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.932223 | 0.975 | 0.291 | 0.642 | 0.448 |  |
| `iqr` | 3.350615 | 0.968 | 0.230 | 0.611 | 0.372 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qab`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and computes `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, in the same order as the selective backbone. Cheaper in flash, more work per frame.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
