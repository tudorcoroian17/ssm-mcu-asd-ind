# Deployment setup: `w_all_perchannel_qabar_euclidean_fp32`

Combo 3 of the case 3 deployment matrix.

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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.390568**.

At this threshold on the case 3 test split: P=0.986, R=0.540, A=0.766, F1=0.698.

Ranking quality (threshold-independent): AUC=0.9086, pAUC=0.7927.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.214116 | 0.937 | 0.619 | 0.789 | 0.745 |  |
| `percentile_same_machine_99.0` | 2.390568 | 0.986 | 0.540 | 0.766 | 0.698 | **<- active** |
| `percentile_same_machine_99.5` | 8.455829 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.197284 | 0.938 | 0.630 | 0.794 | 0.754 |  |
| `parametric_p95_equiv` | 2.505986 | 0.984 | 0.479 | 0.736 | 0.645 |  |
| `kde_p95_equiv` | 2.300822 | 0.975 | 0.592 | 0.789 | 0.737 |  |
| `evt_p99_equiv` | 2.775502 | 0.981 | 0.389 | 0.691 | 0.557 |  |
| `parametric_p99_equiv` | 2.959680 | 0.974 | 0.287 | 0.640 | 0.443 |  |
| `kde_p99_equiv` | 2.601892 | 0.984 | 0.453 | 0.723 | 0.620 |  |
| `evt_p999_equiv` | 7.820169 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.566543 | 0.962 | 0.192 | 0.592 | 0.321 |  |
| `kde_p999_equiv` | 10.404196 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.938894 | 0.975 | 0.291 | 0.642 | 0.448 |  |
| `iqr` | 3.366620 | 0.968 | 0.230 | 0.611 | 0.372 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
