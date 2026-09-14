# Deployment setup: `true_int8_h16_qabar_knn16_int8`

Combo 26 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16_sumsq` = **1.276433** (float form). For the int8 head the decision uses the int32 `ssm_threshold_knn16_sumsq` constant.

At this threshold on the case 2 test split: P=0.982, R=0.838, A=0.911, F1=0.904.

Ranking quality (threshold-independent): AUC=0.9434, pAUC=0.9220.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.029606408) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.243625 | 1764 | 0.931 | 0.868 | 0.902 | 0.898 |  |
| `percentile_same_machine_99.0` | 1.276433 | 1859 | 0.982 | 0.838 | 0.911 | 0.904 | **<- active** |
| `percentile_same_machine_99.5` | 1.297752 | 1921 | 0.986 | 0.815 | 0.902 | 0.893 |  |
| `evt_p95_equiv` | 1.242196 | 1760 | 0.931 | 0.868 | 0.902 | 0.898 |  |
| `parametric_p95_equiv` | 1.240100 | 1754 | 0.924 | 0.868 | 0.898 | 0.895 |  |
| `kde_p95_equiv` | 1.247510 | 1775 | 0.935 | 0.864 | 0.902 | 0.898 |  |
| `evt_p99_equiv` | 1.282719 | 1877 | 0.982 | 0.838 | 0.911 | 0.904 |  |
| `parametric_p99_equiv` | 1.273203 | 1849 | 0.978 | 0.842 | 0.911 | 0.905 |  |
| `kde_p99_equiv` | 1.287826 | 1892 | 0.982 | 0.830 | 0.908 | 0.900 |  |
| `evt_p999_equiv` | 1.321301 | 1992 | 1.000 | 0.781 | 0.891 | 0.877 |  |
| `parametric_p999_equiv` | 1.305227 | 1944 | 0.995 | 0.800 | 0.898 | 0.887 |  |
| `kde_p999_equiv` | 1.335840 | 2036 | 1.000 | 0.766 | 0.883 | 0.868 |  |
| `mad` | 1.403407 | 2247 | 1.000 | 0.698 | 0.849 | 0.822 |  |
| `iqr` | 1.498555 | 2562 | 1.000 | 0.521 | 0.760 | 0.685 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: `qabar`

This is the classic (`selective=False`) matrix. Its extra axis, which the selective matrix does not have, is how the discretized recurrence coefficients reach the device.

`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are discretized at export time and ship already quantized, so the firmware has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer instead of one, so more flash, but no per-frame `expf`, `softplus` or clamp.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights.

Both forms are calibrated against the same `ranges.json` entries (`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay comparable between them and against the selective matrix.

`ssm_backbone.c` is byte-for-byte identical across both forms: the difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / `SSM_B_BAR` macros.
