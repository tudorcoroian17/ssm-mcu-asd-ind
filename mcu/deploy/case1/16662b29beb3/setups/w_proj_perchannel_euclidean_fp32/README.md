# Deployment setup: `w_proj_perchannel_euclidean_fp32`

Combo 5 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
- Backbone: weight-only int8 (projections tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.889243**.

At this threshold on the case 1 test split: P=0.840, R=0.080, A=0.532, F1=0.145.

Ranking quality (threshold-independent): AUC=0.9441, pAUC=0.7345.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.233697 | 0.936 | 0.606 | 0.782 | 0.736 |  |
| `percentile_same_machine_99.0` | 3.889243 | 0.840 | 0.080 | 0.532 | 0.145 | **<- active** |
| `percentile_same_machine_99.5` | 8.637962 | 0.900 | 0.034 | 0.515 | 0.066 |  |
| `evt_p95_equiv` | 2.308771 | 0.949 | 0.568 | 0.769 | 0.711 |  |
| `parametric_p95_equiv` | 2.204670 | 0.916 | 0.621 | 0.782 | 0.740 |  |
| `kde_p95_equiv` | 2.354644 | 0.948 | 0.557 | 0.763 | 0.702 |  |
| `evt_p99_equiv` | 4.078913 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.632003 | 0.938 | 0.402 | 0.688 | 0.562 |  |
| `kde_p99_equiv` | 8.043212 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 19.997555 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.210193 | 0.922 | 0.178 | 0.581 | 0.298 |  |
| `kde_p999_equiv` | 8.890301 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.621782 | 0.875 | 0.985 | 0.922 | 0.927 |  |
| `iqr` | 1.727543 | 0.879 | 0.939 | 0.905 | 0.908 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
