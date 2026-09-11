# Deployment setup: `w_all_perchannel_euclidean_fp32`

Combo 1 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.900297**.

At this threshold on the case 1 test split: P=0.840, R=0.080, A=0.532, F1=0.145.

Ranking quality (threshold-independent): AUC=0.9441, pAUC=0.7340.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.243671 | 0.936 | 0.606 | 0.782 | 0.736 |  |
| `percentile_same_machine_99.0` | 3.900297 | 0.840 | 0.080 | 0.532 | 0.145 | **<- active** |
| `percentile_same_machine_99.5` | 8.635736 | 0.900 | 0.034 | 0.515 | 0.066 |  |
| `evt_p95_equiv` | 2.320091 | 0.949 | 0.568 | 0.769 | 0.711 |  |
| `parametric_p95_equiv` | 2.212578 | 0.912 | 0.625 | 0.782 | 0.742 |  |
| `kde_p95_equiv` | 2.366130 | 0.948 | 0.557 | 0.763 | 0.702 |  |
| `evt_p99_equiv` | 4.089236 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.641329 | 0.938 | 0.402 | 0.688 | 0.562 |  |
| `kde_p99_equiv` | 8.040213 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 20.010021 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.221411 | 0.923 | 0.182 | 0.583 | 0.304 |  |
| `kde_p999_equiv` | 8.888040 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.626763 | 0.875 | 0.985 | 0.922 | 0.927 |  |
| `iqr` | 1.729328 | 0.880 | 0.943 | 0.907 | 0.910 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
