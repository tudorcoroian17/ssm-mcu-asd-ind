# Deployment setup: `w_all_pertensor_euclidean_fp32`

Combo 3 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
- Backbone: weight-only int8 (all tensors, per-tensor)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.871795**.

At this threshold on the case 1 test split: P=0.852, R=0.087, A=0.536, F1=0.158.

Ranking quality (threshold-independent): AUC=0.9461, pAUC=0.7429.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.244730 | 0.931 | 0.614 | 0.784 | 0.740 |  |
| `percentile_same_machine_99.0` | 3.871795 | 0.852 | 0.087 | 0.536 | 0.158 | **<- active** |
| `percentile_same_machine_99.5` | 8.553407 | 0.818 | 0.034 | 0.513 | 0.065 |  |
| `evt_p95_equiv` | 2.319654 | 0.951 | 0.583 | 0.777 | 0.723 |  |
| `parametric_p95_equiv` | 2.228329 | 0.928 | 0.636 | 0.794 | 0.755 |  |
| `kde_p95_equiv` | 2.365534 | 0.950 | 0.572 | 0.771 | 0.714 |  |
| `evt_p99_equiv` | 4.051706 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.649999 | 0.941 | 0.424 | 0.699 | 0.585 |  |
| `kde_p99_equiv` | 7.965969 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 19.504390 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.218189 | 0.938 | 0.227 | 0.606 | 0.366 |  |
| `kde_p999_equiv` | 8.800889 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.656229 | 0.875 | 0.985 | 0.922 | 0.927 |  |
| `iqr` | 1.757519 | 0.877 | 0.943 | 0.905 | 0.909 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
