# Deployment setup: `w_proj_pertensor_knn16_fp32`

Combo 8 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
- Backbone: weight-only int8 (projections tensors, per-tensor)
- Head: knn_clustered_16 head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.318238**.

At this threshold on the case 1 test split: P=0.984, R=0.466, A=0.729, F1=0.632.

Ranking quality (threshold-independent): AUC=0.9117, pAUC=0.8605.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.736199 | 0.949 | 0.769 | 0.864 | 0.849 |  |
| `percentile_same_machine_99.0` | 2.318238 | 0.984 | 0.466 | 0.729 | 0.632 | **<- active** |
| `percentile_same_machine_99.5` | 2.389047 | 0.982 | 0.420 | 0.706 | 0.589 |  |
| `evt_p95_equiv` | 1.770997 | 0.966 | 0.742 | 0.858 | 0.839 |  |
| `parametric_p95_equiv` | 1.644179 | 0.902 | 0.833 | 0.871 | 0.866 |  |
| `kde_p95_equiv` | 1.750326 | 0.957 | 0.758 | 0.862 | 0.846 |  |
| `evt_p99_equiv` | 2.203966 | 0.977 | 0.492 | 0.741 | 0.655 |  |
| `parametric_p99_equiv` | 1.867482 | 0.979 | 0.693 | 0.839 | 0.812 |  |
| `kde_p99_equiv` | 2.327445 | 0.984 | 0.462 | 0.727 | 0.629 |  |
| `evt_p999_equiv` | 3.136984 | 1.000 | 0.080 | 0.540 | 0.147 |  |
| `parametric_p999_equiv` | 2.154023 | 0.978 | 0.504 | 0.746 | 0.665 |  |
| `kde_p999_equiv` | 2.492985 | 1.000 | 0.326 | 0.663 | 0.491 |  |
| `mad` | 1.486169 | 0.891 | 0.924 | 0.905 | 0.907 |  |
| `iqr` | 1.591634 | 0.887 | 0.860 | 0.875 | 0.873 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
