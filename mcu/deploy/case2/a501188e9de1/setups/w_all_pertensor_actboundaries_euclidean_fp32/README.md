# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.650300**.

At this threshold on the case 2 test split: P=0.996, R=0.917, A=0.957, F1=0.955.

Ranking quality (threshold-independent): AUC=0.9867, pAUC=0.9777.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.599021 | 0.962 | 0.955 | 0.958 | 0.958 |  |
| `percentile_same_machine_99.0` | 1.650300 | 0.996 | 0.917 | 0.957 | 0.955 | **<- active** |
| `percentile_same_machine_99.5` | 1.673040 | 0.996 | 0.902 | 0.949 | 0.947 |  |
| `evt_p95_equiv` | 1.600576 | 0.962 | 0.955 | 0.958 | 0.958 |  |
| `parametric_p95_equiv` | 1.603026 | 0.962 | 0.947 | 0.955 | 0.954 |  |
| `kde_p95_equiv` | 1.609469 | 0.977 | 0.947 | 0.962 | 0.962 |  |
| `evt_p99_equiv` | 1.644132 | 0.996 | 0.921 | 0.958 | 0.957 |  |
| `parametric_p99_equiv` | 1.646158 | 0.996 | 0.921 | 0.958 | 0.957 |  |
| `kde_p99_equiv` | 1.659916 | 0.996 | 0.909 | 0.953 | 0.951 |  |
| `evt_p999_equiv` | 1.704189 | 1.000 | 0.875 | 0.938 | 0.934 |  |
| `parametric_p999_equiv` | 1.687893 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `kde_p999_equiv` | 1.709390 | 1.000 | 0.875 | 0.938 | 0.934 |  |
| `mad` | 1.844560 | 1.000 | 0.709 | 0.855 | 0.830 |  |
| `iqr` | 1.982313 | 1.000 | 0.638 | 0.819 | 0.779 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
