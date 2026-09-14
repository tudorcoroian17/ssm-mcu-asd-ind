# Deployment setup: `w_all_pertensor_actboundaries_knn16_fp32`

Combo 10 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.423007**.

At this threshold on the case 3 test split: P=0.976, R=0.909, A=0.943, F1=0.941.

Ranking quality (threshold-independent): AUC=0.9835, pAUC=0.9684.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.369534 | 0.952 | 0.977 | 0.964 | 0.965 |  |
| `percentile_same_machine_99.0` | 1.423007 | 0.976 | 0.909 | 0.943 | 0.941 | **<- active** |
| `percentile_same_machine_99.5` | 1.453096 | 0.987 | 0.891 | 0.940 | 0.937 |  |
| `evt_p95_equiv` | 1.368782 | 0.945 | 0.977 | 0.960 | 0.961 |  |
| `parametric_p95_equiv` | 1.404176 | 0.973 | 0.936 | 0.955 | 0.954 |  |
| `kde_p95_equiv` | 1.391325 | 0.966 | 0.951 | 0.958 | 0.958 |  |
| `evt_p99_equiv` | 1.444656 | 0.987 | 0.891 | 0.940 | 0.937 |  |
| `parametric_p99_equiv` | 1.511754 | 0.995 | 0.758 | 0.877 | 0.861 |  |
| `kde_p99_equiv` | 1.465781 | 0.987 | 0.853 | 0.921 | 0.915 |  |
| `evt_p999_equiv` | 1.578320 | 1.000 | 0.626 | 0.813 | 0.770 |  |
| `parametric_p999_equiv` | 1.620735 | 1.000 | 0.509 | 0.755 | 0.675 |  |
| `kde_p999_equiv` | 1.797164 | 1.000 | 0.223 | 0.611 | 0.364 |  |
| `mad` | 1.983059 | 1.000 | 0.174 | 0.587 | 0.296 |  |
| `iqr` | 2.289088 | 1.000 | 0.117 | 0.558 | 0.209 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
