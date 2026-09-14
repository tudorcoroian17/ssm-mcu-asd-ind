# Deployment setup: `full_fp32_euclidean_fp32`

Combo 19 of the case 3 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 3
- Model hash: `2437e6cfa1ab`
- Backbone: full fp32 (no quantization)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.921931**.

At this threshold on the case 3 test split: P=0.989, R=0.977, A=0.983, F1=0.983.

Ranking quality (threshold-independent): AUC=0.9927, pAUC=0.9617.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.634562 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `percentile_same_machine_99.0` | 1.921931 | 0.989 | 0.977 | 0.983 | 0.983 | **<- active** |
| `percentile_same_machine_99.5` | 3.829719 | 0.935 | 0.109 | 0.551 | 0.196 |  |
| `evt_p95_equiv` | 1.631661 | 0.946 | 1.000 | 0.972 | 0.972 |  |
| `parametric_p95_equiv` | 1.741170 | 0.971 | 1.000 | 0.985 | 0.985 |  |
| `kde_p95_equiv` | 1.669009 | 0.957 | 1.000 | 0.977 | 0.978 |  |
| `evt_p99_equiv` | 2.068965 | 0.992 | 0.925 | 0.958 | 0.957 |  |
| `parametric_p99_equiv` | 1.946751 | 0.988 | 0.970 | 0.979 | 0.979 |  |
| `kde_p99_equiv` | 1.968922 | 0.992 | 0.966 | 0.979 | 0.979 |  |
| `evt_p999_equiv` | 5.329137 | 1.000 | 0.034 | 0.517 | 0.066 |  |
| `parametric_p999_equiv` | 2.206171 | 0.990 | 0.755 | 0.874 | 0.857 |  |
| `kde_p999_equiv` | 4.787879 | 1.000 | 0.045 | 0.523 | 0.087 |  |
| `mad` | 1.989391 | 0.992 | 0.962 | 0.977 | 0.977 |  |
| `iqr` | 2.237464 | 0.989 | 0.694 | 0.843 | 0.816 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
