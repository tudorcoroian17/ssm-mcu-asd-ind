# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.572522**.

At this threshold on the case 2 test split: P=0.992, R=0.943, A=0.968, F1=0.967.

Ranking quality (threshold-independent): AUC=0.9787, pAUC=0.9784.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.527986 | 0.938 | 0.962 | 0.949 | 0.950 |  |
| `percentile_same_machine_99.0` | 1.572522 | 0.992 | 0.943 | 0.968 | 0.967 | **<- active** |
| `percentile_same_machine_99.5` | 1.581539 | 0.992 | 0.940 | 0.966 | 0.965 |  |
| `evt_p95_equiv` | 1.530735 | 0.948 | 0.962 | 0.955 | 0.955 |  |
| `parametric_p95_equiv` | 1.542146 | 0.973 | 0.962 | 0.968 | 0.968 |  |
| `kde_p95_equiv` | 1.534624 | 0.959 | 0.962 | 0.960 | 0.960 |  |
| `evt_p99_equiv` | 1.566243 | 0.988 | 0.947 | 0.968 | 0.967 |  |
| `parametric_p99_equiv` | 1.589784 | 0.992 | 0.940 | 0.966 | 0.965 |  |
| `kde_p99_equiv` | 1.574406 | 0.992 | 0.943 | 0.968 | 0.967 |  |
| `evt_p999_equiv` | 1.605675 | 1.000 | 0.925 | 0.962 | 0.961 |  |
| `parametric_p999_equiv` | 1.643712 | 1.000 | 0.902 | 0.951 | 0.948 |  |
| `kde_p999_equiv` | 1.611712 | 1.000 | 0.921 | 0.960 | 0.959 |  |
| `mad` | 1.675817 | 1.000 | 0.891 | 0.945 | 0.942 |  |
| `iqr` | 1.758431 | 1.000 | 0.834 | 0.917 | 0.909 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
