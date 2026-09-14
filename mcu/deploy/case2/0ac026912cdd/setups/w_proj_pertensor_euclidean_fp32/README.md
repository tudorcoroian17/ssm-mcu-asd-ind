# Deployment setup: `w_proj_pertensor_euclidean_fp32`

Combo 7 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
- Backbone: weight-only int8 (projections tensors, per-tensor)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.551698**.

At this threshold on the case 2 test split: P=0.988, R=0.951, A=0.970, F1=0.969.

Ranking quality (threshold-independent): AUC=0.9775, pAUC=0.9806.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.523172 | 0.962 | 0.966 | 0.964 | 0.964 |  |
| `percentile_same_machine_99.0` | 1.551698 | 0.988 | 0.951 | 0.970 | 0.969 | **<- active** |
| `percentile_same_machine_99.5` | 1.567798 | 1.000 | 0.947 | 0.974 | 0.973 |  |
| `evt_p95_equiv` | 1.522784 | 0.963 | 0.970 | 0.966 | 0.966 |  |
| `parametric_p95_equiv` | 1.523661 | 0.962 | 0.966 | 0.964 | 0.964 |  |
| `kde_p95_equiv` | 1.527138 | 0.969 | 0.958 | 0.964 | 0.964 |  |
| `evt_p99_equiv` | 1.554614 | 0.988 | 0.951 | 0.970 | 0.969 |  |
| `parametric_p99_equiv` | 1.550341 | 0.988 | 0.951 | 0.970 | 0.969 |  |
| `kde_p99_equiv` | 1.561315 | 0.992 | 0.947 | 0.970 | 0.969 |  |
| `evt_p999_equiv` | 1.584720 | 1.000 | 0.943 | 0.972 | 0.971 |  |
| `parametric_p999_equiv` | 1.575928 | 1.000 | 0.947 | 0.974 | 0.973 |  |
| `kde_p999_equiv` | 1.596250 | 1.000 | 0.936 | 0.968 | 0.967 |  |
| `mad` | 1.689020 | 1.000 | 0.879 | 0.940 | 0.936 |  |
| `iqr` | 1.775646 | 1.000 | 0.815 | 0.908 | 0.898 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
