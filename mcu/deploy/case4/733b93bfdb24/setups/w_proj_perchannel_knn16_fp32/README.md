# Deployment setup: `w_proj_perchannel_knn16_fp32`

Combo 6 of the case 4 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 4
- Model hash: `733b93bfdb24`
- Backbone: weight-only int8 (projections tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.443733**.

At this threshold on the case 4 test split: P=0.981, R=0.992, A=0.987, F1=0.987.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.414794 | 0.960 | 0.992 | 0.975 | 0.976 |  |
| `percentile_same_machine_99.0` | 1.443733 | 0.981 | 0.992 | 0.987 | 0.987 | **<- active** |
| `percentile_same_machine_99.5` | 1.454856 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `evt_p95_equiv` | 1.416107 | 0.960 | 0.992 | 0.975 | 0.976 |  |
| `parametric_p95_equiv` | 1.427337 | 0.978 | 0.992 | 0.985 | 0.985 |  |
| `kde_p95_equiv` | 1.421341 | 0.967 | 0.992 | 0.979 | 0.980 |  |
| `evt_p99_equiv` | 1.445900 | 0.981 | 0.992 | 0.987 | 0.987 |  |
| `parametric_p99_equiv` | 1.488520 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `kde_p99_equiv` | 1.457073 | 0.989 | 0.992 | 0.991 | 0.991 |  |
| `evt_p999_equiv` | 1.467323 | 0.992 | 0.992 | 0.992 | 0.992 |  |
| `parametric_p999_equiv` | 1.558010 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `kde_p999_equiv` | 1.488817 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `mad` | 1.650623 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `iqr` | 1.782578 | 1.000 | 0.992 | 0.996 | 0.996 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
