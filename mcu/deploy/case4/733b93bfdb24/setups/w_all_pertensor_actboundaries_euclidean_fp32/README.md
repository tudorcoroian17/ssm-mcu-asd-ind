# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 4 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 4
- Model hash: `733b93bfdb24`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.562079**.

At this threshold on the case 4 test split: P=0.996, R=1.000, A=0.998, F1=0.998.

Ranking quality (threshold-independent): AUC=1.0000, pAUC=1.0000.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.505244 | 0.940 | 1.000 | 0.968 | 0.969 |  |
| `percentile_same_machine_99.0` | 1.562079 | 0.996 | 1.000 | 0.998 | 0.998 | **<- active** |
| `percentile_same_machine_99.5` | 1.574687 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `evt_p95_equiv` | 1.509219 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `parametric_p95_equiv` | 1.497829 | 0.933 | 1.000 | 0.964 | 0.965 |  |
| `kde_p95_equiv` | 1.511959 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `evt_p99_equiv` | 1.564774 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `parametric_p99_equiv` | 1.567812 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `kde_p99_equiv` | 1.571130 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `evt_p999_equiv` | 1.598011 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `parametric_p999_equiv` | 1.650150 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `kde_p999_equiv` | 1.614904 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `mad` | 1.673696 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `iqr` | 1.801440 | 1.000 | 1.000 | 1.000 | 1.000 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
