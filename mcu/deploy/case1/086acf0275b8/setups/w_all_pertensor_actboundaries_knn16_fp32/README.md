# Deployment setup: `w_all_pertensor_actboundaries_knn16_fp32`

Combo 10 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.712072**.

At this threshold on the case 1 test split: P=0.981, R=0.598, A=0.794, F1=0.744.

Ranking quality (threshold-independent): AUC=0.8217, pAUC=0.8377.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.593290 | 0.931 | 0.712 | 0.830 | 0.807 |  |
| `percentile_same_machine_99.0` | 1.712072 | 0.981 | 0.598 | 0.794 | 0.744 | **<- active** |
| `percentile_same_machine_99.5` | 1.791671 | 0.986 | 0.534 | 0.763 | 0.693 |  |
| `evt_p95_equiv` | 1.595562 | 0.931 | 0.712 | 0.830 | 0.807 |  |
| `parametric_p95_equiv` | 1.629118 | 0.952 | 0.682 | 0.824 | 0.795 |  |
| `kde_p95_equiv` | 1.609851 | 0.953 | 0.689 | 0.828 | 0.800 |  |
| `evt_p99_equiv` | 1.711495 | 0.981 | 0.598 | 0.794 | 0.744 |  |
| `parametric_p99_equiv` | 1.694403 | 0.982 | 0.610 | 0.799 | 0.752 |  |
| `kde_p99_equiv` | 1.729374 | 0.981 | 0.572 | 0.780 | 0.722 |  |
| `evt_p999_equiv` | 1.948056 | 1.000 | 0.492 | 0.746 | 0.660 |  |
| `parametric_p999_equiv` | 1.758359 | 0.986 | 0.549 | 0.771 | 0.706 |  |
| `kde_p999_equiv` | 1.936416 | 1.000 | 0.492 | 0.746 | 0.660 |  |
| `mad` | 1.754784 | 0.986 | 0.549 | 0.771 | 0.706 |  |
| `iqr` | 1.855732 | 1.000 | 0.515 | 0.758 | 0.680 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
