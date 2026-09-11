# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **4.020795**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8799, pAUC=0.7741.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.107249 | 0.944 | 0.708 | 0.833 | 0.810 |  |
| `percentile_same_machine_99.0` | 4.020795 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 7.744761 | 0.909 | 0.038 | 0.517 | 0.073 |  |
| `evt_p95_equiv` | 2.134426 | 0.942 | 0.674 | 0.816 | 0.786 |  |
| `parametric_p95_equiv` | 2.446906 | 0.942 | 0.489 | 0.729 | 0.643 |  |
| `kde_p95_equiv` | 2.230650 | 0.948 | 0.625 | 0.795 | 0.753 |  |
| `evt_p99_equiv` | 4.176396 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.771408 | 0.922 | 0.314 | 0.644 | 0.469 |  |
| `kde_p99_equiv` | 7.265274 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 52.215644 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.186561 | 0.931 | 0.205 | 0.595 | 0.335 |  |
| `kde_p999_equiv` | 7.951404 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.205518 | 0.949 | 0.640 | 0.803 | 0.765 |  |
| `iqr` | 2.365715 | 0.946 | 0.530 | 0.750 | 0.680 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
