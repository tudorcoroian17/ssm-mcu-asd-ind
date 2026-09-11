# Deployment setup: `true_int8_h8_euclidean_fp32`

Combo 11 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
- Backbone: true int8 arithmetic, h storage = int8
- Head: euclidean head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.563694**.

At this threshold on the case 1 test split: P=0.940, R=0.239, A=0.612, F1=0.381.

Ranking quality (threshold-independent): AUC=0.8968, pAUC=0.7139.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.696993 | 0.919 | 0.519 | 0.737 | 0.663 |  |
| `percentile_same_machine_99.0` | 3.563694 | 0.940 | 0.239 | 0.612 | 0.381 | **<- active** |
| `percentile_same_machine_99.5` | 5.299557 | 0.800 | 0.030 | 0.511 | 0.058 |  |
| `evt_p95_equiv` | 2.693484 | 0.920 | 0.523 | 0.739 | 0.667 |  |
| `parametric_p95_equiv` | 2.634985 | 0.919 | 0.557 | 0.754 | 0.693 |  |
| `kde_p95_equiv` | 2.719863 | 0.918 | 0.508 | 0.731 | 0.654 |  |
| `evt_p99_equiv` | 3.706458 | 0.931 | 0.205 | 0.595 | 0.335 |  |
| `parametric_p99_equiv` | 3.027396 | 0.942 | 0.371 | 0.674 | 0.533 |  |
| `kde_p99_equiv` | 4.687831 | 0.826 | 0.072 | 0.528 | 0.132 |  |
| `evt_p999_equiv` | 7.599892 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.537117 | 0.942 | 0.246 | 0.616 | 0.390 |  |
| `kde_p999_equiv` | 6.057237 | 1.000 | 0.019 | 0.509 | 0.037 |  |
| `mad` | 2.755372 | 0.923 | 0.500 | 0.729 | 0.649 |  |
| `iqr` | 3.163465 | 0.956 | 0.330 | 0.657 | 0.490 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
