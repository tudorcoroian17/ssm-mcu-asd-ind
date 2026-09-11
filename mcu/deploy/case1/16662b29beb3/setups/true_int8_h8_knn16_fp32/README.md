# Deployment setup: `true_int8_h8_knn16_fp32`

Combo 13 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
- Backbone: true int8 arithmetic, h storage = int8
- Head: knn_clustered_16 head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.319137**.

At this threshold on the case 1 test split: P=0.973, R=0.553, A=0.769, F1=0.705.

Ranking quality (threshold-independent): AUC=0.9314, pAUC=0.8428.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.081443 | 0.957 | 0.678 | 0.824 | 0.794 |  |
| `percentile_same_machine_99.0` | 2.319137 | 0.973 | 0.553 | 0.769 | 0.705 | **<- active** |
| `percentile_same_machine_99.5` | 2.450987 | 0.978 | 0.496 | 0.742 | 0.658 |  |
| `evt_p95_equiv` | 2.054475 | 0.944 | 0.701 | 0.830 | 0.804 |  |
| `parametric_p95_equiv` | 2.023533 | 0.936 | 0.720 | 0.835 | 0.814 |  |
| `kde_p95_equiv` | 2.085059 | 0.957 | 0.667 | 0.818 | 0.786 |  |
| `evt_p99_equiv` | 2.444308 | 0.971 | 0.500 | 0.742 | 0.660 |  |
| `parametric_p99_equiv` | 2.249127 | 0.968 | 0.576 | 0.778 | 0.722 |  |
| `kde_p99_equiv` | 2.354633 | 0.973 | 0.545 | 0.765 | 0.699 |  |
| `evt_p999_equiv` | 3.103261 | 1.000 | 0.246 | 0.623 | 0.395 |  |
| `parametric_p999_equiv` | 2.532021 | 1.000 | 0.451 | 0.725 | 0.621 |  |
| `kde_p999_equiv` | 2.640236 | 1.000 | 0.420 | 0.710 | 0.592 |  |
| `mad` | 2.345287 | 0.973 | 0.549 | 0.767 | 0.702 |  |
| `iqr` | 2.646101 | 1.000 | 0.413 | 0.706 | 0.584 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
