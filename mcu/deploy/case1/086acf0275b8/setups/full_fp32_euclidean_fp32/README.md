# Deployment setup: `full_fp32_euclidean_fp32`

Combo 19 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.970509**.

At this threshold on the case 1 test split: P=0.852, R=0.087, A=0.536, F1=0.158.

Ranking quality (threshold-independent): AUC=0.8676, pAUC=0.7727.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.148133 | 0.942 | 0.678 | 0.818 | 0.789 |  |
| `percentile_same_machine_99.0` | 3.970509 | 0.852 | 0.087 | 0.536 | 0.158 | **<- active** |
| `percentile_same_machine_99.5` | 7.753666 | 1.000 | 0.038 | 0.519 | 0.073 |  |
| `evt_p95_equiv` | 2.171725 | 0.946 | 0.663 | 0.812 | 0.780 |  |
| `parametric_p95_equiv` | 2.476340 | 0.944 | 0.515 | 0.742 | 0.667 |  |
| `kde_p95_equiv` | 2.263877 | 0.949 | 0.636 | 0.801 | 0.762 |  |
| `evt_p99_equiv` | 4.017383 | 0.846 | 0.083 | 0.534 | 0.152 |  |
| `parametric_p99_equiv` | 2.796596 | 0.941 | 0.364 | 0.670 | 0.525 |  |
| `kde_p99_equiv` | 7.286382 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 38.715026 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.205050 | 0.942 | 0.246 | 0.616 | 0.390 |  |
| `kde_p999_equiv` | 7.956835 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.261651 | 0.949 | 0.636 | 0.801 | 0.762 |  |
| `iqr` | 2.408420 | 0.950 | 0.576 | 0.773 | 0.717 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
