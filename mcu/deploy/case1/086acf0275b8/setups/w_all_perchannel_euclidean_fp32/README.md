# Deployment setup: `w_all_perchannel_euclidean_fp32`

Combo 1 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.990185**.

At this threshold on the case 1 test split: P=0.852, R=0.087, A=0.536, F1=0.158.

Ranking quality (threshold-independent): AUC=0.8693, pAUC=0.7729.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.142401 | 0.938 | 0.686 | 0.820 | 0.792 |  |
| `percentile_same_machine_99.0` | 3.990185 | 0.852 | 0.087 | 0.536 | 0.158 | **<- active** |
| `percentile_same_machine_99.5` | 7.770971 | 1.000 | 0.038 | 0.519 | 0.073 |  |
| `evt_p95_equiv` | 2.171355 | 0.946 | 0.663 | 0.812 | 0.780 |  |
| `parametric_p95_equiv` | 2.477627 | 0.944 | 0.515 | 0.742 | 0.667 |  |
| `kde_p95_equiv` | 2.263064 | 0.949 | 0.636 | 0.801 | 0.762 |  |
| `evt_p99_equiv` | 4.053547 | 0.846 | 0.083 | 0.534 | 0.152 |  |
| `parametric_p99_equiv` | 2.799583 | 0.941 | 0.364 | 0.670 | 0.525 |  |
| `kde_p99_equiv` | 7.302367 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 40.221388 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.210446 | 0.942 | 0.246 | 0.616 | 0.390 |  |
| `kde_p999_equiv` | 7.975124 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.256226 | 0.949 | 0.636 | 0.801 | 0.762 |  |
| `iqr` | 2.409942 | 0.950 | 0.576 | 0.773 | 0.717 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
