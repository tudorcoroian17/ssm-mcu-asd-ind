# Deployment setup: `true_int8_h8_euclidean_fp32`

Combo 11 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.988140**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8297, pAUC=0.6150.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.272118 | 0.842 | 0.303 | 0.623 | 0.446 |  |
| `percentile_same_machine_99.0` | 3.988140 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 8.571339 | 0.917 | 0.042 | 0.519 | 0.080 |  |
| `evt_p95_equiv` | 2.338038 | 0.869 | 0.277 | 0.617 | 0.420 |  |
| `parametric_p95_equiv` | 2.208774 | 0.845 | 0.330 | 0.634 | 0.474 |  |
| `kde_p95_equiv` | 2.355247 | 0.875 | 0.265 | 0.614 | 0.407 |  |
| `evt_p99_equiv` | 4.431282 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.652261 | 0.873 | 0.208 | 0.589 | 0.336 |  |
| `kde_p99_equiv` | 6.729687 | 0.833 | 0.057 | 0.523 | 0.106 |  |
| `evt_p999_equiv` | 16.335317 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.256029 | 0.844 | 0.102 | 0.542 | 0.182 |  |
| `kde_p999_equiv` | 10.386395 | 1.000 | 0.004 | 0.502 | 0.008 |  |
| `mad` | 1.745680 | 0.839 | 0.633 | 0.756 | 0.721 |  |
| `iqr` | 1.908248 | 0.822 | 0.489 | 0.691 | 0.613 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
