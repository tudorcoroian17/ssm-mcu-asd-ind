# Deployment setup: `true_int8_h8_euclidean_fp32`

Combo 11 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.024781**.

At this threshold on the case 3 test split: P=0.959, R=0.268, A=0.628, F1=0.419.

Ranking quality (threshold-independent): AUC=0.8278, pAUC=0.7082.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.504086 | 0.934 | 0.483 | 0.725 | 0.637 |  |
| `percentile_same_machine_99.0` | 3.024781 | 0.959 | 0.268 | 0.628 | 0.419 | **<- active** |
| `percentile_same_machine_99.5` | 8.680052 | 0.000 | 0.000 | 0.498 | 0.000 |  |
| `evt_p95_equiv` | 2.494168 | 0.934 | 0.483 | 0.725 | 0.637 |  |
| `parametric_p95_equiv` | 2.789411 | 0.938 | 0.343 | 0.660 | 0.503 |  |
| `kde_p95_equiv` | 2.583261 | 0.930 | 0.449 | 0.708 | 0.606 |  |
| `evt_p99_equiv` | 3.441705 | 0.956 | 0.162 | 0.577 | 0.277 |  |
| `parametric_p99_equiv` | 3.406276 | 0.956 | 0.162 | 0.577 | 0.277 |  |
| `kde_p99_equiv` | 3.149982 | 0.968 | 0.226 | 0.609 | 0.367 |  |
| `evt_p999_equiv` | 10.283905 | 0.000 | 0.000 | 0.498 | 0.000 |  |
| `parametric_p999_equiv` | 4.261250 | 0.935 | 0.109 | 0.551 | 0.196 |  |
| `kde_p999_equiv` | 10.997404 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 3.432184 | 0.956 | 0.162 | 0.577 | 0.277 |  |
| `iqr` | 4.058313 | 0.939 | 0.117 | 0.555 | 0.208 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
