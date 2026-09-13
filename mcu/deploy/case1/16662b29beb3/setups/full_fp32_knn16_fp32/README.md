# Deployment setup: `full_fp32_knn16_fp32`

Combo 20 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
- Backbone: full fp32 (no quantization)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.339616**.

At this threshold on the case 1 test split: P=0.983, R=0.432, A=0.712, F1=0.600.

Ranking quality (threshold-independent): AUC=0.9106, pAUC=0.8554.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.722472 | 0.944 | 0.761 | 0.858 | 0.843 |  |
| `percentile_same_machine_99.0` | 2.339616 | 0.983 | 0.432 | 0.712 | 0.600 | **<- active** |
| `percentile_same_machine_99.5` | 2.406699 | 0.981 | 0.390 | 0.691 | 0.558 |  |
| `evt_p95_equiv` | 1.742074 | 0.956 | 0.746 | 0.856 | 0.838 |  |
| `parametric_p95_equiv` | 1.637412 | 0.904 | 0.822 | 0.867 | 0.861 |  |
| `kde_p95_equiv` | 1.733286 | 0.957 | 0.754 | 0.860 | 0.843 |  |
| `evt_p99_equiv` | 2.212954 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `parametric_p99_equiv` | 1.857070 | 0.978 | 0.689 | 0.837 | 0.809 |  |
| `kde_p99_equiv` | 2.339173 | 0.983 | 0.432 | 0.712 | 0.600 |  |
| `evt_p999_equiv` | 3.752682 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 2.138493 | 0.978 | 0.500 | 0.744 | 0.662 |  |
| `kde_p999_equiv` | 2.510290 | 1.000 | 0.299 | 0.650 | 0.461 |  |
| `mad` | 1.475453 | 0.890 | 0.917 | 0.902 | 0.903 |  |
| `iqr` | 1.591548 | 0.892 | 0.845 | 0.871 | 0.868 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
