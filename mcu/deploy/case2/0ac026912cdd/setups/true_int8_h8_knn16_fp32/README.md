# Deployment setup: `true_int8_h8_knn16_fp32`

Combo 13 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.690098**.

At this threshold on the case 2 test split: P=0.991, R=0.875, A=0.934, F1=0.930.

Ranking quality (threshold-independent): AUC=0.9813, pAUC=0.9758.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.580692 | 0.952 | 0.966 | 0.958 | 0.959 |  |
| `percentile_same_machine_99.0` | 1.690098 | 0.991 | 0.875 | 0.934 | 0.930 | **<- active** |
| `percentile_same_machine_99.5` | 1.703665 | 0.991 | 0.860 | 0.926 | 0.921 |  |
| `evt_p95_equiv` | 1.584625 | 0.952 | 0.966 | 0.958 | 0.959 |  |
| `parametric_p95_equiv` | 1.588191 | 0.959 | 0.966 | 0.962 | 0.962 |  |
| `kde_p95_equiv` | 1.589069 | 0.962 | 0.966 | 0.964 | 0.964 |  |
| `evt_p99_equiv` | 1.677189 | 0.992 | 0.894 | 0.943 | 0.940 |  |
| `parametric_p99_equiv` | 1.681115 | 0.992 | 0.883 | 0.938 | 0.934 |  |
| `kde_p99_equiv` | 1.688928 | 0.991 | 0.875 | 0.934 | 0.930 |  |
| `evt_p999_equiv` | 1.779797 | 1.000 | 0.781 | 0.891 | 0.877 |  |
| `parametric_p999_equiv` | 1.791750 | 1.000 | 0.766 | 0.883 | 0.868 |  |
| `kde_p999_equiv` | 1.785216 | 1.000 | 0.766 | 0.883 | 0.868 |  |
| `mad` | 1.809475 | 1.000 | 0.743 | 0.872 | 0.853 |  |
| `iqr` | 1.956848 | 1.000 | 0.498 | 0.749 | 0.665 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
