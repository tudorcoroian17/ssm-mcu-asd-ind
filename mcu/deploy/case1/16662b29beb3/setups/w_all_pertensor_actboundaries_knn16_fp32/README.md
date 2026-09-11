# Deployment setup: `w_all_pertensor_actboundaries_knn16_fp32`

Combo 10 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.308755**.

At this threshold on the case 1 test split: P=0.982, R=0.405, A=0.699, F1=0.574.

Ranking quality (threshold-independent): AUC=0.9134, pAUC=0.8643.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.720251 | 0.957 | 0.765 | 0.866 | 0.851 |  |
| `percentile_same_machine_99.0` | 2.308755 | 0.982 | 0.405 | 0.699 | 0.574 | **<- active** |
| `percentile_same_machine_99.5` | 2.395239 | 0.978 | 0.341 | 0.667 | 0.506 |  |
| `evt_p95_equiv` | 1.751765 | 0.966 | 0.754 | 0.864 | 0.847 |  |
| `parametric_p95_equiv` | 1.770784 | 0.975 | 0.739 | 0.860 | 0.841 |  |
| `kde_p95_equiv` | 1.724838 | 0.957 | 0.765 | 0.866 | 0.851 |  |
| `evt_p99_equiv` | 2.197541 | 0.976 | 0.466 | 0.727 | 0.631 |  |
| `parametric_p99_equiv` | 1.984935 | 0.975 | 0.580 | 0.782 | 0.727 |  |
| `kde_p99_equiv` | 2.325900 | 0.981 | 0.402 | 0.697 | 0.570 |  |
| `evt_p999_equiv` | 3.036503 | 1.000 | 0.083 | 0.542 | 0.154 |  |
| `parametric_p999_equiv` | 2.231903 | 0.976 | 0.455 | 0.722 | 0.620 |  |
| `kde_p999_equiv` | 2.492941 | 1.000 | 0.314 | 0.657 | 0.478 |  |
| `mad` | 1.610633 | 0.912 | 0.860 | 0.888 | 0.885 |  |
| `iqr` | 1.715985 | 0.958 | 0.769 | 0.867 | 0.853 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
