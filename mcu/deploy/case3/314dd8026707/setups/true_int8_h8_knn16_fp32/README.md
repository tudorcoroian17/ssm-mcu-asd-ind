# Deployment setup: `true_int8_h8_knn16_fp32`

Combo 13 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.493684**.

At this threshold on the case 3 test split: P=0.992, R=0.909, A=0.951, F1=0.949.

Ranking quality (threshold-independent): AUC=0.9883, pAUC=0.9753.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.426614 | 0.966 | 0.962 | 0.964 | 0.964 |  |
| `percentile_same_machine_99.0` | 1.493684 | 0.992 | 0.909 | 0.951 | 0.949 | **<- active** |
| `percentile_same_machine_99.5` | 1.548330 | 0.996 | 0.853 | 0.925 | 0.919 |  |
| `evt_p95_equiv` | 1.424452 | 0.962 | 0.962 | 0.962 | 0.962 |  |
| `parametric_p95_equiv` | 1.467856 | 0.976 | 0.936 | 0.957 | 0.956 |  |
| `kde_p95_equiv` | 1.439022 | 0.970 | 0.962 | 0.966 | 0.966 |  |
| `evt_p99_equiv` | 1.512303 | 0.996 | 0.894 | 0.945 | 0.942 |  |
| `parametric_p99_equiv` | 1.581791 | 1.000 | 0.789 | 0.894 | 0.882 |  |
| `kde_p99_equiv` | 1.518851 | 0.996 | 0.887 | 0.942 | 0.938 |  |
| `evt_p999_equiv` | 1.710814 | 1.000 | 0.566 | 0.783 | 0.723 |  |
| `parametric_p999_equiv` | 1.712244 | 1.000 | 0.566 | 0.783 | 0.723 |  |
| `kde_p999_equiv` | 1.812769 | 1.000 | 0.396 | 0.698 | 0.568 |  |
| `mad` | 1.823075 | 1.000 | 0.381 | 0.691 | 0.552 |  |
| `iqr` | 2.035492 | 1.000 | 0.219 | 0.609 | 0.359 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
