# Deployment setup: `w_proj_perchannel_knn16_fp32`

Combo 6 of the case 1 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 1
- Model hash: `086acf0275b8`
- Backbone: weight-only int8 (projections tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.697301**.

At this threshold on the case 1 test split: P=0.982, R=0.614, A=0.801, F1=0.755.

Ranking quality (threshold-independent): AUC=0.8199, pAUC=0.8393.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.610804 | 0.931 | 0.720 | 0.833 | 0.812 |  |
| `percentile_same_machine_99.0` | 1.697301 | 0.982 | 0.614 | 0.801 | 0.755 | **<- active** |
| `percentile_same_machine_99.5` | 1.738560 | 0.987 | 0.580 | 0.786 | 0.730 |  |
| `evt_p95_equiv` | 1.610117 | 0.931 | 0.720 | 0.833 | 0.812 |  |
| `parametric_p95_equiv` | 1.633930 | 0.953 | 0.693 | 0.830 | 0.803 |  |
| `kde_p95_equiv` | 1.625547 | 0.954 | 0.701 | 0.833 | 0.808 |  |
| `evt_p99_equiv` | 1.698378 | 0.982 | 0.614 | 0.801 | 0.755 |  |
| `parametric_p99_equiv` | 1.699722 | 0.982 | 0.614 | 0.801 | 0.755 |  |
| `kde_p99_equiv` | 1.715487 | 0.988 | 0.598 | 0.795 | 0.745 |  |
| `evt_p999_equiv` | 1.859588 | 1.000 | 0.511 | 0.756 | 0.677 |  |
| `parametric_p999_equiv` | 1.764187 | 0.987 | 0.557 | 0.775 | 0.712 |  |
| `kde_p999_equiv` | 1.886944 | 1.000 | 0.511 | 0.756 | 0.677 |  |
| `mad` | 1.805011 | 1.000 | 0.534 | 0.767 | 0.696 |  |
| `iqr` | 1.931659 | 1.000 | 0.504 | 0.752 | 0.670 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
