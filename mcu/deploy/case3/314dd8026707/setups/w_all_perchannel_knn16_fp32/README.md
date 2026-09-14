# Deployment setup: `w_all_perchannel_knn16_fp32`

Combo 2 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.380286**.

At this threshold on the case 3 test split: P=0.973, R=0.808, A=0.892, F1=0.882.

Ranking quality (threshold-independent): AUC=0.9726, pAUC=0.9201.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.332726 | 0.944 | 0.894 | 0.921 | 0.919 |  |
| `percentile_same_machine_99.0` | 1.380286 | 0.973 | 0.808 | 0.892 | 0.882 | **<- active** |
| `percentile_same_machine_99.5` | 1.408788 | 0.985 | 0.728 | 0.858 | 0.837 |  |
| `evt_p95_equiv` | 1.330436 | 0.933 | 0.898 | 0.917 | 0.915 |  |
| `parametric_p95_equiv` | 1.371884 | 0.969 | 0.838 | 0.906 | 0.899 |  |
| `kde_p95_equiv` | 1.354519 | 0.962 | 0.864 | 0.915 | 0.911 |  |
| `evt_p99_equiv` | 1.406231 | 0.985 | 0.736 | 0.862 | 0.842 |  |
| `parametric_p99_equiv` | 1.490758 | 0.992 | 0.487 | 0.742 | 0.653 |  |
| `kde_p99_equiv` | 1.431829 | 0.989 | 0.683 | 0.838 | 0.808 |  |
| `evt_p999_equiv` | 1.543428 | 1.000 | 0.321 | 0.660 | 0.486 |  |
| `parametric_p999_equiv` | 1.612273 | 1.000 | 0.253 | 0.626 | 0.404 |  |
| `kde_p999_equiv` | 1.812541 | 1.000 | 0.204 | 0.602 | 0.339 |  |
| `mad` | 1.995201 | 1.000 | 0.147 | 0.574 | 0.257 |  |
| `iqr` | 2.339347 | 1.000 | 0.117 | 0.558 | 0.209 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
