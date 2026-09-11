# Deployment setup: `w_all_perchannel_knn16_fp32`

Combo 2 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.333873**.

At this threshold on the case 1 test split: P=0.983, R=0.436, A=0.714, F1=0.604.

Ranking quality (threshold-independent): AUC=0.9103, pAUC=0.8524.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.736163 | 0.943 | 0.754 | 0.854 | 0.838 |  |
| `percentile_same_machine_99.0` | 2.333873 | 0.983 | 0.436 | 0.714 | 0.604 | **<- active** |
| `percentile_same_machine_99.5` | 2.406788 | 0.980 | 0.379 | 0.686 | 0.546 |  |
| `evt_p95_equiv` | 1.764396 | 0.961 | 0.742 | 0.856 | 0.838 |  |
| `parametric_p95_equiv` | 1.723912 | 0.935 | 0.761 | 0.854 | 0.839 |  |
| `kde_p95_equiv` | 1.749990 | 0.956 | 0.746 | 0.856 | 0.838 |  |
| `evt_p99_equiv` | 2.216052 | 0.977 | 0.477 | 0.733 | 0.641 |  |
| `parametric_p99_equiv` | 1.954746 | 0.976 | 0.614 | 0.799 | 0.753 |  |
| `kde_p99_equiv` | 2.342371 | 0.983 | 0.428 | 0.710 | 0.596 |  |
| `evt_p999_equiv` | 3.296973 | 1.000 | 0.038 | 0.519 | 0.073 |  |
| `parametric_p999_equiv` | 2.221379 | 0.977 | 0.473 | 0.731 | 0.638 |  |
| `kde_p999_equiv` | 2.515139 | 1.000 | 0.299 | 0.650 | 0.461 |  |
| `mad` | 1.470761 | 0.890 | 0.917 | 0.902 | 0.903 |  |
| `iqr` | 1.588196 | 0.892 | 0.845 | 0.871 | 0.868 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
