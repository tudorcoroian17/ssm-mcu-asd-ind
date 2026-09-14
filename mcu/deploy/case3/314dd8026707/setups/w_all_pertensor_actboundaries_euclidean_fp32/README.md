# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
- Backbone: weight int8 (all, per-tensor) + activation-boundaries int8
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.624043**.

At this threshold on the case 3 test split: P=0.961, R=0.275, A=0.632, F1=0.428.

Ranking quality (threshold-independent): AUC=0.8841, pAUC=0.7323.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.230803 | 0.934 | 0.532 | 0.747 | 0.678 |  |
| `percentile_same_machine_99.0` | 2.624043 | 0.961 | 0.275 | 0.632 | 0.428 | **<- active** |
| `percentile_same_machine_99.5` | 5.951253 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.224431 | 0.934 | 0.536 | 0.749 | 0.681 |  |
| `parametric_p95_equiv` | 2.396977 | 0.939 | 0.468 | 0.719 | 0.625 |  |
| `kde_p95_equiv` | 2.277714 | 0.933 | 0.525 | 0.743 | 0.671 |  |
| `evt_p99_equiv` | 2.983403 | 0.966 | 0.211 | 0.602 | 0.347 |  |
| `parametric_p99_equiv` | 2.961743 | 0.967 | 0.219 | 0.606 | 0.357 |  |
| `kde_p99_equiv` | 2.722902 | 0.973 | 0.272 | 0.632 | 0.425 |  |
| `evt_p999_equiv` | 6.603873 | 0.667 | 0.008 | 0.502 | 0.015 |  |
| `parametric_p999_equiv` | 3.754390 | 0.938 | 0.113 | 0.553 | 0.202 |  |
| `kde_p999_equiv` | 6.776517 | 1.000 | 0.008 | 0.504 | 0.015 |  |
| `mad` | 3.164857 | 0.956 | 0.162 | 0.577 | 0.277 |  |
| `iqr` | 3.742507 | 0.938 | 0.113 | 0.553 | 0.202 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
