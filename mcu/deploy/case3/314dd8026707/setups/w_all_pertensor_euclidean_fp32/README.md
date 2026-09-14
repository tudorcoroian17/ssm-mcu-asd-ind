# Deployment setup: `w_all_pertensor_euclidean_fp32`

Combo 3 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
- Backbone: weight-only int8 (all tensors, per-tensor)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.640387**.

At this threshold on the case 3 test split: P=0.959, R=0.268, A=0.628, F1=0.419.

Ranking quality (threshold-independent): AUC=0.8574, pAUC=0.7131.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.212150 | 0.931 | 0.506 | 0.734 | 0.655 |  |
| `percentile_same_machine_99.0` | 2.640387 | 0.959 | 0.268 | 0.628 | 0.419 | **<- active** |
| `percentile_same_machine_99.5` | 6.109839 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.199659 | 0.931 | 0.509 | 0.736 | 0.659 |  |
| `parametric_p95_equiv` | 2.374005 | 0.930 | 0.404 | 0.687 | 0.563 |  |
| `kde_p95_equiv` | 2.253026 | 0.927 | 0.479 | 0.721 | 0.632 |  |
| `evt_p99_equiv` | 3.005847 | 0.958 | 0.174 | 0.583 | 0.294 |  |
| `parametric_p99_equiv` | 2.956196 | 0.961 | 0.185 | 0.589 | 0.310 |  |
| `kde_p99_equiv` | 2.735245 | 0.971 | 0.257 | 0.625 | 0.406 |  |
| `evt_p999_equiv` | 6.806633 | 0.667 | 0.008 | 0.502 | 0.015 |  |
| `parametric_p999_equiv` | 3.780076 | 0.935 | 0.109 | 0.551 | 0.196 |  |
| `kde_p999_equiv` | 6.940959 | 1.000 | 0.008 | 0.504 | 0.015 |  |
| `mad` | 3.108072 | 0.950 | 0.143 | 0.568 | 0.249 |  |
| `iqr` | 3.713496 | 0.935 | 0.109 | 0.551 | 0.196 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
