# Deployment setup: `w_all_perchannel_euclidean_fp32`

Combo 1 of the case 3 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 3
- Model hash: `314dd8026707`
- Backbone: weight-only int8 (all tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.644203**.

At this threshold on the case 3 test split: P=0.959, R=0.268, A=0.628, F1=0.419.

Ranking quality (threshold-independent): AUC=0.8592, pAUC=0.7155.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.207357 | 0.931 | 0.506 | 0.734 | 0.655 |  |
| `percentile_same_machine_99.0` | 2.644203 | 0.959 | 0.268 | 0.628 | 0.419 | **<- active** |
| `percentile_same_machine_99.5` | 6.178116 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.197879 | 0.931 | 0.509 | 0.736 | 0.659 |  |
| `parametric_p95_equiv` | 2.374476 | 0.940 | 0.415 | 0.694 | 0.576 |  |
| `kde_p95_equiv` | 2.250083 | 0.929 | 0.491 | 0.726 | 0.642 |  |
| `evt_p99_equiv` | 3.018999 | 0.960 | 0.181 | 0.587 | 0.305 |  |
| `parametric_p99_equiv` | 2.958699 | 0.962 | 0.189 | 0.591 | 0.315 |  |
| `kde_p99_equiv` | 2.739706 | 0.971 | 0.257 | 0.625 | 0.406 |  |
| `evt_p999_equiv` | 6.805400 | 0.667 | 0.015 | 0.504 | 0.030 |  |
| `parametric_p999_equiv` | 3.786025 | 0.935 | 0.109 | 0.551 | 0.196 |  |
| `kde_p999_equiv` | 7.030939 | 1.000 | 0.008 | 0.504 | 0.015 |  |
| `mad` | 3.104812 | 0.952 | 0.151 | 0.572 | 0.261 |  |
| `iqr` | 3.721711 | 0.938 | 0.113 | 0.553 | 0.202 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
