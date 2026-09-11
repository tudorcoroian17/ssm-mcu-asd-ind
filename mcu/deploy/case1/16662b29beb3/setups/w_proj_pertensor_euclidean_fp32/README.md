# Deployment setup: `w_proj_pertensor_euclidean_fp32`

Combo 7 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
- Backbone: weight-only int8 (projections tensors, per-tensor)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.861473**.

At this threshold on the case 1 test split: P=0.857, R=0.091, A=0.538, F1=0.164.

Ranking quality (threshold-independent): AUC=0.9472, pAUC=0.7475.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.231266 | 0.938 | 0.633 | 0.795 | 0.756 |  |
| `percentile_same_machine_99.0` | 3.861473 | 0.857 | 0.091 | 0.538 | 0.164 | **<- active** |
| `percentile_same_machine_99.5` | 8.551669 | 0.818 | 0.034 | 0.513 | 0.065 |  |
| `evt_p95_equiv` | 2.302744 | 0.952 | 0.595 | 0.782 | 0.732 |  |
| `parametric_p95_equiv` | 2.219669 | 0.928 | 0.633 | 0.792 | 0.752 |  |
| `kde_p95_equiv` | 2.350687 | 0.950 | 0.576 | 0.773 | 0.717 |  |
| `evt_p99_equiv` | 4.043507 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.639742 | 0.942 | 0.432 | 0.703 | 0.592 |  |
| `kde_p99_equiv` | 7.965239 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 20.169394 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.205790 | 0.938 | 0.231 | 0.608 | 0.371 |  |
| `kde_p999_equiv` | 8.800042 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.650874 | 0.876 | 0.989 | 0.924 | 0.929 |  |
| `iqr` | 1.752652 | 0.877 | 0.943 | 0.905 | 0.909 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
