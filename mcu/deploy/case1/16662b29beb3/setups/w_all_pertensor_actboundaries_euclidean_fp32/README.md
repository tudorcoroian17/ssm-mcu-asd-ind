# Deployment setup: `w_all_pertensor_actboundaries_euclidean_fp32`

Combo 9 of the case 1 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 1
- Model hash: `16662b29beb3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.748794**.

At this threshold on the case 1 test split: P=0.826, R=0.072, A=0.528, F1=0.132.

Ranking quality (threshold-independent): AUC=0.9566, pAUC=0.7930.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.096241 | 0.944 | 0.705 | 0.831 | 0.807 |  |
| `percentile_same_machine_99.0` | 3.748794 | 0.826 | 0.072 | 0.528 | 0.132 | **<- active** |
| `percentile_same_machine_99.5` | 8.391753 | 0.875 | 0.027 | 0.511 | 0.051 |  |
| `evt_p95_equiv` | 2.144693 | 0.956 | 0.663 | 0.816 | 0.783 |  |
| `parametric_p95_equiv` | 2.217264 | 0.952 | 0.606 | 0.788 | 0.741 |  |
| `kde_p95_equiv` | 2.217346 | 0.952 | 0.606 | 0.788 | 0.741 |  |
| `evt_p99_equiv` | 3.973542 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.604739 | 0.938 | 0.398 | 0.686 | 0.559 |  |
| `kde_p99_equiv` | 7.833858 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 29.877194 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.120091 | 0.929 | 0.197 | 0.591 | 0.325 |  |
| `kde_p999_equiv` | 8.624033 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.799535 | 0.873 | 0.936 | 0.900 | 0.903 |  |
| `iqr` | 1.943786 | 0.888 | 0.875 | 0.883 | 0.882 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
