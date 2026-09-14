# Deployment setup: `w_all_pertensor_euclidean_fp32`

Combo 3 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.848516**.

At this threshold on the case 4 test split: P=0.983, R=0.883, A=0.934, F1=0.930.

Ranking quality (threshold-independent): AUC=0.9690, pAUC=0.9456.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.805220 | 0.913 | 0.909 | 0.911 | 0.911 |  |
| `percentile_same_machine_99.0` | 1.848516 | 0.983 | 0.883 | 0.934 | 0.930 | **<- active** |
| `percentile_same_machine_99.5` | 1.860883 | 0.983 | 0.875 | 0.930 | 0.926 |  |
| `evt_p95_equiv` | 1.802464 | 0.909 | 0.909 | 0.909 | 0.909 |  |
| `parametric_p95_equiv` | 1.810113 | 0.927 | 0.906 | 0.917 | 0.916 |  |
| `kde_p95_equiv` | 1.818598 | 0.945 | 0.902 | 0.925 | 0.923 |  |
| `evt_p99_equiv` | 1.857446 | 0.983 | 0.879 | 0.932 | 0.928 |  |
| `parametric_p99_equiv` | 1.880210 | 0.996 | 0.868 | 0.932 | 0.927 |  |
| `kde_p99_equiv` | 1.880920 | 0.996 | 0.868 | 0.932 | 0.927 |  |
| `evt_p999_equiv` | 1.906930 | 0.996 | 0.849 | 0.923 | 0.916 |  |
| `parametric_p999_equiv` | 1.948793 | 1.000 | 0.826 | 0.913 | 0.905 |  |
| `kde_p999_equiv` | 1.943958 | 1.000 | 0.830 | 0.915 | 0.907 |  |
| `mad` | 2.123383 | 1.000 | 0.709 | 0.855 | 0.830 |  |
| `iqr` | 2.330796 | 1.000 | 0.570 | 0.785 | 0.726 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
