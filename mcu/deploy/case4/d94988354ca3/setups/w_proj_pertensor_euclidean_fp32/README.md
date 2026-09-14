# Deployment setup: `w_proj_pertensor_euclidean_fp32`

Combo 7 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.833061**.

At this threshold on the case 4 test split: P=0.983, R=0.891, A=0.938, F1=0.935.

Ranking quality (threshold-independent): AUC=0.9723, pAUC=0.9477.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.790353 | 0.913 | 0.913 | 0.913 | 0.913 |  |
| `percentile_same_machine_99.0` | 1.833061 | 0.983 | 0.891 | 0.938 | 0.935 | **<- active** |
| `percentile_same_machine_99.5` | 1.845536 | 0.983 | 0.883 | 0.934 | 0.930 |  |
| `evt_p95_equiv` | 1.788016 | 0.910 | 0.913 | 0.911 | 0.911 |  |
| `parametric_p95_equiv` | 1.795502 | 0.923 | 0.909 | 0.917 | 0.916 |  |
| `kde_p95_equiv` | 1.803939 | 0.945 | 0.909 | 0.928 | 0.927 |  |
| `evt_p99_equiv` | 1.842159 | 0.983 | 0.887 | 0.936 | 0.933 |  |
| `parametric_p99_equiv` | 1.864842 | 0.996 | 0.875 | 0.936 | 0.932 |  |
| `kde_p99_equiv` | 1.865494 | 0.996 | 0.875 | 0.936 | 0.932 |  |
| `evt_p999_equiv` | 1.891224 | 0.996 | 0.857 | 0.926 | 0.921 |  |
| `parametric_p999_equiv` | 1.932679 | 1.000 | 0.830 | 0.915 | 0.907 |  |
| `kde_p999_equiv` | 1.927910 | 1.000 | 0.830 | 0.915 | 0.907 |  |
| `mad` | 2.111220 | 1.000 | 0.709 | 0.855 | 0.830 |  |
| `iqr` | 2.310977 | 1.000 | 0.574 | 0.787 | 0.729 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
