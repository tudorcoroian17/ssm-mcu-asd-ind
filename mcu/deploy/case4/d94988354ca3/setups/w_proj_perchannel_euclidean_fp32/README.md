# Deployment setup: `w_proj_perchannel_euclidean_fp32`

Combo 5 of the case 4 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 4
- Model hash: `d94988354ca3`
- Backbone: weight-only int8 (projections tensors, per-channel)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.848031**.

At this threshold on the case 4 test split: P=0.983, R=0.891, A=0.938, F1=0.935.

Ranking quality (threshold-independent): AUC=0.9716, pAUC=0.9469.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.805689 | 0.920 | 0.913 | 0.917 | 0.917 |  |
| `percentile_same_machine_99.0` | 1.848031 | 0.983 | 0.891 | 0.938 | 0.935 | **<- active** |
| `percentile_same_machine_99.5` | 1.860441 | 0.987 | 0.883 | 0.936 | 0.932 |  |
| `evt_p95_equiv` | 1.802274 | 0.910 | 0.913 | 0.911 | 0.911 |  |
| `parametric_p95_equiv` | 1.809828 | 0.920 | 0.909 | 0.915 | 0.915 |  |
| `kde_p95_equiv` | 1.818401 | 0.937 | 0.898 | 0.919 | 0.917 |  |
| `evt_p99_equiv` | 1.856927 | 0.983 | 0.883 | 0.934 | 0.930 |  |
| `parametric_p99_equiv` | 1.879659 | 0.996 | 0.872 | 0.934 | 0.930 |  |
| `kde_p99_equiv` | 1.880140 | 0.996 | 0.872 | 0.934 | 0.930 |  |
| `evt_p999_equiv` | 1.907157 | 0.996 | 0.845 | 0.921 | 0.914 |  |
| `parametric_p999_equiv` | 1.947974 | 1.000 | 0.826 | 0.913 | 0.905 |  |
| `kde_p999_equiv` | 1.943548 | 1.000 | 0.830 | 0.915 | 0.907 |  |
| `mad` | 2.127774 | 1.000 | 0.709 | 0.855 | 0.830 |  |
| `iqr` | 2.329504 | 1.000 | 0.570 | 0.785 | 0.726 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
