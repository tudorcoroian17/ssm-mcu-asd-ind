# Deployment setup: `true_int8_h16_euclidean_fp32`

Combo 15 of the case 2 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 2
- Model hash: `0ac026912cdd`
- Backbone: true int8 arithmetic, h storage = int16
- Head: euclidean head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.879577**.

At this threshold on the case 2 test split: P=0.983, R=0.894, A=0.940, F1=0.937.

Ranking quality (threshold-independent): AUC=0.9742, pAUC=0.9591.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.848088 | 0.946 | 0.925 | 0.936 | 0.935 |  |
| `percentile_same_machine_99.0` | 1.879577 | 0.983 | 0.894 | 0.940 | 0.937 | **<- active** |
| `percentile_same_machine_99.5` | 1.892609 | 0.991 | 0.879 | 0.936 | 0.932 |  |
| `evt_p95_equiv` | 1.848848 | 0.946 | 0.925 | 0.936 | 0.935 |  |
| `parametric_p95_equiv` | 1.851325 | 0.957 | 0.925 | 0.942 | 0.940 |  |
| `kde_p95_equiv` | 1.853008 | 0.961 | 0.925 | 0.943 | 0.942 |  |
| `evt_p99_equiv` | 1.887078 | 0.987 | 0.891 | 0.940 | 0.937 |  |
| `parametric_p99_equiv` | 1.908634 | 1.000 | 0.868 | 0.934 | 0.929 |  |
| `kde_p99_equiv` | 1.892783 | 0.991 | 0.879 | 0.936 | 0.932 |  |
| `evt_p999_equiv` | 1.914937 | 1.000 | 0.857 | 0.928 | 0.923 |  |
| `parametric_p999_equiv` | 1.973511 | 1.000 | 0.800 | 0.900 | 0.889 |  |
| `kde_p999_equiv` | 1.931718 | 1.000 | 0.842 | 0.921 | 0.914 |  |
| `mad` | 2.006582 | 1.000 | 0.755 | 0.877 | 0.860 |  |
| `iqr` | 2.108851 | 1.000 | 0.630 | 0.815 | 0.773 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
