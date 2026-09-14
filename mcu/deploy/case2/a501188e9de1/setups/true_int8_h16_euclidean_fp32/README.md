# Deployment setup: `true_int8_h16_euclidean_fp32`

Combo 15 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.844840**.

At this threshold on the case 2 test split: P=0.996, R=0.940, A=0.968, F1=0.967.

Ranking quality (threshold-independent): AUC=0.9844, pAUC=0.9708.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.794833 | 0.950 | 0.940 | 0.945 | 0.945 |  |
| `percentile_same_machine_99.0` | 1.844840 | 0.996 | 0.940 | 0.968 | 0.967 | **<- active** |
| `percentile_same_machine_99.5` | 1.870080 | 1.000 | 0.928 | 0.964 | 0.963 |  |
| `evt_p95_equiv` | 1.792282 | 0.950 | 0.940 | 0.945 | 0.945 |  |
| `parametric_p95_equiv` | 1.788923 | 0.947 | 0.940 | 0.943 | 0.943 |  |
| `kde_p95_equiv` | 1.798276 | 0.954 | 0.940 | 0.947 | 0.947 |  |
| `evt_p99_equiv` | 1.867199 | 1.000 | 0.932 | 0.966 | 0.965 |  |
| `parametric_p99_equiv` | 1.864489 | 1.000 | 0.932 | 0.966 | 0.965 |  |
| `kde_p99_equiv` | 1.861542 | 1.000 | 0.932 | 0.966 | 0.965 |  |
| `evt_p999_equiv` | 1.945726 | 1.000 | 0.887 | 0.943 | 0.940 |  |
| `parametric_p999_equiv` | 1.952991 | 1.000 | 0.879 | 0.940 | 0.936 |  |
| `kde_p999_equiv` | 1.961857 | 1.000 | 0.879 | 0.940 | 0.936 |  |
| `mad` | 1.960748 | 1.000 | 0.879 | 0.940 | 0.936 |  |
| `iqr` | 2.079513 | 1.000 | 0.774 | 0.887 | 0.872 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
