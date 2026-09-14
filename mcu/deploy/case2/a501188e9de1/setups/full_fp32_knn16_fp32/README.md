# Deployment setup: `full_fp32_knn16_fp32`

Combo 20 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: full fp32 (no quantization)
- Head: knn_clustered_16 head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: knn_clustered_16 only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.380421**.

At this threshold on the case 2 test split: P=0.992, R=0.883, A=0.938, F1=0.934.

Ranking quality (threshold-independent): AUC=0.9364, pAUC=0.9515.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.357741 | 0.972 | 0.906 | 0.940 | 0.938 |  |
| `percentile_same_machine_99.0` | 1.380421 | 0.992 | 0.883 | 0.938 | 0.934 | **<- active** |
| `percentile_same_machine_99.5` | 1.391911 | 0.991 | 0.872 | 0.932 | 0.928 |  |
| `evt_p95_equiv` | 1.355891 | 0.960 | 0.913 | 0.938 | 0.936 |  |
| `parametric_p95_equiv` | 1.364159 | 0.983 | 0.894 | 0.940 | 0.937 |  |
| `kde_p95_equiv` | 1.361868 | 0.979 | 0.894 | 0.938 | 0.935 |  |
| `evt_p99_equiv` | 1.385354 | 0.991 | 0.879 | 0.936 | 0.932 |  |
| `parametric_p99_equiv` | 1.412659 | 0.996 | 0.860 | 0.928 | 0.923 |  |
| `kde_p99_equiv` | 1.390542 | 0.991 | 0.872 | 0.932 | 0.928 |  |
| `evt_p999_equiv` | 1.424417 | 0.996 | 0.853 | 0.925 | 0.919 |  |
| `parametric_p999_equiv` | 1.467636 | 1.000 | 0.796 | 0.898 | 0.887 |  |
| `kde_p999_equiv` | 1.439069 | 1.000 | 0.853 | 0.926 | 0.921 |  |
| `mad` | 1.540611 | 1.000 | 0.675 | 0.838 | 0.806 |  |
| `iqr` | 1.658189 | 1.000 | 0.479 | 0.740 | 0.648 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
