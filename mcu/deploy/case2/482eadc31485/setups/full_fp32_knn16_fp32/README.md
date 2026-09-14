# Deployment setup: `full_fp32_knn16_fp32`

Combo 28 of the case 2 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 2
- Model hash: `482eadc31485`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.324376**.

At this threshold on the case 2 test split: P=0.988, R=0.913, A=0.951, F1=0.949.

Ranking quality (threshold-independent): AUC=0.9620, pAUC=0.9631.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.288532 | 0.950 | 0.936 | 0.943 | 0.943 |  |
| `percentile_same_machine_99.0` | 1.324376 | 0.988 | 0.913 | 0.951 | 0.949 | **<- active** |
| `percentile_same_machine_99.5` | 1.334590 | 0.996 | 0.909 | 0.953 | 0.951 |  |
| `evt_p95_equiv` | 1.289638 | 0.950 | 0.936 | 0.943 | 0.943 |  |
| `parametric_p95_equiv` | 1.295278 | 0.957 | 0.932 | 0.945 | 0.945 |  |
| `kde_p95_equiv` | 1.295445 | 0.957 | 0.932 | 0.945 | 0.945 |  |
| `evt_p99_equiv` | 1.326590 | 0.988 | 0.913 | 0.951 | 0.949 |  |
| `parametric_p99_equiv` | 1.359136 | 0.996 | 0.906 | 0.951 | 0.949 |  |
| `kde_p99_equiv` | 1.334453 | 0.996 | 0.909 | 0.953 | 0.951 |  |
| `evt_p999_equiv` | 1.360768 | 0.996 | 0.906 | 0.951 | 0.949 |  |
| `parametric_p999_equiv` | 1.434464 | 1.000 | 0.898 | 0.949 | 0.946 |  |
| `kde_p999_equiv` | 1.371406 | 0.996 | 0.906 | 0.951 | 0.949 |  |
| `mad` | 1.473829 | 1.000 | 0.879 | 0.940 | 0.936 |  |
| `iqr` | 1.628376 | 1.000 | 0.713 | 0.857 | 0.833 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
