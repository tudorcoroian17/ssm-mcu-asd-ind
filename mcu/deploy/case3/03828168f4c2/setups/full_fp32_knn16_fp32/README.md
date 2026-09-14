# Deployment setup: `full_fp32_knn16_fp32`

Combo 28 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **1.121741**.

At this threshold on the case 3 test split: P=0.985, R=0.989, A=0.987, F1=0.987.

Ranking quality (threshold-independent): AUC=0.9887, pAUC=0.9940.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.088106 | 0.978 | 0.989 | 0.983 | 0.983 |  |
| `percentile_same_machine_99.0` | 1.121741 | 0.985 | 0.989 | 0.987 | 0.987 | **<- active** |
| `percentile_same_machine_99.5` | 1.141737 | 0.989 | 0.989 | 0.989 | 0.989 |  |
| `evt_p95_equiv` | 1.085074 | 0.967 | 0.989 | 0.977 | 0.978 |  |
| `parametric_p95_equiv` | 1.197103 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `kde_p95_equiv` | 1.106068 | 0.981 | 0.989 | 0.985 | 0.985 |  |
| `evt_p99_equiv` | 1.152365 | 0.996 | 0.989 | 0.992 | 0.992 |  |
| `parametric_p99_equiv` | 1.346488 | 1.000 | 0.970 | 0.985 | 0.985 |  |
| `kde_p99_equiv` | 1.165034 | 1.000 | 0.989 | 0.994 | 0.994 |  |
| `evt_p999_equiv` | 1.309540 | 1.000 | 0.974 | 0.987 | 0.987 |  |
| `parametric_p999_equiv` | 1.518851 | 1.000 | 0.868 | 0.934 | 0.929 |  |
| `kde_p999_equiv` | 2.737083 | 1.000 | 0.102 | 0.551 | 0.185 |  |
| `mad` | 1.449424 | 1.000 | 0.928 | 0.964 | 0.963 |  |
| `iqr` | 1.703427 | 1.000 | 0.302 | 0.651 | 0.464 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
