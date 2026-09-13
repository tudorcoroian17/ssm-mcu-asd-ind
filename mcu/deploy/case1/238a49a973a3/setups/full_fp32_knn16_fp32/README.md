# Deployment setup: `full_fp32_knn16_fp32`

Combo 28 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **2.493903**.

At this threshold on the case 1 test split: P=0.984, R=0.462, A=0.727, F1=0.629.

Ranking quality (threshold-independent): AUC=0.8425, pAUC=0.8142.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.941265 | 0.941 | 0.659 | 0.809 | 0.775 |  |
| `percentile_same_machine_99.0` | 2.493903 | 0.984 | 0.462 | 0.727 | 0.629 | **<- active** |
| `percentile_same_machine_99.5` | 2.547635 | 0.991 | 0.424 | 0.710 | 0.594 |  |
| `evt_p95_equiv` | 2.013845 | 0.964 | 0.617 | 0.797 | 0.753 |  |
| `parametric_p95_equiv` | 2.017980 | 0.964 | 0.617 | 0.797 | 0.753 |  |
| `kde_p95_equiv` | 1.966770 | 0.944 | 0.640 | 0.801 | 0.763 |  |
| `evt_p99_equiv` | 2.405966 | 0.977 | 0.473 | 0.731 | 0.638 |  |
| `parametric_p99_equiv` | 2.182358 | 0.972 | 0.527 | 0.756 | 0.683 |  |
| `kde_p99_equiv` | 2.503877 | 0.992 | 0.455 | 0.725 | 0.623 |  |
| `evt_p999_equiv` | 2.810156 | 1.000 | 0.258 | 0.629 | 0.410 |  |
| `parametric_p999_equiv` | 2.349608 | 0.977 | 0.481 | 0.735 | 0.645 |  |
| `kde_p999_equiv` | 2.673401 | 1.000 | 0.348 | 0.674 | 0.517 |  |
| `mad` | 2.004471 | 0.965 | 0.621 | 0.799 | 0.756 |  |
| `iqr` | 2.168927 | 0.972 | 0.534 | 0.759 | 0.689 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
