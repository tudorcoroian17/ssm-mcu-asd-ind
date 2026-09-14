# Deployment setup: `full_fp32_knn16_fp32`

Combo 28 of the case 4 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 4
- Model hash: `3660d1d10006`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **0.894152**.

At this threshold on the case 4 test split: P=1.000, R=0.992, A=0.996, F1=0.996.

Ranking quality (threshold-independent): AUC=0.9925, pAUC=0.9960.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 0.867243 | 0.953 | 0.992 | 0.972 | 0.972 |  |
| `percentile_same_machine_99.0` | 0.894152 | 1.000 | 0.992 | 0.996 | 0.996 | **<- active** |
| `percentile_same_machine_99.5` | 0.906519 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `evt_p95_equiv` | 0.867346 | 0.953 | 0.992 | 0.972 | 0.972 |  |
| `parametric_p95_equiv` | 0.870646 | 0.960 | 0.992 | 0.975 | 0.976 |  |
| `kde_p95_equiv` | 0.868519 | 0.960 | 0.992 | 0.975 | 0.976 |  |
| `evt_p99_equiv` | 0.894304 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `parametric_p99_equiv` | 0.893440 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `kde_p99_equiv` | 0.897195 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `evt_p999_equiv` | 0.928120 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `parametric_p999_equiv` | 0.919208 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `kde_p999_equiv` | 0.929715 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `mad` | 0.920360 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `iqr` | 0.955766 | 1.000 | 0.992 | 0.996 | 0.996 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
