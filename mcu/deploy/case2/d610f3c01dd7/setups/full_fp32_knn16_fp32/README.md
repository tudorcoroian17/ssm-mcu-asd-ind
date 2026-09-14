# Deployment setup: `full_fp32_knn16_fp32`

Combo 28 of the case 2 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 2
- Model hash: `d610f3c01dd7`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_knn16` = **0.867016**.

At this threshold on the case 2 test split: P=1.000, R=0.996, A=0.998, F1=0.998.

Ranking quality (threshold-independent): AUC=0.9962, pAUC=0.9980.

## Swapping the threshold by hand

Edit the `ssm_threshold_knn16` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 0.840038 | 0.971 | 0.996 | 0.983 | 0.983 |  |
| `percentile_same_machine_99.0` | 0.867016 | 1.000 | 0.996 | 0.998 | 0.998 | **<- active** |
| `percentile_same_machine_99.5` | 0.875534 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `evt_p95_equiv` | 0.840477 | 0.971 | 0.996 | 0.983 | 0.983 |  |
| `parametric_p95_equiv` | 0.841377 | 0.971 | 0.996 | 0.983 | 0.983 |  |
| `kde_p95_equiv` | 0.843931 | 0.971 | 0.996 | 0.983 | 0.983 |  |
| `evt_p99_equiv` | 0.868211 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `parametric_p99_equiv` | 0.884387 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `kde_p99_equiv` | 0.873674 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `evt_p999_equiv` | 0.895789 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `parametric_p999_equiv` | 0.935214 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `kde_p999_equiv` | 0.904791 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `mad` | 0.944057 | 1.000 | 0.992 | 0.996 | 0.996 |  |
| `iqr` | 1.052566 | 1.000 | 0.955 | 0.977 | 0.977 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
