# Deployment setup: `full_fp32_euclidean_fp32`

Combo 27 of the case 1 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 1
- Model hash: `238a49a973a3`
- Backbone: full fp32 (no quantization)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **4.396334**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.8826, pAUC=0.6771.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.728007 | 0.905 | 0.470 | 0.710 | 0.618 |  |
| `percentile_same_machine_99.0` | 4.396334 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 9.314260 | 0.769 | 0.038 | 0.513 | 0.072 |  |
| `evt_p95_equiv` | 2.804790 | 0.924 | 0.417 | 0.691 | 0.574 |  |
| `parametric_p95_equiv` | 2.562143 | 0.872 | 0.542 | 0.731 | 0.668 |  |
| `kde_p95_equiv` | 2.826672 | 0.928 | 0.390 | 0.680 | 0.549 |  |
| `evt_p99_equiv` | 4.782290 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 3.070929 | 0.896 | 0.261 | 0.616 | 0.405 |  |
| `kde_p99_equiv` | 8.625432 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 14.043333 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 3.762241 | 0.846 | 0.083 | 0.534 | 0.152 |  |
| `kde_p999_equiv` | 9.626896 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.166732 | 0.873 | 0.731 | 0.812 | 0.796 |  |
| `iqr` | 2.371027 | 0.860 | 0.629 | 0.763 | 0.726 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
