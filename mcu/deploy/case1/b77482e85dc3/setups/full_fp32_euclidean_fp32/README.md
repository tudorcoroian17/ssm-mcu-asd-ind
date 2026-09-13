# Deployment setup: `full_fp32_euclidean_fp32`

Combo 27 of the case 1 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 1
- Model hash: `b77482e85dc3`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **3.686203**.

At this threshold on the case 1 test split: P=0.818, R=0.068, A=0.527, F1=0.126.

Ranking quality (threshold-independent): AUC=0.9485, pAUC=0.8239.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.710113 | 0.955 | 0.811 | 0.886 | 0.877 |  |
| `percentile_same_machine_99.0` | 3.686203 | 0.818 | 0.068 | 0.527 | 0.126 | **<- active** |
| `percentile_same_machine_99.5` | 11.007386 | 0.846 | 0.042 | 0.517 | 0.079 |  |
| `evt_p95_equiv` | 1.720689 | 0.955 | 0.795 | 0.879 | 0.868 |  |
| `parametric_p95_equiv` | 1.978543 | 0.954 | 0.545 | 0.759 | 0.694 |  |
| `kde_p95_equiv` | 1.886783 | 0.956 | 0.652 | 0.811 | 0.775 |  |
| `evt_p99_equiv` | 3.381324 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `parametric_p99_equiv` | 2.362037 | 0.938 | 0.231 | 0.608 | 0.371 |  |
| `kde_p99_equiv` | 10.254967 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `evt_p999_equiv` | 38.308988 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `parametric_p999_equiv` | 2.880910 | 0.818 | 0.068 | 0.527 | 0.126 |  |
| `kde_p999_equiv` | 11.360783 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 1.711294 | 0.955 | 0.811 | 0.886 | 0.877 |  |
| `iqr` | 1.876231 | 0.956 | 0.652 | 0.811 | 0.775 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
