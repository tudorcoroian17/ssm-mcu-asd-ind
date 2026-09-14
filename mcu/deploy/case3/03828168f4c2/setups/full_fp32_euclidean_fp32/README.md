# Deployment setup: `full_fp32_euclidean_fp32`

Combo 27 of the case 3 deployment matrix.

## Matchup

- Config: `352f70960ed3.yaml`
- Held-out case: 3
- Model hash: `03828168f4c2`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.822431**.

At this threshold on the case 3 test split: P=0.959, R=0.264, A=0.626, F1=0.414.

Ranking quality (threshold-independent): AUC=0.7828, pAUC=0.6748.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.489614 | 0.919 | 0.430 | 0.696 | 0.586 |  |
| `percentile_same_machine_99.0` | 2.822431 | 0.959 | 0.264 | 0.626 | 0.414 | **<- active** |
| `percentile_same_machine_99.5` | 10.359066 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.467338 | 0.906 | 0.434 | 0.694 | 0.587 |  |
| `parametric_p95_equiv` | 2.843242 | 0.958 | 0.260 | 0.625 | 0.409 |  |
| `kde_p95_equiv` | 2.577800 | 0.938 | 0.396 | 0.685 | 0.557 |  |
| `evt_p99_equiv` | 3.403096 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `parametric_p99_equiv` | 3.508205 | 0.951 | 0.147 | 0.570 | 0.255 |  |
| `kde_p99_equiv` | 3.046906 | 0.968 | 0.226 | 0.609 | 0.367 |  |
| `evt_p999_equiv` | 10.129754 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 4.440077 | 0.926 | 0.094 | 0.543 | 0.171 |  |
| `kde_p999_equiv` | 11.880712 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 3.430123 | 0.955 | 0.158 | 0.575 | 0.272 |  |
| `iqr` | 4.054522 | 0.935 | 0.109 | 0.551 | 0.196 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
