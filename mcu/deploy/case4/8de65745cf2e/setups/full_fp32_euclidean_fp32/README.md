# Deployment setup: `full_fp32_euclidean_fp32`

Combo 27 of the case 4 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 4
- Model hash: `8de65745cf2e`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.270727**.

At this threshold on the case 4 test split: P=0.978, R=1.000, A=0.989, F1=0.989.

Ranking quality (threshold-independent): AUC=1.0000, pAUC=1.0000.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.227935 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `percentile_same_machine_99.0` | 1.270727 | 0.978 | 1.000 | 0.989 | 0.989 | **<- active** |
| `percentile_same_machine_99.5` | 1.288368 | 0.989 | 1.000 | 0.994 | 0.994 |  |
| `evt_p95_equiv` | 1.227786 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `parametric_p95_equiv` | 1.206156 | 0.923 | 1.000 | 0.958 | 0.960 |  |
| `kde_p95_equiv` | 1.230085 | 0.950 | 1.000 | 0.974 | 0.974 |  |
| `evt_p99_equiv` | 1.276323 | 0.978 | 1.000 | 0.989 | 0.989 |  |
| `parametric_p99_equiv` | 1.254888 | 0.971 | 1.000 | 0.985 | 0.985 |  |
| `kde_p99_equiv` | 1.277965 | 0.978 | 1.000 | 0.989 | 0.989 |  |
| `evt_p999_equiv` | 1.313835 | 0.993 | 1.000 | 0.996 | 0.996 |  |
| `parametric_p999_equiv` | 1.311856 | 0.993 | 1.000 | 0.996 | 0.996 |  |
| `kde_p999_equiv` | 1.325897 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `mad` | 1.286778 | 0.989 | 1.000 | 0.994 | 0.994 |  |
| `iqr` | 1.362962 | 1.000 | 1.000 | 1.000 | 1.000 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
