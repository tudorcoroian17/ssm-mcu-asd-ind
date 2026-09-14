# Deployment setup: `full_fp32_euclidean_fp32`

Combo 27 of the case 3 deployment matrix.

## Matchup

- Config: `f2578cb06991.yaml`
- Held-out case: 3
- Model hash: `5295aa5e5f6a`
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **2.391726**.

At this threshold on the case 3 test split: P=0.986, R=0.536, A=0.764, F1=0.694.

Ranking quality (threshold-independent): AUC=0.9050, pAUC=0.7889.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.216689 | 0.937 | 0.619 | 0.789 | 0.745 |  |
| `percentile_same_machine_99.0` | 2.391726 | 0.986 | 0.536 | 0.764 | 0.694 | **<- active** |
| `percentile_same_machine_99.5` | 8.458533 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `evt_p95_equiv` | 2.198008 | 0.933 | 0.626 | 0.791 | 0.749 |  |
| `parametric_p95_equiv` | 2.508594 | 0.984 | 0.479 | 0.736 | 0.645 |  |
| `kde_p95_equiv` | 2.302843 | 0.975 | 0.581 | 0.783 | 0.728 |  |
| `evt_p99_equiv` | 2.774437 | 0.981 | 0.389 | 0.691 | 0.557 |  |
| `parametric_p99_equiv` | 2.963581 | 0.974 | 0.287 | 0.640 | 0.443 |  |
| `kde_p99_equiv` | 2.604318 | 0.984 | 0.453 | 0.723 | 0.620 |  |
| `evt_p999_equiv` | 7.970508 | 0.714 | 0.019 | 0.506 | 0.037 |  |
| `parametric_p999_equiv` | 3.572353 | 0.962 | 0.189 | 0.591 | 0.315 |  |
| `kde_p999_equiv` | 10.411563 | 0.000 | 0.000 | 0.500 | 0.000 |  |
| `mad` | 2.967040 | 0.974 | 0.283 | 0.638 | 0.439 |  |
| `iqr` | 3.385720 | 0.968 | 0.226 | 0.609 | 0.367 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
## Recurrence form: full fp32 (no quantization)

This is the classic (`selective=False`) matrix's unquantized baseline. It uses the same runtime discretize step as `qab` -- `delta = softplus(dt)`, `A = -exp(A_log)`, `A_bar = 1 + clamp(delta*A, -1.9)`, `B_bar = delta*B`, computed every frame -- except nothing is quantized: `A_log`, `dt` and `B` ship as plain fp32.

Tensors in this folder's `ssm_weights.c`: `in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights.

`ssm_backbone.c` is byte-for-byte identical to every other setup in this matrix: the difference is entirely in `ssm_weights.c`'s values, which are fp32 here instead of int8.
