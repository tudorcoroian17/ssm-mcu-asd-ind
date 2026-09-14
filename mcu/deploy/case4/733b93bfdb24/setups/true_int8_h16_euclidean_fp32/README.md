# Deployment setup: `true_int8_h16_euclidean_fp32`

Combo 15 of the case 4 deployment matrix.

## Matchup

- Config: `f4cd557b7e3b.yaml`
- Held-out case: 4
- Model hash: `733b93bfdb24`
- Backbone: true int8 arithmetic, h storage = int16
- Head: euclidean head (fp32 arithmetic)
- Threshold method: percentile (same-machine calibration)

## Files in this folder

Point one STM32CubeIDE source folder at this directory. The eight C files are self-contained; no file outside this folder is read.

| File | Role |
| :--- | :--- |
| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |
| `ssm_weights.h` / `.c` | This setup's quantized weights, scales, LUTs.
| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |
| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: euclidean only). |

The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- this folder can never be silently confused for the one scoring the other head off the same backbone.

## Active threshold (baked into `ssm_head_ref.c`)

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean` = **1.988313**.

At this threshold on the case 4 test split: P=0.993, R=1.000, A=0.996, F1=0.996.

Ranking quality (threshold-independent): AUC=1.0000, pAUC=1.0000.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean` constant in `ssm_head_ref.c` to any value from the table below, then rebuild. No re-export needed.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 1.903441 | 0.964 | 1.000 | 0.981 | 0.981 |  |
| `percentile_same_machine_99.0` | 1.988313 | 0.993 | 1.000 | 0.996 | 0.996 | **<- active** |
| `percentile_same_machine_99.5` | 2.028224 | 0.996 | 1.000 | 0.998 | 0.998 |  |
| `evt_p95_equiv` | 1.902309 | 0.964 | 1.000 | 0.981 | 0.981 |  |
| `parametric_p95_equiv` | 1.854278 | 0.927 | 1.000 | 0.960 | 0.962 |  |
| `kde_p95_equiv` | 1.909617 | 0.964 | 1.000 | 0.981 | 0.981 |  |
| `evt_p99_equiv` | 2.007492 | 0.993 | 1.000 | 0.996 | 0.996 |  |
| `parametric_p99_equiv` | 1.967798 | 0.989 | 1.000 | 0.994 | 0.994 |  |
| `kde_p99_equiv` | 2.007477 | 0.993 | 1.000 | 0.996 | 0.996 |  |
| `evt_p999_equiv` | 2.085976 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `parametric_p999_equiv` | 2.103324 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `kde_p999_equiv` | 2.099418 | 1.000 | 0.996 | 0.998 | 0.998 |  |
| `mad` | 2.043694 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `iqr` | 2.260936 | 1.000 | 0.962 | 0.981 | 0.981 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
