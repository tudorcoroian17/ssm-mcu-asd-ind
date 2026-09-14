# Deployment setup: `true_int8_h8_euclidean_int8`

Combo 12 of the case 2 deployment matrix.

## Matchup

- Config: `b39731b66741.yaml`
- Held-out case: 2
- Model hash: `a501188e9de1`
- Backbone: true int8 arithmetic, h storage = int8
- Head: euclidean head (int32 arithmetic)
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

Default method: **`percentile_same_machine_99.0`**, baked in as `ssm_threshold_euclidean_sumsq` = **2.316669** (float form). For the int8 head the decision uses the int32 `ssm_threshold_euclidean_sumsq` constant.

At this threshold on the case 2 test split: P=0.959, R=0.351, A=0.668, F1=0.514.

Ranking quality (threshold-independent): AUC=0.6672, pAUC=0.7022.

## Swapping the threshold by hand

Edit the `ssm_threshold_euclidean_sumsq` constant in `ssm_head_ref.c`. The int8 head decides on the int32 sum-of-squares form; the float score is reported but not used for the decision. To convert a float threshold `t` (from the table below) into the int32 form: `round((t / 0.0305022799) ** 2)`. Every method's int32 form is precomputed in the table.

## All 14 threshold methods

Every method below was fit on this setup's own `calib_normal` scores (same held-out machine, disjoint from train and from the balanced test split), then evaluated on the test split. Numbers match `diagnostics.json` in this folder exactly. The active default is marked.

| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `percentile_same_machine_95.0` | 2.173043 | 5075 | 0.917 | 0.460 | 0.709 | 0.613 |  |
| `percentile_same_machine_99.0` | 2.316669 | 5769 | 0.959 | 0.351 | 0.668 | 0.514 | **<- active** |
| `percentile_same_machine_99.5` | 2.367861 | 6026 | 0.987 | 0.287 | 0.642 | 0.444 |  |
| `evt_p95_equiv` | 2.173325 | 5077 | 0.917 | 0.460 | 0.709 | 0.613 |  |
| `parametric_p95_equiv` | 2.184161 | 5127 | 0.922 | 0.449 | 0.706 | 0.604 |  |
| `kde_p95_equiv` | 2.182245 | 5118 | 0.923 | 0.453 | 0.708 | 0.608 |  |
| `evt_p99_equiv` | 2.310419 | 5737 | 0.959 | 0.351 | 0.668 | 0.514 |  |
| `parametric_p99_equiv` | 2.291061 | 5642 | 0.960 | 0.362 | 0.674 | 0.526 |  |
| `kde_p99_equiv` | 2.331949 | 5845 | 0.966 | 0.317 | 0.653 | 0.477 |  |
| `evt_p999_equiv` | 2.514435 | 6795 | 0.979 | 0.177 | 0.587 | 0.300 |  |
| `parametric_p999_equiv` | 2.412666 | 6256 | 0.985 | 0.242 | 0.619 | 0.388 |  |
| `kde_p999_equiv` | 2.535439 | 6909 | 0.977 | 0.158 | 0.577 | 0.273 |  |
| `mad` | 2.424933 | 6320 | 0.984 | 0.230 | 0.613 | 0.373 |  |
| `iqr` | 2.588393 | 7201 | 0.968 | 0.113 | 0.555 | 0.203 |  |

Some methods (notably `evt_*` at aggressive false-alarm targets) can collapse on certain score distributions -- see `findings/543`. A high threshold with near-zero recall in the table above is that collapse, not a backbone problem; the AUC is unaffected.
