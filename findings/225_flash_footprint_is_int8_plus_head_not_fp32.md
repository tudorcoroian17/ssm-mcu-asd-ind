# Flash-footprint figures in findings/220 are int8-plus-head, not fp32

**Feeds:** `03_phase_2_ablation_and_loso.md` (footprint methodology); `findings/220` (every
flash-column figure); `05_phase_4_backbone_port.md` step 3 (the fp32 budget for the C port).

## Result

`findings/220`'s cited flash-footprint figures (36.06, 55.97, 79.94 KB, and by extension every
other value in its tables) are not fp32 backbone weight sizes. Cross-checking three of them
against `runs/phase2/footprints.csv` shows each equals that config's `flash_footprint_int8_kb`
column plus a flat 4.00 KB, rounded to two decimals:

| Config | `findings/220` figure | csv `fp32_kb` | csv `int8_kb` | cited minus int8 |
| :--- | ---: | ---: | ---: | ---: |
| `f4cd557b7e3b` | 36.06 | 128.25 | 32.0625 | 4.0025 |
| `352f70960ed3` | 55.97 | 207.875 | 51.96875 | 4.00125 |
| `78ca5922c9bf` | 79.94 | 303.75 | 75.9375 | 3.9975 |

The 4.00 KB constant is `16 x 64 x 4` bytes: the `knn_clustered_16` reference set (16 centroids,
`d_model=64`, fp32), the "4 KB head" master doc section 18's phase 1 summary already names.

## Verification

Hand-derived parameter count for `f4cd557b7e3b` / case 1 (`d_model=64, d_state=16, expand=1,
d_conv=4, n_layers=2`, selective, `dt_rank=4`):

| Module | Parameters |
| :--- | ---: |
| `in_proj` | 8,192 |
| `conv` (depthwise, with bias) | 320 |
| `x_proj` | 2,304 |
| `dt_proj` (with bias) | 320 |
| `A_log` | 1,024 |
| `D` | 64 |
| `out_proj` | 4,096 |
| Per-block total x 2 layers | 32,640 |
| 3x RMSNorm (2 blocks + final) | 192 |
| **Total** | **32,832** |

`32,832 / 1024 = 32.0625` matches the csv's int8 column exactly (1 byte/parameter). At 4
bytes/parameter, that's `128.25` KB fp32. `src/eval/footprint.py` computes this correctly; the
number that reached `findings/220`'s prose is a different, derived quantity.

## Consequence

`findings/220`'s accuracy/footprint trade-off, including the Pareto frontier in section 8, plots
int8-scale flash (plus a fixed fp32 head cost) against AUC measured at fp32. That comparison is
only fair if int8 quantization of the backbone is lossless, which is precisely
`00_index.md` open item 10, still unresolved. Whether this was deliberate (report the
deployment-target footprint next to the accuracy ceiling) or a mix-up during write-up isn't
something the numbers alone can settle.

For Phase 4, which is fp32-first by design, the correct backbone flash budget for the config
being ported (`f4cd557b7e3b`, case 1, `runs/case1/16662b29beb3/`) is **128.25 KB**, not 36.06 KB.

**Claude comment:** I flagged this rather than quietly using whichever number seemed more
convenient, because the two numbers lead to different conclusions in different places -- 128.25
KB is still comfortably inside the "use external XSPI flash" plan from `findings/310` and
`findings/400`, so nothing about feasibility changes. But if the Pareto-frontier figure in
`findings/220` §8 is meant for the paper as-is, it's worth deciding now whether it needs a
same-precision recomputation, rather than after the Phase 4/5 numbers exist and the two curves
have to be reconciled retroactively.