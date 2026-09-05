# Finding: full 72-config sweep results, and the four-config refit for the selectivity trade-off

**Status:** Partially resolved. The case-1 sweep is complete and analyzed (§2–§6). The
four-config refit on cases 2–4 is specified and queued but not yet run; its numbers are the
open item in §7 and gate the final claim. Supersedes the single-fold selection language in
`findings/210_change_ablation_strategy.md` §5 and reframes stage 3 from "pick a deployment
winner" to "characterize the selective/classic trade-off," per §1.

**Context:** The 72-config full-factorial sweep from `findings/210` completed on case 1, all runs
`status: ok`. This document records what the sweep found, why the deployment framing was dropped,
and which four configurations were selected for the cross-fold refit. Analysis reads only the
collected tables under `runs/phase2/` (`configs.csv`, `footprints.csv`, `diagnostics.csv`,
`scores.csv`), built by `runs/collect_results.py`.

**Claude comment:** the selection story in this document changed direction twice as the numbers
came in, and the sections where that happened keep the superseded reasoning visible rather than
presenting only the final choice. The record of why the obvious pick was wrong is the more useful
half, same convention as `findings/130`.

---

## 1. What this phase is for, and what changed

`findings/210` framed stage 3 as identifying a single deployment winner and confirming it across
folds. That framing was dropped. The thesis claim in `00_master_file.md` §1 is a trade-off
between AUC and memory, not a single recommended model, so stage 3 was refocused to characterize
how selectivity trades against footprint at two representative points on the accuracy/memory
frontier.

This is a deliberate scope change, recorded here rather than left implicit:

- **From:** train the sweep winner, the best fixed config, and the smallest-within-noise config
  on cases 2–4; report a deployment recommendation.
- **To:** train the selective and classic variants of two representative shapes on cases 2–4;
  report the selectivity trade-off and its fold dependence.

The reason is scope discipline. Deployment feasibility remains the primary spine
(`00_master_file.md` §1), but a single argmax recommendation was never the deliverable, and
selecting one invited a seed-confirmation step (`findings/210` §5 stage 2) whose cost was not
justified by the trade-off figure the thesis actually needs. §6 records the decision to skip that
step and what was substituted for it.

---

## 2. Sweep ranking, and why case 1 carries it alone

All analysis uses `mean` pooling with `knn_clustered_16`, the ranking metric pre-registered in
`findings/210` §5. Case 1 is the only discriminative fold: `findings/130` §8 measured cases 2–4
at 0.99 and above with near-zero seed variance, and §9 showed their errors are a handful of named
clips rather than a continuous quality signal. Case 1 alone has the dynamic range to rank 72
configurations.

### 2.1 Best configurations by branch

Top three per branch, case 1:

| branch | config | d_state | n_layers | expand | discretization | AUC | flash (KB) |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| selective | `78ca5922c9bf` | 32 | 2 | 2 | euler | 0.9343 | 79.94 |
| selective | `f19d5ca15add` | 16 | 2 | 2 | euler | 0.9256 | 67.94 |
| selective | `522ea9b11b24` | 16 | 2 | 1 | zoh | 0.9181 | 36.06 |
| classic | `352f70960ed3` | 8 | 2 | 2 | euler | 0.9275 | 55.97 |
| classic | `3b440664aeb5` | 16 | 2 | 2 | zoh | 0.9230 | 58.00 |
| classic | `61a995157d85` | 8 | 4 | 1 | zoh | 0.9218 | 56.12 |

Every one of these six uses `n_layers=2`. Depth converged on a single level without being
constrained to; §4 covers what depth does elsewhere in the design.

### 2.2 The headline result: selectivity does not buy accuracy at the top of the design

Best selective (`78ca5922c9bf`, 0.9343) against best classic (`352f70960ed3`, 0.9275):

- **AUC gap: 0.0068**, roughly half the case-1 seed spread of 0.0132 (`findings/210`, derived
  from the three-seed default-model measurement in `findings/130` §8). Not distinguishable from
  seed noise.
- **Flash: 55.97 KB against 79.94 KB**, a 30% reduction for the classic config.

The best classic configuration matches the best selective configuration in accuracy, within seed
noise, at 24 KB less flash and with no runtime `softplus`, `expm1`, or per-timestep projection.
This is the primary sweep finding and it points the `05_phase_4_backbone_port.md` branch toward
the simpler, static-recurrence path.

**Claude comment:** this is a stronger and less expected result than "selectivity helps" or
"selectivity is neutral." It says the two knobs interact: selectivity's value depends on where
in the design space it is applied, which §5 makes precise.

---

## 3. Knob trade-offs, from the forest analysis

`runs/knob_cost.py` fits one OLS per (pooling, head) over the 72 case-1 configs, with
adjacent-step contrasts read through `t_test` for correct confidence intervals, at
`mean + knn_clustered_16`. Signs are oriented so a positive value means the cheaper or simpler
option scores higher.

The deployment-relevant reading is cost against footprint freed:

| step | AUC change | flash freed (KB) | reading |
| :--- | ---: | ---: | :--- |
| `expand` 2 → 1 | about −0.01 | about 35 | near-free saving |
| `d_state` 32 → 8 | about −0.04 | about 7 | poor trade |

`expand` frees roughly five times the memory of `d_state` for about a quarter of the AUC cost.
The mechanism: `expand` scales `d_inner = expand × d_model`, which multiplies both the depthwise
convolution and the state matrices, so reducing it touches more parameters than reducing
`d_state` alone. Reducing `d_state` shrinks only the state dimension while leaving the projections
at full width.

**`expand` is the knob to reach for first when footprint must come down.** This holds across the
pooling and head cells, not only at `mean + knn_clustered_16`; the sign is stable, only the
magnitude moves.

---

## 4. Depth is the largest footprint lever and the one real accuracy cost

`n_layers` was split into adjacent steps (`4 → 2` and `2 → 1`) rather than measured against the
cheapest level, so each row is one design decision. The two steps are not equal:

- **`2 → 1`** carries the larger accuracy penalty. Removing the last layer is expensive.
- **`4 → 2`** costs less accuracy while freeing more flash. The first layer removed is close to
  free.

This is why every strong configuration in §2.1 sits at `n_layers=2`: it is the point where the
cheap depth reduction has been taken and the expensive one has not. Reading depth as a single
`4 → 1` penalty would have hidden that the cost is concentrated in the final step.

---

## 5. Selectivity is an interaction, not a main effect

The paired-delta analysis in the sweep measured selectivity's effect across all 36 exact
one-flip pairs at a spread of −0.32 to +0.21, with a near-zero median. Selectivity has no usable
main effect; its sign depends on the other knobs. Four named configurations make the interaction
concrete.

### 5.1 A crossover, not diminishing returns

Two representative shapes, each in both branches, case 1:

| shape | branch | config | AUC | flash (KB) | selectivity effect |
| :--- | :--- | :--- | ---: | ---: | ---: |
| A: d_state=16, expand=1 | classic | `f2578cb06991` | 0.8425 | 31.13 | |
| A: d_state=16, expand=1 | selective | `f4cd557b7e3b` | 0.9106 | 36.06 | **+0.068** |
| B: d_state=8, expand=2 | classic | `352f70960ed3` | 0.9275 | 55.97 | |
| B: d_state=8, expand=2 | selective | `b39731b66741` | 0.8191 | 61.94 | **−0.108** |

At shape A, selectivity is a large win for a small memory cost. At shape B, selectivity is
strictly worse on both axes: it costs 0.108 AUC **and** adds 5.97 KB. The classic branch
dominates the selective branch completely at shape B.

This is a sign flip, not a narrowing. Selectivity helps in one architecture and hurts in another,
with no monotonic story connecting them. The −0.108 value sits well inside the −0.32 to +0.21
paired-delta range, so this is a named instance of the aggregate interaction, not an outlier.

### 5.2 A hypothesis for the mechanism

At `d_state=8` the state is the tight bottleneck. Selectivity spends parameters on the
data-dependent `x_proj` and `dt_proj` gating rather than on state capacity. At eight states, that
trade may cost more representational capacity than the adaptivity recovers; at 16 states there is
room for both.

This is a hypothesis, not a demonstrated cause. Isolating it cleanly would need a `d_state ×
selective` sweep at fixed `expand`, which the sweep provides only partially. Recorded as open in
§9.

### 5.3 A third data point, near zero

`1df2b3ea9ccc` (classic) and `522ea9b11b24` (selective) share a shape (d_state=16, n_layers=2,
expand=1, zoh) and differ by 0.0065 in AUC, inside a third of the seed spread. This sits almost
exactly at the neutral point, between shape A's strong-positive and shape B's strong-negative,
and completes the range: selectivity's effect at this operating point is neither, it is nothing.

---

## 6. Stage 2 seed reruns were skipped

`findings/210` §5 named a stage-2 step: rerun the top five configurations at seeds 42 and 824 to
test whether the case-1 ranking is a seed artifact. That step was skipped.

- **Reason:** the time cost of even a reduced seed sweep was judged not worth it against the
  downstream refit cost, given that the deliverable is a trade-off figure rather than a single
  argmax that seed confirmation would protect.
- **Substituted:** the case-1 seed spread of 0.0132 (`findings/210`, from the default config) is
  used as the reference noise band for every AUC comparison in this document. This assumes seed
  variance does not scale substantially with architecture size, which is unverified for these
  specific shapes.
- **Partial protection already present:** the selected configurations were derived from averaged
  paired-delta effects, each a mean over 24 to 36 pairs, rather than from a single-config argmax.
  The variance of a mean over 24 pairs is far below that of one raw config's AUC, so the knob-level
  choices already carry much of what stage 2 was meant to protect against.
- **Residual risk not covered:** whether the specific four-config combination sits in an
  unusually favorable or unfavorable corner of the selectivity interaction. §5 shows the
  interaction is large, so this risk is real and is exactly what the cross-fold refit in §7
  tests.

**Claude comment:** the 0.0132 band is doing real work across this document and it was measured on
one architecture. Treating it as a worst-case ceiling for different architectures is the weakest
assumption in the analysis. It is stated wherever it is used rather than buried.

---

## 7. The four-config refit: specification and open numbers

The four configurations from §5.1 are trained on cases 2, 3, and 4, giving four folds per
configuration. Case 1 already exists from the sweep, so this is 4 × 3 = 12 runs.

| config | shape | branch | case-1 AUC | flash (KB) |
| :--- | :--- | :--- | ---: | ---: |
| `f4cd557b7e3b` | A: d_state=16, expand=1, euler | selective | 0.9106 | 36.06 |
| `f2578cb06991` | A: d_state=16, expand=1, euler | classic | 0.8425 | 31.13 |
| `352f70960ed3` | B: d_state=8, expand=2, euler | classic | 0.9275 | 55.97 |
| `b39731b66741` | B: d_state=8, expand=2, euler | selective | 0.8191 | 61.94 |

### 7.1 What the refit is testing, specifically

The question is not the generic "does case 1 generalize." It is sharper: **does the selectivity
sign flip in §5.1 survive on cases 2–4, or is it a case-1 idiosyncrasy?** `findings/130` §9
established that cases 2–4 fail through a few concentrated fault codes rather than the diffuse
overlap case 1 shows, so the crossover may soften or vanish there. If it does, "the effect is
fold-dependent" is itself the result, with precedent in the reference-set finding of `findings/130`
§13.

### 7.2 Results

**Open.** To be filled once the 12 runs complete. Per-fold AUC and flash for all four
configurations, with case 1 labeled as the development fold whose numbers are optimistic per
`findings/210` §6.2, and cases 2–4 as the selection-clean folds.

---

## 8. Pareto frontier membership

Frontier computed at `mean + knn_clustered_16`, case 1, over all 72 configurations, minimizing
flash and maximizing AUC. The full frontier is eight configurations.

Of the four refit configurations, **only `352f70960ed3` is on the frontier.** The other three are
dominated:

- `f4cd557b7e3b` (36.06 KB, 0.9106) is dominated by `1df2b3ea9ccc` (31.13 KB, 0.9116), which is
  cheaper and higher. Its own same-shape twin `522ea9b11b24` beats it at identical flash on the
  discretization choice alone (`zoh` over `euler`, +0.0075 at zero cost).
- `f2578cb06991` (shape A classic) and `b39731b66741` (shape B selective) are the low side of
  each pair and are not frontier candidates by design.

**This constrains the figure caption.** The four-config plot is not "two points on the Pareto
frontier." It is one frontier point (`352f70960ed3`) and one illustrative pair
(`f4cd557b7e3b` against `f2578cb06991`) showing selectivity's effect at a non-frontier shape. The
distance from that pair to the frontier is itself informative: even after selectivity's full
+0.068 gain, shape A still sits below the curve that shape B's classic variant lies on.

**Claude comment:** the honest framing here is stronger than "both optimal" would have been. It
shows selectivity closing part of a gap without being the most efficient way to close it, which is
exactly the nuance the branch discussion in `05_phase_4_backbone_port.md` needs.

---

## 9. Parity artifacts for the refit configurations

The Phase 4 hand-off (`05_phase_4_backbone_port.md` step 4) requires a `parity_vectors.npz`
golden reference for each deployed configuration, generated before `_scan()` and `discretize()`
are refactored for streaming.

The default config's parity artifacts already exist and were validated three ways (`findings/210`
§8): byte-identical across two runs, `pooled_mean` matching the time-mean of `final_norm_output`
to 1.19e-7, and matching the independently produced `emb_mean.npz` to 2.38e-7. That was the
machinery dry run.

**Open.** The refit configurations need their own parity artifacts generated after the §7 runs
complete and before any `ssm_block.py` restructuring. The `selective=False` configurations are
the ones that matter most for Phase 4, since their `A_bar` and `B_bar` are constant across the
clip (batch- and time-invariant), which is the property the static-recurrence C port exploits.

---

## 10. Open items

- **The §7.2 cross-fold numbers.** Blocking the final trade-off claim and the §7.1 sign-flip
  question.
- **The §9 parity artifacts** for the four refit configurations. Blocking Phase 4.
- **Whether the §5.2 mechanism is correct** — that selectivity's cost at low `d_state` is a
  capacity trade. Needs a `d_state × selective` sweep at fixed `expand` to isolate; the current
  sweep provides only partial coverage.
- **Documents flagged in `findings/210` §7** remain unamended: `00_master_file.md` §13 and §17
  item 5, `03_phase_2_ablation_and_loso.md` §2.2–§2.4 and exit gate item 3, and `00_index.md`.
  Bookkeeping, not blocking, but outstanding since `findings/210`.
- **Whether the six broken `checks/` scripts** from `findings/130` §1 were repaired. The
  winner-tier check suite in the Phase 2 exit depends on them.

---

## 11. Decision

**Proceed with the four-config refit on cases 2–4, then generate parity artifacts, then refactor
`ssm_block.py` for streaming.** The case-1 sweep supports a clear provisional claim — selectivity
does not improve accuracy at the top of the design and lets footprint fall further instead, with
its effect swinging by architecture — but that claim rests on one fold until §7.2 lands. The refit
is the check, and it is cheap: 12 runs against the alternative of a wrong cross-fold story
propagating into the C port.