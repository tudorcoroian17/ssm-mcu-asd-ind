# Finding: Phase 2 ablation strategy changed from one-at-a-time nested to full-factorial single-fold screening

**Status:** Decision record. Supersedes `plans/highlevel/03_phase_2_ablation_and_loso.md`
§2.2–§2.4 and amends `00_master_file.md` §13 and §17 item 5. No experimental results here;
this file records what changed, why, and what was accepted as risk. Results go in a later
findings file.

**Context:** Written at the start of Phase 2, before any sweep run. Phase 1 is complete
(`00_index.md`). The plan as written called for a one-at-a-time ablation of 14 configurations
under nested validation, 60 runs after mitigation 1. That design is replaced.

---

## 1. What changed

| | plan as written | what will run |
| :--- | :--- | :--- |
| design | one-at-a-time from a default, 14 configs | full factorial, 72 configs |
| axes | `d_state`, `selective`, `discretization`, `d_conv`, `n_layers`, `horizon_k`, input representation | `d_state`, `n_layers`, `expand`, `selective`, `discretization` |
| validation | nested, single fixed inner split per outer fold | single-fold screening plus a three-fold confirmation set |
| folds per config | 4 outer, 1 inner each | 1 for screening, 4 for the confirmed configs only |
| run count | 60 | ~87 |
| selection basis | mean AUC over four inner folds | mean AUC on case 1, confirmed at three seeds |

The factorial is `d_state` {8, 16, 32} × `n_layers` {1, 2, 4} × `expand` {1, 2} × `selective`
{true, false} × `discretization` {zoh, euler} = 72 configurations. Config files were generated
for the full crossing with `horizon_k` {2, 5} and `target` {residual, absolute}, giving 288
files and a 1,152-row manifest at `configs/000_config_manifest.csv`. The sweep filters to
`horizon_k == 2` and a single `target`, leaving 72 configurations.

`target` is fixed by the residual-versus-absolute pilot (`01_design_decisions.md` §8.2), which
runs before the sweep. Until it resolves, the filter value is unset.

### 1.1 Axes dropped relative to the plan

- **`d_conv`** (master doc §13 secondary axis 5) is not in the factorial. Dropped for run count.
- **Input representation, log-mel versus per-frame statistics** (master doc §13 axis 4) is not
  an ablation. It requires a new feature extractor, a fresh cache under a new feature hash,
  `model.n_mels` dropping from 64 to 4, and therefore `GAP 3` in `backbone.py` implemented,
  since the code currently relies on `n_mels == d_model` and has no input embedding layer. It is
  a separate workstream.
- **`d_state = 64`** is not in the factorial. At batch 32 it materializes an `A_bar` of about
  360 MB with a matching `B_bar`, which does not fit comfortably in 6 GB alongside autograd.
- **`n_layers = 3`** is skipped; the levels are 1, 2, and 4.

### 1.2 Axis added relative to the plan

- **`expand`** {1, 2}. Not in master doc §13's list. It scales `d_inner = expand × d_model` and
  is therefore a direct footprint knob, in the same family as `d_state`. Record it as an
  addition when §13 is amended.

---

## 2. Why full factorial rather than one-at-a-time

Master doc §13 rejected full factorial on run-count grounds: "too many runs given 8+ candidate
axes × 4 LOSO folds × 2 machine types." Three of those three terms have since shrunk. The axis
list is five, not eight; `01_design_decisions.md` §6 deferred ToyTrain, so there is one machine
type; and §4 below reduces the fold count for screening. The original objection no longer holds
at the current scope.

The design is also better matched to the question. Master doc §1 asks which architectural lever
is *primary*. One-at-a-time gives one delta per lever, measured at a single point in
configuration space, and cannot detect interactions at all. A factorial estimates each main
effect by averaging over 24 or 36 configurations and makes interactions measurable. Whether
selectivity matters more at `d_state = 8` than at `d_state = 32` is directly load-bearing for
`05_phase_4_backbone_port.md`'s branch point, and only the factorial can answer it.

**Claude comment:** this is the one part of the change that is a straightforward upgrade. The
rest of the file is about what the fold reduction costs.

---

## 3. Why screening happens on case 1 alone

Three of the four folds are saturated and cannot rank configurations.

From `findings/130` §8, best of the 12 pooling × head combinations per fold, three seeds:

| case | mean ± std |
| :--- | :--- |
| 1 | 0.9600 ± 0.0132 |
| 2 | 0.9962 ± 0.0000 |
| 3 | 0.9903 ± 0.0015 |
| 4 | 0.9993 ± 0.0010 |

`findings/130` §9 explains the mechanism: cases 2, 3, and 4 achieve perfect separation apart
from one, three, and two specific reproducible clips respectively. Their AUC is quantized in
units of 1/265 = 0.0038 and has almost no dynamic range left. Case 1 fails differently (§9.3):
essentially no buried clips, but 211 to 257 anomalies diffusely overlapping at least one normal,
so its AUC varies continuously with model quality.

Two consequences follow.

**Case 1 is the only fold with the dynamic range to separate 72 configurations.** Screening on
case 4 would have returned a near-tie across most of the design space.

**A four-fold mean would have been a shrunken version of the case 1 ranking.** With three of four
terms pinned near a ceiling at near-zero variance, `mean(case1..case4)` is approximately
`0.25 × case1 + constant`. Training all 288 runs would cost four times the compute to produce a
ranking close to monotone in the one case 1 already gives. What it would buy is robustness at
the bottom of the ranking, where a configuration bad enough to break saturation on cases 2 to 4
would finally register. That region is not decision-relevant.

---

## 4. The procedure

| stage | runs | purpose |
| :--- | :--- | :--- |
| 1. Screen all 72 configurations on case 1, seed 158 | 68 | case 1 at the default config already exists |
| 2. Re-run the top 5 on case 1 at seeds 42 and 824 | 10 | test whether the case 1 ranking is a seed artifact |
| 3. Train 3 selected configurations on cases 2, 3, and 4 | 9 | confirmation set; see §5 |
| **total** | **~87** | roughly 3.5 to 4 days |

Stage 2 exists because the maximum of 72 single-seed estimates is upward-biased, and case 1
carries roughly ten times the seed variance of the other folds (`findings/130` §8). Ten runs
place replication exactly where the bias bites.

Stage 3 trains three configurations, not one:

1. **The stage 2 winner.** Master doc §1's `[X]%` figure is the deployed configuration's LOSO
   accuracy, and `01_eval_spec.md` §7 requires a per-fold table. A single fold is not reportable.
2. **The best `selective = false` configuration.** `05_phase_4_backbone_port.md` opens with a
   branch table: if fixed matches selective, Phase 4 ports a static recurrence, master doc §11
   point 1 evaporates, and Phase 5 becomes conventional quantization. If selective wins, the
   MambaLite-Micro fusion trick becomes load-bearing and Phase 5 is materially harder. That is a
   weeks-of-C-work decision and must not rest on one fold of one machine.
3. **The smallest configuration within seed noise of the winner.** The deployment candidate. If
   it is the same configuration as (1) or (2), stage 3 shrinks accordingly.

The Phase 1 default is already trained on all four folds and comes along as a free fourth
reference point.

### 4.1 Run order

Stage 1 runs in a fixed random permutation with a logged seed, so that any prefix of an
interrupted sweep is an unbiased subset of the design rather than a subset confounded with
factor levels.

### 4.2 Footprint precheck, before any training

`estimate_streaming_ram()` needs only the config dictionary, and the parameter count needs a
model instance but not a trained one. The full accuracy-versus-footprint axis for all 72
configurations is computable in about a minute of CPU, and should be computed before the GPU
is committed. Estimated span: roughly 13k to 160k parameters (13 KB to 156 KB int8) against 2 MB
of flash on the leanest board, and roughly 11 KB to 137 KB of peak streaming RAM in fp32 against
the RP2040's 264 KB.

If that estimate holds, the entire design space fits every target in fp32, the RP2040's binding
constraint is cycles on an FPU-less M0+ rather than memory, and the sweep's purpose narrows from
"find what fits" to "measure what `selective = false` costs." Record the measured table before
starting stage 1; it may justify a further reduction in scope.

---

## 5. Selection rule, pre-registered

Recorded before any sweep result exists.

- **Ranking metric:** AUC under `mean` pooling with `knn_clustered_16`. `mean` pooling per
  `00_index.md` open item 17; `max` is excluded because `findings/130` §2.3 shows it is unstable
  on case 1 specifically, which is the screening fold. `knn_clustered_16` rather than `knn_full`
  because it is the deployment head (4 KB versus 810 KB, `findings/130` §11), and the
  configuration being selected is one to deploy. Report `knn_full` alongside as the accuracy
  ceiling; do not select on it.
- **Stage 1 to stage 2:** take the top 5 configurations by case 1 AUC at seed 158.
- **Stage 2 to stage 3:** take the configuration with the highest mean AUC across seeds 42, 158,
  and 824 on case 1, plus the two additional configurations named in §4.
- **Ties:** break on footprint, smallest first.

---

## 6. What this gives up, stated plainly

### 6.1 Nested validation is not being run

Master doc §17 item 5 records nested validation as RESOLVED and adopted, and
`03_phase_2_ablation_and_loso.md` §2.2 says explicitly not to mitigate by dropping it. This
change drops it. The replacement is not nested validation and should not be called that. It is
single-fold screening followed by a confirmation set.

The justification is the priority ordering already committed in master doc §1: deployment
feasibility is the primary spine, cross-machine generalization is a supporting claim. Nested
validation protects the supporting claim at a cost of roughly nine additional days before the
MCU port begins. That trade is refused.

### 6.2 Case 1's reported number is optimistic

Case 1 selects the configuration, so case 1's AUC is contaminated by selection.

**Do not fix this by reporting the mean of cases 2, 3, and 4 only.** Those are the saturated
folds, and excluding the hardest fold would inflate the headline in the other direction.
`01_eval_spec.md` §7 already requires per-fold reporting rather than a bare mean, so the honest
form is nearly free: report all four folds, label case 1 as the development fold, and state that
its number is optimistic while the other three are selection-clean.

### 6.3 Fold cannot be used as a blocking factor

`findings/130` §7 and `01_eval_spec.md` §5 item 5 both establish that folds differ systematically
in difficulty and in normalization scale. With one fold in stage 1, that variation cannot be
modeled, so main-effect estimates describe case 1 and not ToyCar in general. State this when
reporting the factorial: the effect sizes are measured on one machine.

### 6.4 The screening fold is atypical

This is the substantive risk, not a bookkeeping one. Case 1's errors are diffuse where cases 2
to 4's are concentrated in a handful of fault codes (`ab22`, `ab49`, `ab08`, `ab33`,
`findings/130` §9.2). Case 1 also has its own pathology: §7.4 shows loudness latching drives its
pAUC specifically, at two of three seeds. A configuration could therefore win on case 1 for a
reason that does not exist on the other three machines, whether by suiting diffuse overlap or by
being incidentally loudness-insensitive.

Stage 3 is the check for exactly this. If the stage 2 winner also leads on cases 2, 3, and 4,
the risk did not materialize. If it does not, that disagreement is itself a result and should be
written up rather than resolved by picking whichever fold agrees with the preferred conclusion.

**Claude comment:** the failure this guards against is concrete. Configuration #57 beats the
default by 0.03 on case 1, gets ported, and turns out in Phase 6 to score 0.985 on the other
three folds where the default scored 0.996. Nine runs catch that. By Phase 6 the C engine is
already written.

---

## 7. Documents to amend

1. `00_master_file.md` §13 — "Method: one-at-a-time from a single baseline/config — NOT full
   factorial" is no longer accurate. Add `expand` to the axis list; mark `d_conv` and the input
   representation axis as deferred.
2. `00_master_file.md` §17 item 5 — nested validation is no longer the scheme in use. Point at
   this file.
3. `03_phase_2_ablation_and_loso.md` §2.2 — the run-count table, the nested pseudocode, and the
   mitigation ladder all describe an experiment that is not running.
4. `03_phase_2_ablation_and_loso.md` §2.3 — the execution order is superseded. The factorial
   covers all axes simultaneously; the selective-versus-fixed answer arrives from the stage 1
   marginal means rather than from a sequenced first axis.
5. `03_phase_2_ablation_and_loso.md` §2.4 — reporting changes from per-axis delta tables to
   factorial main effects plus a per-fold table for the confirmed configurations.
6. `03_phase_2_ablation_and_loso.md` exit gate item 3 — "a winning config identified through the
   nested procedure" is not achievable as written. Replace with the §5 rule.
7. `00_index.md` — add an amendment entry pointing here, in the style of amendments 1 through 8.

---

## 8. Open items

- The residual-versus-absolute pilot has not run. Its result sets the `target` filter for the
  whole sweep. Blocking.
- The §4.2 footprint table has not been computed. It may justify reducing the design below 72
  configurations before stage 1 starts.
- Whether the stage 2 top-5 threshold is the right cut is untested. If the top 20 on case 1 fall
  within one seed standard deviation (0.0132), widen it or accept that the design space is flat
  and report that as the finding.
- `03_phase_2_ablation_and_loso.md`'s handoff manifest item 4, `parity_vectors.npz`, is unchanged
  by this decision and still gates Phase 4.