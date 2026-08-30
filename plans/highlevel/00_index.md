# Implementation Plan — Index

**Project:** SSM-based audio anomaly detection, deployed on MCUs.
**Authoritative context:** `00_master_file.md` — the *what and why* (committed claims, design rationale). These files are the *how and in what order*. The master doc wins on any conflict of intent; these files win on implementation detail.

---

> **Status note (this project, `ssm-mcu-asd-ind`):** this file and the rest of
> `plans/highlevel/` are inherited from `ssm-mcu-asd` largely as-is. Phase 1's architecture
> work (backbone, SSM block, training objective) is complete and carried over — see
> `01_design_decisions.md` §7 for the one thing that isn't carried over: **the data source.
> CNT is dropped; training/val/test are IND-only now.** Read §7 before trusting anything below
> that references CNT, silence filtering, or `fold_norm_stats.json` — those describe the prior
> project's path, kept here as record, not this project's current pipeline.

---

## File map

| file | contents | read when |
|---|---|---|
| `00_index.md` | this file — environment, session protocol, amendments to master, open items | every session |
| `01_design_decisions.md` | the training objective and why it is what it is | every session (short) |
| `01_eval_spec.md` | data split, balancing, metrics, threshold methods | every session (short) |
| `02_phase_1_gpu_prototype.md` | scaffold, dataset, log-mel, SSM core, training, head, diagnostics | Phase 1 |
| `03_phase_2_ablation_and_loso.md` | nested validation, ablation sweep, LOSO folds, run-count budget | Phase 2 |
| `04_phase_3_mcu_feature_pipeline.md` | CMSIS-DSP log-mel + stats, parity harness, per-board costs | Phase 3 (parallel) |
| `05_phase_4_backbone_port.md` | conditional skeleton — C port, fp32 first | Phase 4 |
| `06_phase_5_quantization_and_head.md` | conditional skeleton — the novel engineering result | Phase 5 |
| `07_phase_6_results_and_writeup.md` | conditional skeleton — three-way table, framing | Phase 6 |
| `99_glossary.md` | living glossary — **add to it as we go** | whenever a term is unfamiliar |

---

## Environment (confirmed)

- **Local NVIDIA GPU**, own machine. No queue, no timeout — wall-clock is the only budget.
- **ToyCar/ToyTrain on disk. No reusable code from the thesis** — the log-mel pipeline is written fresh, not ported.
- **Working mode: skeletons with named gaps.** Code blocks in these files are deliberately incomplete; gaps are marked `GAP n` with a question attached. The answers are derivable from master doc Sections 6–8.

---

## Session handoff protocol

Master doc Section 18 is explicit that later phases cannot be planned until earlier ones produce results. Concretely, for each new session bring:

1. `00_master_file.md` — noting Sections 6–8 are skippable background.
2. `00_index.md` + `01_design_decisions.md` — always.
3. **The one phase file** for the phase being worked.
4. **The previous phase's handoff manifest, materialised** — actual files, configs, numbers. Not the plan's description of them.

Point 4 is the one that gets skipped and shouldn't. A Phase 4 session holding Phase 2's winning config and `parity_vectors.npz` produces a specific roadmap; one working from "whatever the ablation determines" produces another plan.

Each phase file ends with its own **exit gate** and **handoff manifest**. Do not start the next phase until the gate passes.

---

## Detail policy

Phases 1–3 are specified concretely. Phases 4–6 are **conditional skeletons** — decision points, branches, and handoff manifests. Writing false precision into Phase 5 today would contradict the sequencing logic the master doc already commits to.

---

## Amendments these files make to the master doc

Propagate these back into `00_master_file.md` so the two do not drift.

1. **Section 10, Option 3** — "no extra training beyond the backbone itself" refers to the *head only*. The backbone is trained by autoregressive next-frame prediction. See `01_design_decisions.md`.
2. **Section 13** — bidirectionality moves from *deprioritized* to **forbidden**. It breaks the training objective's causality requirement, not merely MCU deployability.
3. **Section 13** — new ablation axis: **prediction horizon `k`**.
4. **Section 17 item 12** — downgraded from risk to ablation. The prediction head yields `S_recon` essentially free, so distance-only vs fused becomes a measurable comparison rather than an apology.
5. **Section 12** — the GPU autoencoder baseline moves to the end of the project (Phase 6).
6. **Data source** — CNT (`NormalSound_CNT`) is dropped entirely; every fold trains, validates,
   and tests on IND clips only. Not an amendment to any specific master doc section so much as
   a full replacement of the "which recordings" question underlying Phase 1 and Phase 2 — see
   `01_design_decisions.md` §7 for the contamination finding, the decisive IND-only experiment,
   and the tradeoffs accepted. `manifest.csv`, `configs/default.yaml`, and the fold function all
   need rebuilding for this project; none survive the port from `ssm-mcu-asd`.
7. **Run ID excludes the seed, deliberately.** Phase 1 §1.0 specified "run ID = hash of config + seed." `runs/compute_hash.py:train_config_hash()` hashes `held_out_case`, `training`, `features`, and `model` only. Seed is deliberately excluded: seed is not an ablation axis, and the project has settled on seed 158 as the single working seed. The three-seed sweep that resolved fold difficulty (`findings/130` §8) was archived by hand to `archive/SEED_{42,158,824}/` before each subsequent run. **If a future phase ever varies seed programmatically, add** `'seed': cfg['seed']` to train_config_hash first — without it, runs at different seeds collide on the same directory name and silently overwrite.
8. **Master doc Section 2 and Section 17 item 12 corrected**. The DCASE SSM paper's anomaly head fuses across encoder depth, not reconstruction-versus-latent-distance, and its training regime is two-stage supervised rather than self-supervised. The correction was made in the predecessor project and did not carry into this one; re-applied here. See master doc Section 2.

---

## Open items — consolidated

Continuing master doc Section 17's numbering.

| # | item | status |
|---|---|---|
| 9 | Parameter budget per target platform | open (master doc) |
| 10 | Quantization survival | open — the headline unknown (master doc) |
| 11 | Chip spec diff H7S3L8 vs STM32H747XIH6 | open (master doc) |
| 12 | Fusion-gap framing | **downgraded to an ablation** — see amendment 4 |
| 13 | Backbone training objective | **RESOLVED** — `01_design_decisions.md` |
| 14 | GPU autoencoder baseline | **RESOLVED: deferred to Phase 6.** Consequence recorded in `phase_1_gpu_prototype.md` §1.8 |
| 15 | Board availability | **open — needs your answer.** Affects Phase 3 ordering |
| 16 | Per-run wall-clock time | **RESOLVED**. ~80 s/epoch; 40 ± 10 epochs/fold on 3 cases (3,240 clips). ~53 min/fold typical, ~77 min worst observed (case3, 58 epochs). GTX 1660 Ti, no queue. Phase 2 arithmetic in `03_phase_2` §2.2 |
| 17 | Pooling choice | **RESOLVED**: **`mean` pooling**. Confirmed across three seeds and four folds (`findings/130` §2). `max` is unstable on case1 (std 0.104–0.134 across seeds); `concat_mean_last` produces ill-conditioned covariance in every fold at every seed and is disqualified for Mahalanobis. Distance head: `knn_clustered_16` for deployment (4 KB, mean AUC 0.9599), `knn_full` as accuracy upper bound (810 KB, 0.9770) |
| 18 | State utilisation | **RESOLVED, decisively**. Both diagnostics run on all four IND folds. Zeroing the recurrent state drives skill from ~+0.54 to ~−2.5, i.e. far worse than the persistence baseline — the model actively depends on the recurrence. Decay half-lives reach 0.3–15 s at every layer, well past the conv's 128 ms window. Master doc Section 1's state-dimension lever is real. Artifacts: `runs/case{N}/<hash>/{zero_state.csv, decay_half_life.csv, half_life_histogram.png}` |

---

## Immediate next action

**Phase 1 is complete**. All eight exit-gate items in 02_phase_1_gpu_prototype.md §1.10 pass, and the handoff manifest is materialised under runs/case{N}/<hash>/. Two Phase 1 items were deliberately deferred rather than dropped, and both are Phase 2 prerequisites:

- **`ranges.json`** — per-tensor activation min/max/percentiles, Phase 1 §1.5. Not yet implemented; train.py still has ranges = {} with a TODO. Phase 5 cannot diagnose quantization failures without it, and every checkpoint trained before it exists has no range record.
- **Threshold methods and secondary metrics** — src/eval/thresholds.py, per 01_eval_spec.md §6. In progress.

Phase 2's first action is 03_phase_2_ablation_and_loso.md §2.3 step 1: the selective-versus-fixed axis, since it reshapes the entire second half of the project.