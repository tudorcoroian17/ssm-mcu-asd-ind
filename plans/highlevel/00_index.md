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
| `phase_1_gpu_prototype.md` | scaffold, dataset, log-mel, SSM core, training, head, diagnostics | Phase 1 |
| `phase_2_ablation_and_loso.md` | nested validation, ablation sweep, LOSO folds, run-count budget | Phase 2 |
| `phase_3_mcu_feature_pipeline.md` | CMSIS-DSP log-mel + stats, parity harness, per-board costs | Phase 3 (parallel) |
| `phase_4_backbone_port.md` | conditional skeleton — C port, fp32 first | Phase 4 |
| `phase_5_quantization_and_head.md` | conditional skeleton — the novel engineering result | Phase 5 |
| `phase_6_results_and_writeup.md` | conditional skeleton — three-way table, framing | Phase 6 |
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
| 16 | Per-run wall-clock time | open; resolves at Phase 1 exit gate. Decides whether Phase 2's nested design is affordable |
| 17 | Pooling choice | provisionally mean pooling (Euclidean, AUC=0.9257 on IND-only held-out-1) — needs a Mahalanobis re-run on IND before treating as settled; see 01_design_decisions.md §7.3 |
| 18 | State utilisation | open; if all three diagnostics say the state is inert, the master doc's Section 1 headline claim needs rethinking *before* the sweep runs |

---

## Immediate next action

**Rebuild the data path for IND-only.** Nothing here needs the GPU. In order: (1) build
`manifest.csv` for this project scoped to IND clips only (cross-check `ssm-mcu-asd`'s
manifest-building script for the filename-parsing logic, still valid), (2) port
`get_fold_ind_only()` and `compute_normalization_stats_ind()` from the prior project's
exploration — drafted, not yet transcribed anywhere — as *the* fold/stats functions here, no
"ind_only" naming needed since there's no CNT variant to distinguish from, (3) recompute
`mse_persistence`/`mse_climatology` fresh on this project's IND pool before trusting any
inherited number — the 0.071 ratio in §7.3 came from a single held-out-1 run in the old project
and should be treated as a prior, not a given.