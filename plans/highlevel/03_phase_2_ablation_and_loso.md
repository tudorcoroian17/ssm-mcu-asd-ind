# Phase 2 — Ablation + LOSO Harness

**Prerequisites:** Phase 1 handoff manifest, materialised. In particular the **measured per-run wall-clock time** — §2.2's run-count arithmetic is undecidable without it.
**Goal:** resolve the master doc Section 1 placeholders (`[X]%`, `[selective / fixed-parameter]`, `[selectivity / state dimension]`) and produce both LOSO variants.
**Can share a session with Phase 1** (master doc Section 18).

---

## 2.1 Fold generator

- **Within-type LOSO:** 4 folds per machine type. Train on 3 cases (normal only), test on held-out case (normal + anomalous).
- **Cross-type LOSO:** train all 4 ToyCar → test all ToyTrain, and the reverse.

**Cross-type is the higher priority.** Master doc Section 14 confirms no published cross-type numbers exist for this dataset, whereas within-type cross-ID numbers do — making within-type a sanity check / re-measurement under deployment constraints, not a novel result. If time forces a choice, cross-type wins.

**Refit inside every fold, from source data only:**
- feature normalization statistics (per-mel-bin mean/std)
- the Option 3 centroid / covariance
- `mse_persistence` — the skill-score denominator is fold-specific

Any of these computed globally is leakage.

**Near-term scope: within-type LOSO on ToyCar only.** Per `01_design_decisions.md` §6, ToyTrain — and therefore cross-type LOSO — is deferred until an SSM configuration has been evaluated on MCU hardware. Deployment feasibility is the primary spine (master doc Section 1); generalization work is sequenced after it, not run in parallel with it. Cross-type remains the higher-priority variant *when it runs* (master doc Section 14: no published cross-type numbers exist for this dataset, whereas within-type cross-ID numbers do) — but "higher priority within the generalization work" and "in scope right now" are different claims. This document's fold generator, run-count budget, and exit gate below cover ToyCar within-type LOSO only.

---

## 2.2 Nested validation — resolving master doc Section 17 item 5

Master doc Section 13 commits to nested validation and leaves the inner scheme open. Concrete proposal:

```
for outer_case in [1,2,3,4]:                    # 4 outer folds
    train_cases = the other 3
    for inner_case in train_cases:              # 3 inner folds
        fit on the 2 remaining cases (normal only)
        score AUC on inner_case (normal + anomalous)
    pick the config with best mean inner AUC
    refit that config on all 3 train_cases
    report AUC on outer_case                    # touched exactly once
```

The inner held-out case's anomalies are used for **config selection**, never for fitting. Same category as master doc Section 9.1's threshold method 3 — disclose it the same way in the writeup.

### Run-count arithmetic

| axis | levels | non-default runs |
|---|---|---|
| state dim N (8/16/32/64) | 4 | 3 |
| selective vs fixed | 2 | 1 |
| ZOH vs Euler | 2 | 1 |
| log-mel vs per-frame stats | 2 | 1 |
| conv kernel (4 / 2 / none) | 3 | 2 |
| depth (L / L−1 / L−2) | 3 | 2 |
| prediction horizon k (1/2/5/10) | 4 | 3 |
| **default** | — | 1 |
| **total configs** | | **14** |

Per outer fold: 14 configs × 3 inner folds = 42, plus 1 refit = **43 runs**.
× 4 outer folds × **1 machine type (ToyCar)** = **172 runs**.

At 10 min/run ≈ **28.7 hours**. At 45 min/run ≈ **5.4 days**.

(Previously budgeted at 344 runs / 2 machine types, before ToyTrain was deferred per `01_design_decisions.md` §6. Re-add ToyTrain's 172 runs, plus cross-type folds, once that deferral lifts — do not assume the doubled figure until then.)

### Mitigation ladder — apply in this order

1. **Single fixed inner split** instead of 3 inner folds → 14 + 1 = 15 per outer fold → 15 × 4 outer folds = **60 runs**. Best value; costs the least methodological ground.
2. ~~Full sweep on cross-type only; within-type at default config only~~ — moot for now; cross-type isn't in scope until ToyTrain is brought in.
3. Drop `k=1` — known-degenerate, arguably belongs in the Phase 1 pilot, not the sweep → 13 configs.
4. Drop N=8, or the depth axis — least load-bearing for the Section 1 claim.
5. Subsample training clips per fold.

**Do not** mitigate by dropping nested validation — master doc Section 13 already identified that as leakage.

### Mitigation ladder — apply in this order

1. **Single fixed inner split** instead of 3 inner folds → 14 + 1 = 15 per outer fold → **120 runs**. Best value; costs the least methodological ground.
2. Full sweep on **cross-type only**; within-type at default config only.
3. Drop `k=1` — it is the known-degenerate setting and arguably belongs in the Phase 1 pilot, not the sweep → 13 configs.
4. Drop N=8, or the depth axis — least load-bearing for the Section 1 claim.
5. Subsample training clips per fold.

**Do not** mitigate by dropping nested validation. Master doc Section 13 already identified that as leakage.

---

## 2.3 Execution order

**1. Axis 2 (selective vs fixed) first.** Master doc Section 13 says it *"could justify the paper on its own,"* and it determines whether Phase 4 ports a data-dependent recurrence (hard — the Section 11 point 1 problem) or a static one (much easier to quantize). A cheap early answer reshapes the entire second half of the project.

**2. Axis 1 (state dimension).** The other half of Section 1's `[selectivity / state dimension]` placeholder — and simultaneously diagnostic (a) from Phase 1 §1.9.

**3. Horizon axis.** Cheap, and tells you whether the whole training objective is well-posed.

Everything else after.

---

## 2.4 Reporting

- **Per-fold table, mean ± std.** Never a bare mean — master doc Section 14's small-N caveat says per-fold variance is itself informative.
- **Both LOSO variants reported separately** (Section 14, committed). "Generalizes across units but fails across types" and "generalizes across both" are different findings. - OUTDATED
- **Within-type reported now; cross-type deferred.** Master doc Section 14's commitment to reporting both variants still holds — but only within-type (ToyCar) is in scope for this pass. The "generalizes across units vs. across types" comparison isn't answerable until ToyTrain LOSO exists, later, per `01_design_decisions.md` §6.
- **Skill score alongside AUC for every run.** A config with high AUC and near-zero skill is suspicious — investigate before it goes in a table.
- Resolve the Section 1 placeholders.

---

## Exit gate

1. Both LOSO variants complete, per-fold numbers recorded. - OUTDATED
2. **Within-type LOSO (ToyCar) complete**, per-fold numbers recorded. Cross-type LOSO is out of scope for this exit gate — deferred per `01_design_decisions.md` §6, to be re-checked once ToyTrain is brought in post-MCU-deployment.
3. Section 1 placeholders resolvable from the data — you can state which lever the evidence supports.
4. A winning config identified through the nested procedure, not by eyeballing outer-fold results.
5. `parity_vectors.npz` generated *before* the training environment is torn down.

---

## Handoff manifest → Phase 4

This is the manifest that makes a future Phase 4 session productive rather than speculative.

1. **Winning config**, frozen YAML.
2. **Trained checkpoint** for that config.
3. **Exact parameter count and per-tensor shapes** — every weight, dtype, byte count. This is what gets compared against each board's flash budget.
4. **`parity_vectors.npz`** — the critical one. An input frame sequence, every intermediate tensor (post-conv, delta, B, C, A_bar, hidden state at selected timesteps, final embedding), and the resulting anomaly score. MambaLite-Micro validated their C engine against a PyTorch reference to ~1.7×10⁻⁵ (master doc Section 3). You cannot run that test in Phase 4 without these vectors, and generating them afterwards means rebuilding a training environment you have moved on from.
5. **`ranges.json`** — per-tensor activation dynamic ranges. Phase 5 input.
6. **Fitted Option 3 head parameters** — centroid, covariance/inverse, thresholds from all three Section 9.1 methods.
7. **Prediction-head weights** — needed for the fused-score variant (`01_design_decisions.md` §4).
8. **The measured accuracy ceiling** — cell (3) of master doc Section 12.
