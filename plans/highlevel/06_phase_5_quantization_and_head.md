# Phase 5 — Quantization + On-device Anomaly Head

> **Conditional skeleton.** Depends on Phase 4. This is a decision tree rather than a step list *because there is no prior art to copy* — see below.

**Prerequisites:** Phase 4 handoff manifest, plus `ranges.json` from Phase 2.
**Goal:** the project's first genuinely novel engineering result.

---

## Why this is a tree and not a list

Master doc Section 17 item 10 calls this the single most open technical question and *"likely the first genuinely novel engineering result the project will produce."* MambaLite-Micro ran full fp32 and never tested quantization (master doc Section 3). There is no existing evidence either way for whether the selective-scan recurrence survives int8.

That means the plan cannot be "do X" — it has to be "try X, and here is what to do when it fails."

---

## Escalation ladder — descend only as each rung fails

1. **Post-training int8 (PTQ), whole model.** The cheapest thing that could work.

2. **Mixed precision.** Quantize the linear projections; keep the recurrence in int16 or fp32. `ranges.json` from Phase 2 identifies *which* tensors force this — that is exactly why Phase 1 §1.5 insists on logging activation ranges during training, long before they look useful.

3. **Quantization-aware training (QAT).** Expensive; requires returning to the GPU environment and re-running part of Phase 1/2. Budget for this properly if you reach it.

4. **Documented failure with diagnosis.** Which operation, which tensor, what dynamic range, why int8 could not represent it. **Publishable per master doc Section 2**, not a fallback — "we attempted MCU deployment of an SSM-based ASD model; here is what broke and why" is a legitimate citable contribution and is what protects this project from being scooped.

Whichever rung you land on, the *diagnosis* is the paper content. Record dynamic ranges and failure modes as you go, not retrospectively.

---

## On-device anomaly head

- Option 3's distance computation (centroid / Mahalanobis per Phase 1 §1.6).
- All three master doc Section 9.1 threshold methods.
- **The per-machine-calibration question** (Section 9.1): does a single global threshold transfer across units, or does each deployed unit need to calibrate its own from its own normal running data after installation? If the latter, that is itself a reportable finding about what does and does not generalize — and it connects the deployment spine to the LOSO story.

---

## Added scope — the fusion ablation

From `01_design_decisions.md` §4: the prediction head yields `S_recon` essentially free.

Compare on-device:
- distance-only: `S_latent`
- fused: `S_total = α·Norm(S_recon) + (1−α)·Norm(S_latent)`

Measure the memory and compute cost of each. This turns master doc Section 17 item 12 from an apology into a result — you can state what the fusion is worth on constrained hardware, which the published paper cannot.

---

## Handoff manifest → Phase 6

- Quantized model per board, or documented failure with diagnosis per board
- Accuracy at each rung of the ladder
- Footprint at each rung
- Threshold-transfer results (global vs per-machine)
- Fusion ablation: accuracy and cost, distance-only vs fused
