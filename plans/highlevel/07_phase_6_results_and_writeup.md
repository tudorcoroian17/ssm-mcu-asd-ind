# Phase 6 — Cross-platform Results + Writeup

> **Conditional skeleton.** Depends on Phase 5 across all three platforms.

**Prerequisites:** Phase 5 handoff manifest for each board.

---

## 6.1 Build the deferred GPU autoencoder baseline

This is where it lands (your decision, recorded in Phase 1 §1.8). Until it exists, every SSM accuracy number in the project is uninterpretable in isolation.

It supplies cell (1) of the master doc Section 12 comparison, and — more importantly — it is the only thing that answers "was the architecture change worth it?"

---

## 6.2 Assemble the three-way comparison

Master doc Section 12:

| | accuracy | memory/flash | latency |
|---|---|---|---|
| (1) Autoencoder on MCU | | | |
| (2) SSM on MCU | | | |
| (3) SSM on GPU, unconstrained | | | |

- Gap (3) → (2) = **cost of deployment/quantization specifically**.
- Gap (1) → (2) = **whether the architecture change was worth it despite that cost**.

Master doc Section 5 and Section 12 both stress **memory/flash as the primary metric, not latency** — the audio capture window dwarfs inference time. Report latency, but do not lead with it.

Report per board. Master doc Section 4 pre-commits that a "doesn't fit" row for the RP2040, with a clear diagnosis of why, makes the deployment-feasibility story *stronger*, not weaker.

---

## 6.3 Framing decisions

**Choose the final Section 1 claim** based on what Phase 2 axis 2 actually showed:

- **Conservative** — if ablation results were messy: first MCU deployment of an SSM-based ASD detector, measuring the accuracy cost of quantization and reduced state dimension.
- **Moderate** (current default) — deployment feasibility as the spine, with an identified architectural lever governing the accuracy–efficiency tradeoff.
- **Ambitious** — only if fixed-parameter genuinely matched selective: the components hardest to deploy are not the components responsible for generalization.

**The fusion gap** (master doc Section 17 item 12) is now answerable with data rather than explanation, thanks to Phase 5's fusion ablation.

**State the DCASE-SOTA relationship explicitly** (master doc Section 2): the published SSM ASD work already establishes that SSMs work well on this dataset family. Novelty lives entirely in deployment feasibility. Say so before a reviewer asks why the accuracy number is treated as novel.

**The mechanistic evidence** from Phase 1 §1.9(c) — decay half-lives read off `A` — is the strongest available support for Section 1's claim that state dimension is a real architectural lever. Use it; a measured time constant beats a correlation across an ablation sweep.
