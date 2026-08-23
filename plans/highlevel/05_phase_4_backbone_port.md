# Phase 4 — SSM Backbone MCU Port

> **Conditional skeleton.** Cannot be specified further until Phase 2's handoff manifest exists — specifically the winning config and `parity_vectors.npz`. Master doc Section 18 is explicit that this phase has nothing to port until the GPU prototype exists. What follows is the decision structure, not a step list.

**Prerequisites:** Phase 2 handoff manifest (all 8 items), Phase 3 handoff manifest (real-time margin per board).
**Goal:** the winning config running in fp32 C on each board, numerically verified against the PyTorch reference.

---

## The branch point

Everything about this phase depends on what Phase 2 axis 2 showed.

| Phase 2 result | what you port | consequence |
|---|---|---|
| **fixed ≈ selective** | a static recurrence | Master doc Section 11 point 1 evaporates. Quantization becomes conventional. Section 1's **ambitious** framing becomes claimable |
| **selective wins clearly** | the data-dependent recurrence | MambaLite-Micro's fusion trick becomes load-bearing. Section 11 point 1 is live and Phase 5 gets much harder |

Run axis 2 first in Phase 2 precisely so this is known early.

---

## Step shape, either way

1. **Diff the chips.** Nucleo-H7S3L8 vs MambaLite-Micro's STM32H747XIH6 (master doc Section 17 item 11) — flash, RAM, clock, FPU. **Do not assume their result transfers** just because both are H7-family.

2. **Read their C engine.** `github.com/Whiten-Rock/MambaLite-Micro`, MIT licensed, confirmed live. Identify precisely which intermediate tensor their operator fusion avoids materialising (the ¯A / ¯Bu construction), and whether your winning config has the same one. If your config differs structurally, their 83% peak-memory reduction may not transfer at the same magnitude.

3. **Implement in C, fp32 first.** Match their proven approach before adding a second unproven variable. Quantization is Phase 5, deliberately.

4. **Parity test against `parity_vectors.npz`.** Target their ~1.7×10⁻⁵ order of magnitude. Test intermediate tensors, not just the final score — an error that only shows at the output tells you nothing about where it originated.

5. **Measure flash / RAM / cycles per board**, and check against Phase 3's remaining real-time margin.

6. **RP2040 is expected to fail.** Master doc Section 4 pre-commits that a documented "doesn't fit, here's why" is a valid outcome — so the **diagnosis is the deliverable** there, not a workaround. Record: which tensor, how many bytes over, whether the binding constraint is flash or RAM, and whether the absence of an FPU or the memory ceiling bites first.

---

## Handoff manifest → Phase 5

- Working fp32 C implementation per board
- Measured fp32 footprint per board (flash, RAM, cycles)
- Parity error figures, per intermediate tensor
- **Identified binding constraint per board** — this is what Phase 5's escalation ladder is trying to relieve
