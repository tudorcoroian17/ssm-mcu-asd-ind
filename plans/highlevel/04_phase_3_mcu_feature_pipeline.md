# Phase 3 — Feature Pipeline for MCU

**Prerequisites:** Phase 1 §1.2 must have frozen the feature parameters and exported `mel_filterbank.npy`. Nothing else.
**Decoupled from Phases 1–2** (master doc Section 18) — this does not depend on which SSM config wins, so it can run in parallel at any time.
**Goal:** reproduce the GPU feature pipeline on each board, within a stated numerical tolerance, and measure what it costs.

**Open question blocking the schedule:** do you physically have all three boards? If not, this phase splits into "develop against the one you have" and "port to the others", and the ordering changes.

---

## 3.1 Export the contract

- `mel_filterbank.npy` → C header. Decide float32 array vs q15 fixed-point.
- Window function array (Hann/Hamming) → C header.
- **Record the flash cost:** a dense float filterbank is `n_mels × (n_fft/2 + 1) × 4 bytes`.

**Question:** at `n_mels=64`, `n_fft=1024`, what is that number, and how does it sit against the RP2040's 2 MB flash? Each mel triangle touches only a few FFT bins — is a sparse representation worth the code complexity, or is the dense version affordable?

Master doc Section 7.3 flags this as one of the "cheap and known" baseline costs that the SSM's numbers get compared against in the writeup — so it needs to be a real measured number, not an estimate.

---

## 3.2 CMSIS-DSP chain

Ring buffer → window multiply → `arm_rfft_f32` → magnitude → mel matmul → log.

Master doc Section 7.3 confirms every step here has existing tooling. **This is a parity exercise, not a research exercise** — the novel engineering risk lives in the SSM recurrence (Section 11), not here.

---

## 3.3 The log, per board

- **H7S3L8, ESP32:** hardware FPU — use the library `logf`.
- **RP2040: no FPU.** Master doc Section 7.3's lookup-table approximation. Decide table size and interpolation scheme, then **measure the error it introduces** — that error propagates into every downstream number on that board and must be reported separately.

---

## 3.4 Parity harness — the gate for this phase

Feed the same `.wav` to the Python pipeline and to each board. Compare per-bin.

**Define the tolerance before running, not after.**

**Question:** what tolerance is acceptable, and how do you justify it? It should be tied to something downstream — e.g. the perturbation size at which the Phase 1 GPU model's AUC starts to move — not chosen because it is a round number. Perturbing the GPU features with noise of increasing magnitude and watching AUC is a cheap way to derive a defensible number.

---

## 3.5 Skewness/kurtosis path

Feeds master doc Section 13 axis 4 (per-frame statistical summary instead of full log-mel bins).

Two-pass implementation per master doc Section 8.2 — **not Welford's**, for the reasons in Section 8.3 (division cost dominates at N=64 in-memory).

Then all four Section 8.4 precision checks, especially:

- **Check 2:** at least a 32-bit accumulator for the running power sums, even with 8-bit inputs. Cubing a near-max int8 value gives ~2 million; the 4th power ~260 million.
- **Check 3 — the important one:** compare the **statistic values themselves** against the GPU float values, not just downstream accuracy. A silently overflowed kurtosis can look fine in an AUC number while being numerically garbage.
- **Check 4:** minimum-σ floor for near-silent/flat frames, since both statistics divide by σ³/σ⁴.

---

## 3.6 Measurements per board

| measurement | why |
|---|---|
| flash: constants + code | the "cheap and known" baseline the SSM is compared against |
| RAM: windowing/ring buffers | competes with the SSM's state and activation memory |
| cycles per frame | |
| **real-time margin**: cycles-per-frame vs hop duration | if frame processing exceeds the hop, streaming is impossible regardless of what the model costs |

That last row can independently kill a board. Measure it early.

---

## Exit gate

1. Parity within the stated tolerance on all three boards.
2. RP2040 log-LUT error quantified separately.
3. Real-time margin recorded per board.
4. All four Section 8.4 precision checks passed on the stats path.

---

## Handoff manifest → Phase 4/5

- Working C feature pipeline per board
- Flash/RAM/cycle costs per board
- Parity error figures per board
- RP2040 log-LUT error characterisation
- Real-time margin per board — **the budget the SSM has left to fit into**
