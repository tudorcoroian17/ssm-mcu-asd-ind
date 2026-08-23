# Research Direction: State-Space Models for MCU-Deployed Audio Anomaly Detection

**Status:** All core decisions committed — claim, methodology, target hardware, anomaly head, LOSO scope, validation scheme, and experimental sequencing. Ready to move to concrete implementation planning; remaining open items (Section 17, points 9–12) are technical unknowns to resolve *during* implementation, not blockers to starting it.
**Context:** PhD-track paper on efficient ML deployment for MCU platforms, audio anomaly detection domain. Prior work: thesis using ToyADMOS-based dataset, log-mel spectrogram (2D, 5-window concatenation) → autoencoder, reconstruction-error anomaly scoring.

---

## 1. Current working claim (moderate framing — committed)

> We show that [selective / fixed-parameter] SSM-based ASD models can be deployed on MCUs at [X]% of GPU accuracy, and identify [selectivity / state dimension] as the primary architectural lever governing the accuracy–efficiency tradeoff — with the same lever affecting cross-machine generalization, suggesting deployment constraints and generalization robustness are not independent design axes.

**Primary spine: deployment feasibility.** Cross-machine generalization (via LOSO, see Section 14) is a **supporting claim**, not the main contribution.

Two other framings were considered and rejected for now, but kept here in case results push the paper toward either:
- **Conservative** (fallback if ablation results are messy/inconclusive): "We present the first MCU deployment of an SSM-based audio anomaly detector, measuring the accuracy cost of quantization and reduced state dimension relative to a GPU baseline." — pure feasibility/measurement, low risk, less ambitious.
- **Ambitious** (upgrade path if ablation data clearly supports it, do not start here): "A simplified (fixed-parameter, low-state-dimension) SSM — specifically chosen to be MCU-deployable — matches or exceeds a full selective SSM's cross-machine generalization, suggesting the components hardest to deploy are not the components responsible for generalization." Only claim this if the selective-vs-fixed ablation actually shows it.

**Commitment on methodology: unsupervised only.** Explicitly ruled out training/fine-tuning on labeled anomalous audio (see Section 14) — stay within the unsupervised convention (train on normal-only data) that the rest of the field (DCASE Task 2) uses, rather than breaking from it. Anomalous recordings are for **evaluation/scoring only**, never for training or fine-tuning.

**Dataset confirmed:** ToyCar and ToyTrain datasets, 4 cases (machine IDs) each, each case has separate normal and anomalous recording folders. Note: ToyCar/ToyTrain are standard machine types from the DCASE ASD benchmark family (ToyADMOS2/MIMII DG lineage) — meaning published baseline numbers likely already exist for cross-ID/cross-type generalization on this exact data (see Section 17 — still pending: check DCASE technical reports for existing ToyCar/ToyTrain cross-machine baselines, to confirm the LOSO experiment adds something beyond reproducing known numbers).

---

## 2. The identified gap — original hypothesis

**Hypothesis:** Recent state-space model (SSM) backbones — e.g. the SSM-based approach shown in a DCASE 2025 workshop paper (Emon & Anon, "Efficient State-Space Model for Audio Anomaly Detection with Domain Adaptation") — have **not been evaluated for MCU/embedded deployment**. SSMs are architecturally motivated by *efficiency* (linear cost in sequence length vs. quadratic for transformers), which makes them a plausible but unverified candidate for constrained hardware.

**Confirmed details on the DCASE SSM paper (resolves Section 17 item 2):**
- **Task/dataset:** Unsupervised / First-Shot Anomalous Sound Detection (ASD), evaluated on **MIMII-DG & ToyADMOS2** — the **same dataset family as Tudor's own ToyCar/ToyTrain data** (ToyADMOS2 lineage). **Important implication:** DCASE-SOTA-level accuracy numbers likely already exist for something very close to the exact machine types in this project's dataset. This is not a problem for the project, but it sharpens the framing: novelty must live entirely in **deployment feasibility** (already the committed primary spine, Section 1), not in "does an SSM work well on this data" — that appears to already be established. Worth being explicit about this in the writeup so a reviewer doesn't ask why the accuracy number itself is treated as novel.
- **Parameter count: ~2.5M–6.5M**, depending on whether a lightweight 4–6 layer bidirectional SSM block sequence or a hybrid CNN-SSM encoder is used. **Red flag worth sitting with:** at float32 that's 10–26MB; even aggressively int8-quantized, 2.5–6.5MB. The committed RP2040 target (Section 4) has 2MB total flash — **this architecture, as published, almost certainly does not fit at all on the leanest target, and may be tight even on the H7S3L8** depending on its exact flash size. "Shrink this specific architecture enough to fit" likely needs to be an explicit, first-class methodology step, not an assumed given (see Section 17).
- **Anomaly head:** a **fusion of two heads** — Autoregressive/Reconstruction Error Head (comparable to Option 1, Section 10) **and** Latent Space/Classification Anomaly Head (comparable to Option 2/3, Section 10) — combined as `S_total = α·Norm(S_recon) + (1−α)·Norm(S_latent)`. **Implication for this project's chosen Option 3 (distance-only, no reconstruction):** the published SOTA number depends partly on the reconstruction pathway that was deliberately excluded here for deployment-cost reasons (Section 10, honourable mention). If this project's accuracy comes in below the published DCASE number, this fusion gap is the most likely, and fully expected/explainable, reason — not a flaw in the simplified approach, but a real, reportable trade-off directly tied to the deployment-feasibility spine.

**Why this is a "hard" gap, not a "soft" one:**
- It's not just "nobody got around to it" — there is a concrete, checkable technical reason deployment is nontrivial: the recurrence's B and C matrices (in the "selective"/Mamba-style variant) are computed dynamically per timestep from the input itself, rather than being fixed learned weights. Standard embedded quantization toolchains (CMSIS-NN, TFLite Micro) are built around static weight quantization; a per-timestep data-dependent computation is a different, less-paved problem.
- The DCASE SSM paper reviewed does **not** mention parameter count, latency, or any hardware/deployment target — this space appears genuinely unaddressed in the literature as of this writing (mid-2026), not merely under-published.
- Even a **negative result is publishable**: "we attempted MCU deployment of an SSM-based ASD model; here is what broke, why, and how the standard autoencoder baseline compares" is a legitimate, citable contribution — this protects the project from being "scooped" in the way a soft gap would.

**Open research question (the actual paper question):**
> Does an SSM-based audio anomaly detector, once quantized and fit onto MCU memory/compute budgets, retain enough of its accuracy advantage over a standard autoencoder to justify the added deployment complexity — or does the gain collapse under real hardware constraints?

---

## 3. Gap update — MambaLite-Micro narrows the hypothesis

**Finding:** A paper called **MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs** (Xu et al., arXiv:2509.05488, Sept 2025) has already demonstrated a real, working Mamba deployment on actual MCU hardware (**STM32H747XIH6, Arduino Portenta H7, M7 core**), via a hand-written, runtime-free C inference engine. Key results:
- First known deployment of a Mamba-based architecture on a resource-constrained MCU.
- Solved the memory problem specifically by fusing operations to avoid materializing large intermediate tensors (the ¯A / ¯Bu construction in the standard Mamba algorithm) — **83.0% peak memory reduction**.
- Numerical accuracy essentially preserved vs. PyTorch reference (avg error ~1.7×10⁻⁵).
- **Confirmed: full fp32 precision throughout, no quantization.** Paper notes their fp32 KWS throughput is already comparable to *quantized (int8)* deployments of attention-based models — a strong result, but it means the "does this survive quantization" question is still completely open, for both KWS/HAR and (more importantly) for your ASD use case.
- Evaluated on **keyword spotting (KWS)** and **human activity recognition (HAR)** — both classification tasks, **not audio anomaly detection**.
- **Code confirmed live**, MIT license: `github.com/Whiten-Rock/MambaLite-Micro`.
- **Note:** `STM32H747XIH6` is a *different specific chip* than the committed Nucleo-H7S3L8 target (Section 4), even though both are H7-family — flash/RAM/clock specs should be diffed before assuming the H7S3L8 inherits this result directly (see Section 17).

**What this changes:** Section 11, Point 1 (per-timestep dynamic B_t/C_t resisting quantization) is **no longer an entirely open question for float32** — MambaLite-Micro shows the core selective-scan recurrence CAN be made to run on real MCU hardware without exploding memory, in fp32. **The quantization question itself remains fully open** — this is arguably now the single most important unresolved technical question for the whole project (see Section 17).

**What is still open (the gap is narrower now, not closed):**
1. **STM32H747XIH6 is a relatively high-end/capable MCU** (hardware FPU, larger RAM than typical ultra-low-power TinyML targets). **Confirmed: no quantization used.** Open question: does this approach survive (a) int8/lower-precision quantization, and (b) deployment on a leaner, cheaper MCU class more typical of TinyML papers — i.e. your ESP32/RP2040 targets (Section 4)?
2. **KWS/HAR are short, low-dimensional, well-behaved classification tasks.** They exercise the Mamba *backbone* but not a fused multi-level anomaly-detection *head* (as in the DCASE SSM ASD paper) or the specific structure of audio anomaly detection (longer sequences, subtler distinctions between normal/anomalous vs. clean class boundaries in KWS/HAR).
3. ~~Code availability unconfirmed~~ — **RESOLVED: code is live, MIT licensed.** Usable as a starting point for the backbone implementation.

**Revised working hypothesis (narrower, still hard-gap-shaped):**
> Does an SSM-based audio anomaly detector — using the now-proven MambaLite-Micro-style deployment approach for the backbone — hold up when (a) **quantized** (still fully open, since MambaLite-Micro itself never tested this), (b) deployed on more constrained MCU hardware than STM32H747XIH6, and (c) extended with the anomaly-detection head (not just a classification head)? Does accuracy survive across all three changes simultaneously, or does one of them break it?

This is a good example of exactly the kind of update this document is meant to capture — a literature discovery should narrow/sharpen a hypothesis, not necessarily kill it. Re-check the field periodically for further narrowing before committing.

---

## 4. Target MCU platforms (committed)

Three platforms targeted, deliberately spanning a capability range — from "should be closest to MambaLite-Micro's own result" down to "expected to be the hardest, and a negative result there is still a good outcome":

1. **Nucleo-H7S3L8** — Cortex-M7, hardware FPU, generous flash/RAM. Same family as MambaLite-Micro's own STM32H7 target (Section 3) — the platform closest to reproducing/extending their result directly, and the natural test of whether the anomaly head (not just the backbone) survives on top of a proven MCU-Mamba deployment.
2. **ESP32-WROOM-32** — dual-core Xtensa, has hardware FPU, but meaningfully less RAM than the H7S3L8. Mid-tier test: still FPU-capable, but a real step down in memory headroom.
3. **Arduino Nano (RP2040)** — Cortex-M0+, **no hardware FPU**, least RAM/flash of the three. Leanest target; most likely to hit real walls. Directly exercises the log-step fallback flagged in Section 7.3 (lookup-table log approximation for FPU-less MCUs), and stresses the accumulator-width/quantization concerns from Section 8.4.

**Explicit framing, agreed:** a "doesn't fit" result on the RP2040 is still a valid, useful outcome for the thesis/paper — not a failure to route around. Three points across a capability range (comfortable fit → fits with compromises → doesn't fit, with a clear diagnosis of *why*) makes a stronger deployment-feasibility story than a single yes/no on one chip, and lines up with the "even a negative result is publishable" framing already established in Section 2.

**Not yet done:** get concrete flash/RAM numbers for all three boards side by side, and cross-check against MambaLite-Micro's reported STM32H7 variant/RAM (Section 3, still-open item 1) to know exactly how much headroom the H7S3L8 target has relative to their published result.

---

## 5. Background: what came before this hypothesis (context for future-me)

Before settling on SSMs, the following alternative directions were considered and explicitly ruled out or deprioritized — worth remembering *why*, to avoid re-treading the same ground:

- **"Deploy two models: one learned feature extractor + one detector."** Rejected as not novel — a standard autoencoder (encoder = learned feature extractor, decoder+reconstruction error = detector) already does this, just packaged as one end-to-end model. Splitting it doesn't inherently help unless there's a specific reason a separately-trained extractor would produce better features than joint training — no such reason was identified.
- **"Optimize for latency (≤2x inference time)."** Reframed: for audio anomaly detection, the audio capture window (often 1+ seconds) typically dwarfs model inference time (milliseconds). **Memory/flash footprint is the more binding constraint** for MCU deployment, not latency — this should be the primary metric going forward, not latency.
- **"Combine statistical feature extraction with the existing autoencoder."** Rejected — identified as a "fishing expedition" (trying combinations without a diagnosed reason to expect improvement) rather than a hypothesis-driven contribution. Lesson: before swapping any pipeline component, first identify *why* the current component is expected to be a weak point.
- **Knowledge distillation from large pretrained models (BEATs, PANNs, wav2vec2.0, AST) into a small MCU-sized student.** Considered but deprioritized in favor of the SSM angle. Key open question if revisited: how much does a distilled student need to shrink before it loses the *fine-grained* sensitivity anomaly detection needs (as opposed to broad classification, where distillation is well-proven)? This remains a valid fallback/alternative direction.
- **"Train on one machine, fine-tune with normal+anomalous sounds, test on other machines."** Rejected: this is already an established DCASE benchmark task ("domain generalization" 2022, "first-shot ASD" 2023–2025), so the framing itself isn't novel — and the fine-tuning-on-anomalies step specifically breaks the unsupervised commitment (Section 1) and has direct counter-evidence (Section 14).

---

## 6. State-space models (SSMs) — conceptual and mathematical primer

*(Theoretical background — safe to skip when building the concrete implementation step list; kept for conceptual grounding.)*

### 6.1 Core intuition
Instead of re-reading an entire audio sequence at every step (as attention/transformers do), an SSM keeps a compact **running state** — a fixed-size summary of relevant history — updated incrementally as each new piece of input arrives. Think of it as a structured, learnable recurrence, similar in spirit to an RNN but parameterized for efficient training.

### 6.2 The base equations
At each time step *t*:

- **State update:** `state_t = (A × state_(t-1)) + (B × x_t)`
- **Output:** `output_t = C × state_t`

Where:
- **A** — controls how much of the previous state persists vs. decays.
- **B** — controls how much the new input `x_t` influences the state.
- **C** — projects the internal state into a usable output.

Because the update depends only on the *previous state* and *current input* (not the full history directly), the compute cost per step is constant, and total cost scales **linearly** with sequence length — unlike attention, which explicitly compares every timestep against every other timestep (**quadratic** cost).

### 6.3 What makes Mamba-style SSMs "selective"
Plain SSMs use the same A, B, C for every input regardless of content — efficient but rigid. **Selective SSMs (Mamba)** make B and C (and effectively how much A lets the state decay) **data-dependent** — computed fresh from the current input at each timestep. This lets the model dynamically decide "this matters, retain it" vs. "this is unremarkable, let it fade" — closer to attention's flexibility, achieved via a computation trick (the "selective scan") that still runs in linear time.

**This data-dependence is precisely the part that complicates MCU deployment** — see Section 11.

---

## 7. Log-mel spectrograms — conceptual and MCU implementation primer

*(Theoretical background — safe to skip when building the concrete implementation step list; kept for conceptual grounding.)*

### 7.1 What it is, step by step
1. **Raw waveform** — a microphone records air pressure over time → one long list of numbers (samples). Nothing more than that at this stage.
2. **Why we need frequency, not just loudness** — a raw waveform shows loudness moment-to-moment but not *which pitches* are present. A **Fourier Transform** breaks a chunk of signal into its component frequencies.
3. **Short-Time Fourier Transform (STFT)** — sound changes over time, so instead of one Fourier Transform on the whole clip, slide a small window (e.g. 25ms) across the waveform with overlap, and run a Fourier Transform on each window separately. Produces, per time-slice, a breakdown of how much of each frequency is present.
4. **Spectrogram** — stack those per-window frequency breakdowns side by side (time left-to-right, frequency bottom-to-top) → a 2D picture: literally many small Fourier Transforms stitched together.
5. **Mel scale** — human hearing (and useful machine-sound patterns) distinguish low frequencies much better than high ones. The mel scale is a fixed, warped frequency axis that compresses highs and stretches lows. A **mel filterbank** — a fixed set of overlapping triangular weightings, computed once — groups/averages the many raw frequency bins into fewer mel bins (e.g. 512 → 64) using this warping.
6. **Log** — audio energy spans a huge range; quiet-but-meaningful detail can get drowned out by loud moments. Taking the **logarithm** of the mel spectrogram values compresses that range so quiet detail isn't lost.

**Log-mel spectrogram = raw waveform → STFT (sliding-window Fourier Transform) → mel filterbank (fixed weighted grouping of frequency bins) → log.** Every step is fixed, deterministic math — nothing learned, same formula every time.

### 7.2 Where this sits in the rest of the plan
- This is the **first arrow** in the Section 9 pipeline (audio → spectrogram). Each "frame" the SSM processes one at a time in the recurrence (Section 9, Step 3) is literally one column of this log-mel spectrogram.
- Directly feeds **Section 13, Axis 4** (full log-mel bins vs. per-frame statistical summary): the statistical summary (mean/std/skew/kurtosis) would be computed *across the mel bins of one column* — the exact object defined here.
- Prior thesis autoencoder work used the same computation (STFT → mel → log), just packaged differently (5-window concatenation into a flattened 2D "image") rather than consumed as a time sequence.
- Near-universal starting point across the field's literature (DCASE baselines, all papers reviewed in Section 16 reading list) — assumed infrastructure, not treated as a novel contribution in the literature.
- **Key asymmetry vs. the SSM backbone:** log-mel computation is fixed/deterministic (no data-dependence). The SSM's selective B_t/C_t (Section 6.3) are computed *from the data* at each timestep. This asymmetry is why log-mel is the "easy," well-paved part of the MCU pipeline, while the SSM recurrence is the genuinely novel engineering risk (see 7.3 and Section 11).

### 7.3 What this means for MCU implementation (why it's the "easy" part)
1. **Capturing/windowing** — rolling buffer of raw samples in RAM (one window + overlap). Each window multiplied by a fixed window function (e.g. Hamming/Hann) — a precomputed constant array, one multiply per window. Small, manageable RAM cost.
2. **FFT** — the main compute cost, but well-paved: ARM Cortex-M MCUs have **CMSIS-DSP**, a standard library with ready-made, hand-optimized real FFT functions (`arm_rfft_f32`, or fixed-point `arm_rfft_q15`). Not a research problem — a solved engineering problem with existing tooling, comparable to calling a Python library function but at a lower level.
3. **Mel filterbank** — fixed triangular weights, determined once by sample rate and bin count (never learned/changes). Compute offline in Python once, store as constants in MCU flash. On-device, just a small/sparse weighted sum against known constants — no runtime computation of the filterbank itself.
4. **Log** — needs either (a) `log()` from CMSIS-DSP/standard math library if the target MCU has hardware floating-point (FPU) support, or (b) a precomputed lookup-table approximation if avoiding floating point entirely (relevant for lower-end, FPU-less MCUs). Ties directly to the STM32H7-vs-leaner-MCU open question (Section 3/4/17).

**Bottom line:** every piece (windowing, FFT, filterbank, log) is provided or made straightforward by existing embedded DSP tooling (CMSIS-DSP or equivalent) — assembling known building blocks, not inventing new ones. This is explicitly **not** where the paper's novel engineering risk lives; that's the SSM's data-dependent recurrence (Section 11).

**Concrete numbers still needed once a target MCU is chosen (see Section 17):** flash cost of mel filterbank constants (scales with mel-bin count × FFT size) and RAM cost of the windowing buffer — small, boring, but needed as the "cheap and known" baseline cost that the SSM's numbers get compared against in the writeup.

---

## 8. Skewness/kurtosis computation — formulas, two-pass implementation, precision checks

*(Theoretical background — feeds directly into Section 13 Axis 4 implementation; safe to skip when building the concrete step list.)*

### 8.1 Formulas

Notation: `x_1...x_N` = values (for this project, N = mel bins in one frame, e.g. 64). `μ` = mean, `σ` = standard deviation.

- **Mean:** `μ = (1/N) × Σ x_i`
- **Variance / std:** `σ² = (1/N) × Σ (x_i − μ)²`, `σ = √(σ²)`
- **Skewness:** `skewness = [(1/N) × Σ (x_i − μ)³] / σ³` — cubing preserves sign (unlike squaring), so it captures asymmetry: values above the mean pull positive, below pull negative. Dividing by σ³ normalizes for scale so the result reflects shape, not loudness.
- **Kurtosis (excess):** `kurtosis = [(1/N) × Σ (x_i − μ)⁴] / σ⁴ − 3` — 4th power is always positive and dominated by rare, extreme values (heavy tails) — the "spike/click vs. steady hum" signal. The `−3` makes 0 = normal-distribution-like tails, positive = heavier tails, negative = lighter. Dividing by σ⁴ normalizes for scale.

### 8.2 Two-pass implementation, mapped to the formulas
- **Pass 1:** loop once over all `x_i`, accumulate `Σ x_i`, divide by N → `μ`.
- **Pass 2:** loop a second time; for each value compute deviation `d_i = x_i − μ` **once**, reuse it to accumulate `Σ d_i²`, `Σ d_i³`, `Σ d_i⁴` in the same pass (one subtraction per value, then just multiplications to build the powers — no recomputation).
- **Finalize (once, at the end, not per-value):** `variance = (Σ d_i²)/N`, `σ = √variance`, `skewness = [(Σ d_i³)/N] / σ³`, `kurtosis = [(Σ d_i⁴)/N] / σ⁴ − 3`.

### 8.3 Why not Welford's/Terriberry's incremental algorithm
Welford's advantage (single pass, no need to hold all data in memory) doesn't apply here — a frame's ~64 mel-bin values already sit fully in memory (needed for the SSM input anyway), so there's no streaming/memory-pressure problem to solve. Terriberry's incremental update requires a **division at every incoming value** (~64 divisions for 64 values), vs. two-pass's ~4–5 divisions total (only at finalization) — division is expensive on MCUs, more so on FPU-less/no-hardware-divide Cortex-M chips where it's emulated in software. **Precision is roughly comparable** between the two for a small, fixed, in-memory batch — Welford's precision edge only shows up over very long streams with accumulated floating-point drift, which doesn't apply at N=64 per frame. **Conclusion: use two-pass, not Welford, for this per-frame computation.** (Caveat: exact cycle counts are compiler/hardware-dependent — worth an empirical benchmark on the actual target MCU if this ends up mattering for final numbers.)

### 8.4 Precision checks to run before sending results downstream
1. **Why this needs checking at all:** skewness cubes deviations, kurtosis raises them to the 4th power — both amplify the *scale* of numbers and any *quantization error* already present in the input. Small rounding mistakes get magnified exponentially by the higher powers, more than mean/variance would ever reveal.
2. **Accumulator width vs. quantized input range:** if working in low-precision integer arithmetic (e.g. int8, range ~−128 to 127), cubing a near-max value gives ~2 million and the 4th power gives ~260 million — both already overflow a 16-bit accumulator, and summing many such values pushes further. **Use at least a 32-bit accumulator for the running power sums**, even if the input values themselves are 8-bit. Independent of two-pass vs. Welford — applies either way.
3. **Compare actual numeric values, not just downstream accuracy.** Compute skewness/kurtosis in float on GPU, then in the target fixed-point/quantized scheme on MCU, and diff the *statistic values themselves* — a value that silently overflowed or got truncated can still look "fine" in a downstream accuracy metric while being numerically wrong. This check should happen before trusting these features in the ablation (Section 13, Axis 4) results.
4. **Sanity-check normalization terms specifically:** since both skewness and kurtosis divide by σ³/σ⁴, a near-zero or poorly-estimated σ (e.g. a nearly-silent/flat frame) can blow up the result — worth checking for degenerate near-constant frames in the data and deciding how to handle them (e.g. a minimum-σ floor) before this becomes a silent source of outlier feature values.

---

## 9. Concrete pipeline: audio input → anomaly decision (SSM version)

| Step | What happens | Compare to current autoencoder pipeline |
|---|---|---|
| 1. Raw audio → spectrogram | Waveform → log-mel spectrogram (2D: freq bins × time frames) | Identical to current approach |
| 2. Spectrogram → sequence | Spectrogram treated as a **sequence of time-frame columns**, fed one at a time (optionally downsampled via "pixel unshuffle" — a reshaping op, not a learned computation, mainly a training-speed trick) | Current approach instead treats the spectrogram as a static 2D image (5-window concatenation into 1D) |
| 3. Recurrence over frames | At each frame: `state_t = (A × state_(t-1)) + (B_t × x_t)`, `output_t = C_t × state_t`. In the selective variant, B_t and C_t are computed per-frame from the input. | No analog — this is the key structural difference |
| 4. Final state/output → embedding | Fixed-size vector summarizing the whole clip's temporal structure | Analogous to the autoencoder's bottleneck/latent vector |
| 5. Embedding → decision | Compare embedding to a model of "normal" (e.g. distance to cluster center, or an auxiliary-task classifier) → anomaly score | Current approach instead uses reconstruction error (input vs. decoder output) |

---

## 10. Anomaly head design options — memory/compute tradeoffs

**Context:** the SSM backbone alone is a sequence encoder — good at compressing temporal audio structure (order, persistence/decay of patterns, transients vs. steady-state) into a fixed-size embedding. It has **no built-in notion of "normal" vs. "anomalous"** — that's entirely the job of whatever head sits after it. Three options were considered.

**Decision status: Option 3 (distance-based) chosen as the primary anomaly head.** Option 2 (auxiliary-task classifier) is kept open as a **stretch goal, not discarded** — if time allows, implement it as a comparison point to check whether its added complexity actually earns better accuracy over Option 3, or whether the cheap distance-based head is already sufficient. Option 1 (reconstruction) remains excluded (see honourable mention below).

### Option 2 — Self-supervised auxiliary-task head (classifier on top of embedding) — STRETCH GOAL, if time allows
- **What it is:** a small classifier (1–2 dense layers) on top of the SSM embedding, trained on a proxy task solvable with normal data only (e.g. "which machine ID/condition is this?"). At test time, anomaly score = classifier confidence/entropy, or distance in the pre-classifier embedding space.
- **Memory cost:** small addition — a few KB for the dense layers, not comparable to a full decoder.
- **Compute cost:** small, one-shot — a couple of matrix-vector multiplies after the SSM runs once. No second pass over the sequence.
- **Training complexity:** moderate–high. Needs the proxy task set up; the domain-adversarial gradient-reversal layer mentioned in the published paper (Section 11, Point 4) is likely **not needed for this project**, since the dataset has no domain shift to correct for (Section 15) — simplifies this option if pursued.
- **Why this route might be chosen:** more deliberate shaping of the embedding space toward separating normal/anomalous, vs. relying on the backbone's raw embedding quality alone.

### Option 3 — Embedding-distance/density-based head (no trained classifier) — CHOSEN, primary anomaly head
- **What it is:** no classifier at all — compare the test embedding's distance to a stored model of "normal" (cluster centroid, GMM, k-NN reference points, or Mahalanobis distance using a covariance estimate). Anomaly score = that distance.
- **Memory cost:** lowest of all options — one reference vector (same size as the embedding) or a small reference set/covariance matrix. No additional trained weights.
- **Compute cost:** lowest — a single distance computation (Euclidean/cosine/Mahalanobis) per inference. No extra layers, no extra recurrence pass.
- **Training complexity:** low — often no extra training beyond the backbone itself.
- **Trade-off to be honest about:** quality depends entirely on how good the SSM's embedding space already is *without* task-specific fine-tuning pushing normal/anomalous apart — leans fully on the backbone being a naturally good feature extractor.
- **Why this route might be chosen:** cheapest on every axis relevant to the deployment-feasibility spine; also has direct empirical backing — the DCASE 2025 finding already in this file (Section 14) that training-free scoring backends outperformed fine-tuned systems.

#### 9.1 Threshold selection for Option 3 — COMMITTED: use all three approaches together

**Decision: implement and report all three threshold approaches, not just one** — treat them as complementary rather than mutually exclusive, since they answer slightly different questions:

1. **Percentile-based, from normal-only data** (most consistent with the unsupervised commitment). Run normal training/validation data through the pipeline, collect the distance distribution for genuinely normal sounds, pick a threshold at a chosen percentile (e.g. 95th or 99th) of that distribution. Never touches labeled anomalous data — stays fully inside the unsupervised methodology. **Use as the default/primary deployed threshold.**
2. **Statistical-distribution-based** (if a distribution shape can be assumed). If using Mahalanobis distance and normal embeddings are reasonably Gaussian, squared Mahalanobis distance follows a known chi-square distribution — threshold can be picked analytically (e.g. the value beyond which only 1% of a chi-square variable would fall) rather than empirically. **Use as a cross-check against the percentile-based threshold** — if they diverge a lot, that's a sign the Gaussian assumption doesn't hold for this data, itself a useful thing to report.
3. **Validation-set threshold selection using held-out labeled anomalies** (a step further — flag explicitly in writeup). If some labeled anomalies are set aside purely for *validation/threshold-calibration* (not training/fitting the model), thresholds can be swept to hit a target operating point (best F1, a precision/recall tradeoff, or closest point to the ROC curve's top-left corner). The model itself still never trains on anomalies, but this does use labeled anomalies for calibration — a strict reading of "unsupervised" sometimes avoids this too, so be explicit about the distinction in the writeup. **Use as an upper-bound/best-case reference** — how much would a peek at anomalies improve threshold choice over the two purely-unsupervised approaches above?

**For paper evaluation (headline results), still report AUC/pAUC regardless of which threshold(s) are used for deployment** — sweeping the distance value across all possible thresholds is the DCASE-standard approach and keeps results comparable to the baselines already in Section 16.

**Deployment wrinkle tied to the LOSO/generalization story (Section 14):** a single global threshold picked from one machine's normal-data distribution may not transfer well to a different unit — different machines can have different baseline "normal" distance distributions even when healthy. Open question worth testing across all three threshold methods: does the threshold need to be **per-machine-calibrated** (each deployed unit calibrates its own threshold from its own normal running data after installation) rather than fixed at training time? If so, that's itself a reportable finding about what does and doesn't generalize.

**Side-by-side summary (anomaly head options):**

| | Extra memory | Extra compute | Training complexity |
|---|---|---|---|
| Auxiliary classifier (Option 2) | Small (few KB) | Small (1 pass + tiny head) | Moderate–high (proxy task, possibly domain adaptation) |
| Distance-based (Option 3) | Lowest (one reference vector/stats) | Lowest (one distance calc) | Low (often no extra training) |

### Honourable mention — Option 1: Reconstruction-based head (SSM encoder + SSM decoder pair), NOT expected to be feasible for this project
- **What it is:** pair the SSM encoder with a mirrored SSM decoder that reconstructs the input sequence from the embedding; anomaly score = reconstruction error, frame by frame — the direct SSM-analog of the current autoencoder, with SSM layers instead of CNN layers.
- **Why it's excluded:** roughly **doubles the model footprint** (decoder ≈ comparable size to encoder — e.g. encoder at 40KB → decoder adds a comparable amount), **doubles the compute** (the recurrence runs twice — once to encode, once to decode — plus an extra full-sequence pass just to score the reconstruction), and the decoder is itself another SSM, meaning it inherits the same "no CMSIS-DSP function for this, must hand-implement" problem as the encoder (Section 11) — effectively doubling the amount of novel, unproven-on-MCU code the project is responsible for. This works directly against the deployment-feasibility spine (Section 1) rather than supporting it. Kept here only as a "why we didn't do this" reference point, not a real candidate.

---

## 11. Where this pipeline is expected to break on an MCU (and what to watch for)

1. **Per-timestep dynamic parameters (B_t, C_t) resist standard quantization.**
 Embedded toolchains (CMSIS-NN, TFLite Micro) are built around quantizing **static, fixed weights**. A computation where key matrices are freshly derived from the input at every timestep is a much less standard, less-supported operation. This is the single biggest expected point of failure — flag any tooling that claims to "just support" this without evidence. (Partially de-risked by MambaLite-Micro, see Section 3.)

2. **Floating-point arithmetic exposure.**
 The recurrence and the input-dependent projections (deriving B_t, C_t from x_t) may require floating-point operations that don't reduce cleanly to int8 without accuracy loss. Needs explicit testing — don't assume standard post-training quantization recipes transfer.

3. **No prior art on parameter count for MCU budgets.**
 The reviewed DCASE SSM paper does not report parameter count. Before committing further, obtain/estimate this — if it's already far outside MCU flash/RAM budgets (as full pretrained-embedding models like BEATs are), the "shrink it" problem may itself require a separate distillation step (see Section 5's deprioritized direction — may need to be revisited as a sub-component, not a competitor).

4. **Training-only components should drop cleanly at inference — verify this explicitly.**
 The domain-adversarial gradient reversal layer (used for domain adaptation during training) is expected to be droppable at inference time, since its role is only to shape training gradients. **Likely moot for this project anyway**, since the dataset has been confirmed to have no domain shift (Section 15) — this component probably isn't needed here at all, regardless of the drop-at-inference question.

5. **Multi-level feature fusion in the anomaly head.**
 The paper's anomaly head "fuses multi-level features" — unclear yet whether this fusion mechanism is deployment-friendly or introduces additional operations (e.g. attention-like weighting) that resist quantization. Needs direct inspection of the paper's architecture details or code (if released).

**Framing for write-up:** each of the above is a legitimate "trade-off point" — if any of them force a compromise (e.g. dropping the selective/data-dependent mechanism in favor of a fixed-parameter SSM to make quantization tractable), that compromise and its accuracy cost is itself a valid, reportable finding.

---

## 12. Proposed experimental design — three-way comparison

**Three-way comparison** (to cleanly separate "architecture gain" from "deployment cost"):

1. **Autoencoder on MCU** — existing baseline, real deployed numbers (accuracy, memory footprint, latency).
2. **SSM on MCU** — the contribution: quantized/deployed SSM-based detector, same metrics.
3. **SSM on GPU, unconstrained** — full-precision ceiling, no memory/quantization limits.

- Gap between (3) and (2) → cost of deployment/quantization specifically.
- Gap between (1) and (2) → whether the architecture change was worth it *despite* that cost.
- Without (3), it's impossible to distinguish "SSMs aren't actually better" from "SSMs are better, but deployment erased the gains."

**Primary metric emphasis:** memory/flash footprint over latency (see Section 5 — latency is likely not the binding constraint for this application; audio capture windows dwarf inference time).

**Sequencing — CONFIRMED: GPU-first.** Prototype and validate step (3), the SSM on GPU, before starting any deployment/quantization work on steps (1)/(2). This de-risks the project by confirming an accuracy gap worth chasing exists before investing in embedded engineering — this is the natural starting point for implementation work in a new session.

---

## 13. Mechanism-level ablation plan

**Method: one-at-a-time from a single baseline/default config** — NOT full factorial (too many runs given 8+ candidate axes × 4 LOSO folds × 2 machine types). Pick one default SSM configuration; vary one axis at a time from that default, holding everything else fixed.

**Priority axes (keep — directly serve deployment spine and/or generalization spine):**
1. **State dimension (N)** — e.g. 8/16/32/64. Headline axis: tests both overfitting-to-source-machine (generalization) AND is a direct memory-footprint knob (deployment). Top priority.
2. **Selective vs. fixed/non-selective SSM (S4-style)** — input-dependent Δ/B/C vs. fixed. Most important axis for the whole thesis arc: if fixed-parameter performs comparably, that sidesteps the core deployment obstacle (data-dependent B_t/C_t resisting quantization, see Section 11 Point 1) at little accuracy cost. Could justify the paper on its own.
3. **Discretization scheme (ZOH vs. Euler approximation)** — directly relevant to numerical stability under quantization; cheap to test.
4. **Input representation: full log-mel bins vs. per-frame statistical summary** (mean/std/skewness/kurtosis computed *across frequency bins, per time frame* — preserves temporal sequence, unlike whole-clip statistics which would destroy it and stop being a fair SSM ablation). Tests whether a drastically smaller per-timestep input (e.g. 64 bins → 4 stats) retains detection power. **Success framing: non-inferiority** (accuracy retained % vs. per-frame memory reduction %), not "must beat log-mel on raw accuracy" — fits the deployment-feasibility spine better than requiring an outright win. **Numerical risk to test explicitly:** skewness/kurtosis involve cubed/4th-power deviations — more prone to precision loss/overflow under quantization than mean/variance; compute in float on GPU vs. quantized on MCU and check whether the *statistic values themselves* drift, separately from downstream accuracy (see Section 8.4).

**Secondary axes (keep if budget allows):**
5. **Short conv kernel before SSM** — present (size 4) vs. smaller vs. removed. Tests whether local conv captures generalizable spectral texture or a machine-specific fingerprint shortcut.
6. **Depth (number of Mamba blocks)** — full stack vs. −1 vs. −2. Standard capacity/overfit axis.

**Deprioritized / cut for now (candidates for "future work" mention only):**
- **Bidirectionality** (causal vs. bidirectional) — flagged as likely not causally deployable on MCU anyway; only revisit if explicitly framed as an "offline ceiling" comparison, which is a different purpose than the deployment spine.
- **Δ parameterization** (per-channel vs. shared) — plausible but speculative; lower priority.
- **Input tokenization/patch size** (frame-level vs. multi-frame patches) — plausible but speculative; lower priority.

**Leakage risk — RESOLVED, nested validation chosen:** with only 4 cases per machine type, using LOSO folds to both *select* the best config (e.g. best N) and *report* final accuracy on those same folds is leakage — reported numbers would look better than true generalization. **Decision: use nested validation** (folds-within-folds for selection — e.g. within each outer LOSO fold's training cases, hold out one further case for config selection before testing on the true outer held-out case) rather than treating this as purely exploratory characterization. Implementation detail still to work out: with only 3 training cases per outer fold, the inner selection loop has very little data to split further — worth deciding a concrete inner-fold scheme (e.g. leave-one-in, or a simpler train/validate split within the 3) before coding this up.

---

## 14. LOSO (Leave-One-Subject-Out) validation plan — supporting claim (domain generalization)

**Mapping:** "subject" = machine ID/case, not a person.

With 4 ToyCar cases: Fold 1 trains (normal-only) on cases 2,3,4 → tests on case 1 (normal + anomalous, unseen in training). Repeat rotating the held-out case across all 4 folds. Same structure separately for ToyTrain.

**Two variants — both to be reported (not one-or-the-other):**
- **Within-type LOSO**: train on 3 ToyCar cases, test on the 4th ToyCar case. Tests generalization across *units of the same machine type* — matches DCASE's "domain generalization" framing. **Confirmed (resolves Section 17 item 4): published cross-ID numbers for this exist in the DCASE literature.** This variant's value is now mainly as a **sanity check / re-measurement under MCU deployment constraints**, not a novel generalization finding on its own — frame it accordingly in the writeup.
- **Cross-type LOSO**: train on all ToyCar cases, test on ToyTrain (or vice versa). Tests generalization across *different machine types* — matches DCASE's harder "first-shot" framing. **Confirmed: no published cross-type numbers found for this dataset.** This is the **higher-priority, more novel variant** if time forces a choice between the two — it's the one actually establishing a new result rather than reproducing a known one.
- Reporting both separately is more informative than picking one — "generalizes across units but fails across types" vs. "generalizes across both" are different, useful findings. **CONFIRMED: report both variants.**

**Hard methodological commitment (already settled, do not revisit):** train ONLY on normal recordings in every fold. Anomalous recordings are for evaluation/scoring only. Do NOT fine-tune on anomalous audio even from the "source" case(s) — this breaks from the unsupervised convention the rest of the field uses, and there's direct empirical counter-evidence: in the DCASE 2025 first-shot evaluation, large fine-tuned SSL systems trained with additional data actually *underperformed* training-free/unsupervised methods (different machine-label definitions and input mixtures between pretraining and evaluation created a mismatch that hurt fine-tuned performance).

**Small-N caveat:** 4 folds per machine type is a small sample of generalization estimates — report mean ± spread across folds, not just a single averaged number; with this few folds, per-fold variance is itself informative and shouldn't be hidden.

---

## 15. Key terms encountered (glossary, for quick recall)

- **DCASE Task 2** — the annual "Detection and Classification of Acoustic Scenes and Events" challenge's unsupervised anomalous sound detection task; a major benchmark/leaderboard for this field.
- **Domain shift** — when recording conditions (room, load, background noise) differ between training and test data for the *same* machine type. **RESOLVED for this dataset: no domain shift.** Per the ToyCar dataset paper, all recordings were made with a fixed setup — a single physical "mini 4WD" toy car rig, four fixed microphones, and a consistent inspection-device/recording-room arrangement. The 4 cases differ only in which motor/bearing combination is installed on that same rig — i.e. the *machine* varies across cases, but the *recording condition* (room, mic placement, device) does not. This means domain-shift-handling machinery (oversampling, adversarial adaptation) is **not needed for this project** — simplifies scope. (Cases still differ from each other in the underlying sound-generating hardware, which is exactly what LOSO, Section 14, is testing generalization across — that's a separate thing from domain/recording-condition shift.)
- **Self-supervised auxiliary task framing** — training a model on a proxy task solvable with only normal data (e.g. classify machine ID/operating condition), then using the model's confidence/embedding on that task as the anomaly signal. Contrast with reconstruction-error framing (current thesis approach).
- **Knowledge distillation** — training a small "student" model to mimic a large pretrained "teacher" model's outputs/embeddings, rather than training on raw labels alone. Open question for ASD specifically: how much fine-grained sensitivity survives shrinking the student to MCU scale.
- **Reconstruction-error vs. embedding-distance scoring** — two different ways of turning a learned representation into an anomaly score. Autoencoder approach: distance between input and its reconstruction. SSM/self-supervised approach: distance between embedding and a "normal" reference (cluster, classifier confidence).

---

## 16. Reading list / links gathered so far

- Systematic literature mapping (anomaly detection + TinyML/MCU, 2021–2023): https://www.sciencedirect.com/science/article/pii/S2542660524000052
- DCASE 2023 Task 2 challenge writeup: https://arxiv.org/abs/2305.07828
- DCASE 2025 Task 2 challenge writeup: https://arxiv.org/abs/2506.10097
- TinyML for Acoustic Anomaly Detection survey: https://www.academia.edu/145800997/TinyML_for_Acoustic_Anomaly_Detection_in_IoT_Sensor_Networks
- "An Efficient Anomalous Sound Detection System for Microcontrollers" (STM32H747I deployment, multi-point optimization template — feature extractor + predictor + loss + memory-aware RL pruning): https://pmc.ncbi.nlm.nih.gov/articles/PMC11644479/
- "Analysis of Feature Representations for Anomalous Sound Detection" (pretrained cross-domain representations + GMM beating autoencoder baseline): https://arxiv.org/abs/2012.06282
- SSM-based DCASE 2025 paper (Emon & Anon) — found via DCASE 2025 workshop proceedings page: https://dcase.community/workshop2025/proceedings
 — **Confirmed details:** Task = Unsupervised/First-Shot ASD on **MIMII-DG & ToyADMOS2** (same dataset family as this project); ~2.5M–6.5M parameters (lightweight bidirectional SSM stack or hybrid CNN-SSM encoder); anomaly head fuses reconstruction + latent-distance scores (`S_total = α·Norm(S_recon) + (1−α)·Norm(S_latent)`) — see Section 2 and Section 10. Earlier caution about inconsistent proceedings-page text (mixing a Task 3 Track B mention) still applies to any *other* numbers not listed above — verify against the paper PDF directly.
- "Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations" (Yadav & Tan, Interspeech 2024) — cleaner, well-corroborated anchor for the general SSM-for-audio claim if the DCASE paper's details prove unreliable.
- "RawBMamba" (end-to-end bidirectional SSM for audio deepfake detection) — another SSM-for-audio precedent, different task, useful for architecture reference: arXiv:2406.06086
- **MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs** (Xu et al., Sept 2025) — first real Mamba deployment on MCU hardware (**STM32H747XIH6, Arduino Portenta H7**), 83% peak memory reduction via operator fusion, full fp32 precision (no quantization tested), evaluated on KWS/HAR (not ASD). Code confirmed live, MIT license. Single most important prior-art check for this whole direction: https://arxiv.org/abs/2509.05488 — code: `github.com/Whiten-Rock/MambaLite-Micro`
- **ASDNet** ("An Efficient Self-Supervised Convolutional Network for Anomalous Sound Detection") — claims efficiency/embeddability via FLOP/parameter-count comparisons only; **confirmed no actual MCU deployment**. No independent prior art here to build on or compete with (Section 10, Option 2).

---

## 17. Open items / immediate next steps (consolidated)

**Prior-art / reading checks — all resolved:**
1. ~~Read MambaLite-Micro in full~~ — **RESOLVED**: STM32H747XIH6 (Arduino Portenta H7, M7), full fp32 (no quantization), code live under MIT license (Section 3).
2. ~~Locate and read the full SSM DCASE ASD paper~~ — **RESOLVED**: Task = Unsupervised/First-Shot ASD on MIMII-DG & ToyADMOS2 (same family as this project's dataset); ~2.5M–6.5M parameters; anomaly head is a fusion of reconstruction + latent-distance heads (Section 2, Section 10).
3. ~~Check whether ASDNet has been deployed on MCU hardware~~ — **RESOLVED: no**, only FLOP/parameter-count-based efficiency claims. Option 2 (Section 10) has no independent MCU prior art to compete with or build on.
4. ~~Check DCASE baselines for ToyCar/ToyTrain cross-ID/cross-type numbers~~ — **RESOLVED: cross-ID numbers exist; cross-type numbers do not appear to.** Reprioritizes Section 14: within-type LOSO is now a sanity-check/re-measurement, cross-type LOSO is the higher-priority novel result.

**Design decisions — all resolved:**
5. ~~Ablation leakage-risk handling~~ — **RESOLVED: nested validation** (Section 13). Remaining implementation detail: exact inner-fold scheme given only 3 training cases per outer fold.
6. ~~LOSO reporting scope~~ — **RESOLVED: report both within-type and cross-type variants** (Section 14).
7. ~~GPU-first vs. deployment-first sequencing~~ — **RESOLVED: GPU first** (Section 12). This is the natural starting point for implementation work in a new session.
8. ~~Dataset domain-shift check~~ — **RESOLVED: no domain shift.** Confirmed from the ToyCar dataset paper: fixed recording rig/room/mic setup across all 4 cases; only the motor/bearing hardware varies (Section 15). Domain-shift-handling machinery is not needed for this project, which also simplifies Option 2 (Section 10) if pursued later.

**Remaining open items (genuinely still open, not yet actionable without further work):**
9. **Parameter budget for this project's own (smaller, not-copied) SSM configuration** — get a rough size estimate for the smallest candidate configuration (from the Section 13 ablation axes) on each of the three target platforms (Section 4), so the ablation sweep has a concrete "does this variant fit" checkpoint per platform.
10. **Quantization survival** — the single most open technical question left. MambaLite-Micro never tested it (fp32 only), so there's no existing evidence either way for whether the selective-scan recurrence survives int8 quantization (Section 3, Section 11 Point 1). This is likely the first genuinely novel engineering result the project will produce.
11. **Chip spec diff** — confirm how the committed Nucleo-H7S3L8's flash/RAM compares to MambaLite-Micro's actual STM32H747XIH6 before assuming their result transfers directly (Section 3, Section 4).
12. **Fusion-gap framing for the writeup** — since the published SOTA anomaly head fuses reconstruction + latent-distance scores and this project's Option 3 uses latent-distance only, be ready to explain any accuracy gap vs. the published number as an expected, deliberate trade-off (Section 2, Section 10).

**Placeholders in the committed claim (Section 1) waiting on experimental results:** `[X]%` accuracy-vs-GPU figure, and which of `[selective / fixed-parameter]` and `[selectivity / state dimension]` the data actually supports — both resolve once the Section 13 ablations are run, not before.

---

## 18. Implementation phase breakdown — for splitting future sessions

**Rationale:** asking one session for all implementation steps at once (~200+ steps) risks shallow detail as context fills, and doesn't match reality anyway — later phases genuinely can't be made concrete until earlier phases produce results (e.g. GPU-first sequencing, Section 12, means the MCU port has nothing to port until the GPU prototype exists). Split by phase instead, each phase its own session.

1. **GPU prototype phase** — SSM backbone + Option 3 head (Section 10), trained/evaluated on GPU, unquantized. Produces the "ceiling" number (Section 12, item 3). Nothing downstream can be made concrete before this exists.
2. **Ablation + LOSO harness** — mechanism-level ablation (Section 13, nested validation) and both LOSO variants (Section 14), still on GPU. Produces the numbers that resolve the claim's placeholders (Section 1: `[X]%`, selective-vs-fixed). Can reasonably be combined with Phase 1 in one session, since both are sequential GPU work.
3. **Feature pipeline for MCU** — log-mel (Section 7) + skewness/kurtosis (Section 8) implementation, targeting CMSIS-DSP. Decoupled from Phases 1–2 (doesn't depend on which SSM config wins) — can run in parallel, potentially its own session at any time.
4. **SSM backbone MCU port** — adapting the winning GPU config (from Phases 1–2) to a MambaLite-Micro-style deployment, per platform (Section 4), starting with fp32 (matching their proven approach, Section 3) before attempting quantization. Depends on Phase 1–2 results as concrete input.
5. **Quantization + anomaly head/threshold on-device** — the open engineering risk (Section 17, item 10), plus wiring up Option 3's distance computation and all three threshold approaches (Section 9.1) on-device. Depends on Phase 4.
6. **Cross-platform results + writeup framing** — assembling the three-way comparison (Section 12) across all three boards, plus the fusion-gap explanation for the paper (Section 17, item 12). Depends on Phase 5 across all three platforms.

**Handoff mechanic for each new session:** provide (a) this file for context, explicitly noting Sections 6–8 are skippable background, and (b) the previous phase's *concrete output* (code, configs, winning hyperparameters, or numbers) once it exists — not just this plan. A session planning Phase 4 with Phase 1–2's actual winning config in hand will produce a far more specific roadmap than one working from "whatever the ablation determines."

---

**Sections 6, 7, and 8 are theoretical background** and can be skipped when resuming implementation work in a new session — the goal for that session is to turn this plan into a concrete, discrete, byte-sized implementation step list.

*Document generated from an exploratory/Socratic research-scoping conversation. Claim (Section 1), methodology (unsupervised-only, Section 1/14), primary anomaly head (Option 3, Section 10), target platforms (Section 4), ablation validation scheme (nested, Section 13), LOSO reporting scope (both variants, Section 14), and experimental sequencing (GPU-first, Section 12) are all committed. Remaining items (Section 17, points 9–12) are technical unknowns for a new session to resolve through implementation, not open design questions.
