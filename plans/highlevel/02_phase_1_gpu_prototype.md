# Phase 1 — GPU Prototype

**Prerequisites:** `00_index.md`, `01_design_decisions.md`, dataset on disk.
**Goal:** one reproducible number — full-precision, unconstrained SSM + Option 3 head AUC on ToyCar. This is cell (3) of the master doc Section 12 three-way comparison. Section 12 commits to GPU-first so this number exists before any embedded work starts.
**Can share a session with Phase 2** (master doc Section 18: both are sequential GPU work).
**Status: COMPLETE.** All eight exit-gate items (§1.10) pass. Two items were deliberately deferred to Phase 2 and are flagged inline: ranges.json (§1.5) and the threshold methods (§1.7). The open questions posed throughout this document are answered in the § notes below; the answers live in findings/130_evaluation_of_default_model_across_folds.md.

---

## 1.0 Scaffold and config discipline

Master doc Section 13 commits to **one-at-a-time ablation from a single default config**, which is only mechanically possible if there is literally one default config object.

```
src/
  data/     inventory.py, folds.py
  features/ logmel.py, stats.py, cache.py, baselines.py
  models/   ssm_block.py, backbone.py, heads.py
  eval/     embeddings.py, auc_pauc.py, footprint.py, thresholds.py
  train.py
checks/
  integrity/ parity.py, stability.py, emb_score_analysis.py
  smoke/     decay_half_life.py, zero_state.py, coefficient_of_variation.py,
             embedding_saturation.py
  metrics/   onset_tail_contribution.py, misranked_clips.py,
             score_fusion.py, reference_set_ablation.py
helpers/    reference_scan.py, vis_inspect_model.py, extract_mel_filterbank.py
configs/    default.yaml, mel_filterbank.npy, all_folds_normalisation_stats.json
runs/
  compute_hash.py
  baselines/toycar_all_folds_k2_baselines.json
  case{N}/<config_hash>/
    ckpt.pt, ckpt_descriptor.json
    embeddings/{emb_<pooling>.npz, manifest.csv}
    scores/scores_<pooling>_<head>.npz
    eval/{results.json, results.csv, *.png}
    analysis/analysis_<pooling>_<head>.png
    decay_half_life.csv, half_life_histogram.png, zero_state.csv, footprint.csv
archive/SEED_{42,158,824}/case{N}/
```

**Deviations from the Original Scaffold (Deliberate)**

- `dataset.py` was never needed — `ClipDataset` is six lines inside `train.py`.
- `eval/metrics.py` and `eval/diagnostics.py` became `eval/auc_pauc.py` and the `checks/` tree.
- **Run directory structure:** `case{N}/<config_hash>/` rather than a flat `<run_id>/`, allowing folds to group visibly. `config.yaml` is not copied in; `ckpt_descriptor.json` embeds the configuration verbatim instead, serving the same purpose with one fewer file.
- **Embeddings format:** `embeddings.npy` became `embeddings/emb_<pooling>.npz`, holding `train`, `val-normal`, `val-anomaly`, and `test` embeddings alongside labels and normalization statistics. This enables all downstream analysis to remain pure NumPy without requiring model reloads.
- `ranges.json` is not yet produced (see §1.5).
- Single dataclass, serialized into every run directory verbatim.
- **Run ID = hash of config, seed deliberately excluded**. runs/compute_hash.py:train_config_hash() hashes held_out_case, training, features, and model. Seed is not an ablation axis for this project and the working seed is fixed at 158. The three-seed sweep was archived by hand to archive/SEED_*/ between runs. See 00_index.md amendment 7 for the hazard this creates if a future phase varies seed programmatically.
- **Reproducibility verified**. Same config and seed reproduces train and val skill to within 0.001. Exit gate item 2.
- Seed `torch`, `numpy`, `random`; set `torch.backends.cudnn.deterministic`.

Phase 2 runs 300+ configurations. Results that cannot be traced to an exact config are worthless, and re-running to find out is expensive.

---

## 1.1 Dataset inventory

Write `inventory.py` → `manifest.csv`:

| column | notes |
|---|---|
| `path` | absolute |
| `machine_type` | ToyCar / ToyTrain |
| `case_id` | 1–4 |
| `label` | normal / anomaly |
| `sample_rate` | **verify, do not assume** |
| `n_samples`, `duration_s` | |
| `n_channels` | ToyADMOS2 raw is multi-channel; DCASE redistributions usually are not |

**Verification gate — do not proceed until you can state:**

1. **Uniform sample rate?** If not, resampling becomes a pipeline step *and* a Phase 3 MCU constraint.
2. **Uniform clip duration?** Non-uniform means variable sequence length `T`, changing batching and the MCU RAM budget.
3. **Channel count.** If multi-channel, the channel policy must be reproducible in Phase 3 on a board with one microphone.
4. **Clip counts per `(machine_type, case_id, label)`.** Anomalies are usually far fewer than normals. The ratio determines whether per-fold AUC is stable enough for master doc Section 14's mean±spread reporting.

**Question:** for the smallest case, how many anomalous clips are there? If under ~50, what does that do to the confidence interval on a per-fold AUC, and does it change how you write the Section 14 small-N caveat?

---

## 1.2 Log-mel extraction — with a Phase 3 contract

Master doc Section 7 calls this the easy part. It is easy *to compute* and easy *to get subtly wrong in a way that only surfaces in Phase 3*, when MCU output does not match GPU output and you cannot tell whether it is a bug or quantization.

Parameters to freeze now — they become a **Phase 3 hard contract**:

| param | constraint |
|---|---|
| `sr` | from inventory |
| `n_fft` | **power of two** — CMSIS-DSP `arm_rfft_f32` requires it |
| `hop_length` | drives sequence length `T` and the MCU real-time budget |
| `n_mels` | 64 default (master doc Section 8 assumes N=64 for the stats path) |
| `f_min`, `f_max` | |
| mel scale | **htk vs slaney** — these give *different filterbanks* |
| mel norm | slaney-normalized or not |
| power | magnitude (1.0) vs power (2.0) |
| log | `log(x + eps)` vs `10·log10` (dB) — pick one, record `eps` |

**The trap, stated explicitly.** The Phase 3 handoff artifact is **not this parameter table.** It is the *materialized mel filterbank matrix*, exported as `.npy` and later as a C array. Two libraries given identical parameters produce different filterbanks (edge handling, normalization, area- vs peak-normalized triangles). If Phase 3 recomputes the filterbank from parameters instead of embedding the exported matrix, you will lose days to a parity bug that is not a bug.

**Normalization — leakage trap.** Per-mel-bin mean/std must come from **training-fold normal data only**, recomputed inside every LOSO fold. Global statistics over the whole dataset (the convenient thing) leak held-out case statistics into training and inflate every number in Section 14.

**Caching.** Write features to `.npy`/memmap keyed by feature-config hash. Phase 2 re-reads these 300+ times; recomputing STFTs each run wastes GPU time for nothing.

---

## 1.2b Baselines — compute these BEFORE any training run exists

No model, no GPU, a few lines of NumPy over cached features. These are what make your training loss interpretable. See `01_design_decisions.md` §3 for why.

```python
# X: (n_clips, T, n_mels), already normalized — SAME UNITS as the training loss.
# k = prediction horizon (default 2).

# Persistence ("copy the current frame") — the degenerate solution to beat
mse_persistence = ((X[:, k:] - X[:, :-k]) ** 2).mean()

# Climatology ("always output the training-set average frame") — ignores time entirely
mu = X_train.mean(axis=(0, 1))
mse_mean = ((X[:, k:] - mu) ** 2).mean()      # ≈ the variance of your data
```

These bracket your model:

| where the model lands | interpretation |
|---|---|
| at or above `mse_mean` | learned nothing whatsoever |
| at `mse_persistence` | learned only "sound is continuous" — **the degenerate solution** |
| well below `mse_persistence` | learned real temporal structure |

**Report a skill score, not raw MSE:**

```
skill = 1 − mse_model / mse_persistence
```

`0` = tied the copy baseline. Negative = lost to it. Positive and growing = learning something real. This is the **primary training-health metric for the whole project** — put it on the training curve next to the loss.

Compute both baselines on a held-out fold, per machine type, write to `baselines.json` before the first training run.

---

## 1.3 The SSM core — hand-written selective scan

**Decision (confirmed):** hand-write the recurrence in plain PyTorch. Use `mamba-ssm` as a **correctness oracle**, not as the implementation.

Why hand-written:

1. **Master doc Section 13 axes 2 and 3 are unreachable otherwise.** Selective-vs-fixed and ZOH-vs-Euler both require editing the discretization and the scan body. The fused CUDA kernel hardcodes selective + ZOH.
2. **Phase 4 ports this exact loop into C.** MambaLite-Micro's contribution is a hand-written runtime-free C engine (master doc Section 3). If you never write the loop in Python, you meet the recurrence for the first time in C, on a board, with no reference to diff against.
3. Learning.

**Validation step — do this once, early.** `mamba-ssm` ships a pure-PyTorch `selective_scan_ref` alongside the CUDA kernel. Diff against **that**, not the kernel — same maths, readable, no compile step. Random input → your scan vs `selective_scan_ref` → `assert torch.allclose(...)`. Then set it aside.

*Limitation:* it validates your **default config only**; it cannot express axes 2 or 3 at all. Installing the CUDA kernel itself (plus `causal-conv1d`) can eat an afternoon and is not required.

**Causality is a hard constraint** — see `01_design_decisions.md` §5.

```python
class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2,
                 selective=True, discretization="zoh", dt_rank=None):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.selective = selective
        self.discretization = discretization

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # GAP 1 — depthwise CAUSAL conv (master doc Section 13, axis 5).
        # Must be depthwise (groups=?) and causal (padding=? then trim what?).
        # NOW LOAD-BEARING: a non-causal conv leaks the future frame you are
        # asking the model to predict, and the loss silently becomes meaningless.
        self.conv = nn.Conv1d(...)

        # GAP 2 — the selectivity switch (Section 13 axis 2).
        # if selective:  delta, B, C are PROJECTED FROM THE INPUT each timestep
        # else:          delta, B, C are fixed learned nn.Parameter
        # What are the shapes in each branch? Which branch's parameter count
        # depends on d_state, and which on both d_state and T?
        if selective:
            self.x_proj = nn.Linear(...)   # -> dt_rank + 2 * d_state
            self.dt_proj = nn.Linear(...)  # -> d_inner
        else:
            self.B = nn.Parameter(...)
            self.C = nn.Parameter(...)
            self.dt = nn.Parameter(...)

        # GAP 3 — A stored as A_log, not A. Why?
        # (Section 6.2: A controls persistence vs decay. What must be true of
        #  A's eigenvalues for the state not to blow up over 300 timesteps,
        #  and how does log-parameterization enforce it structurally?)
        self.A_log = nn.Parameter(...)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def discretize(self, delta, A, B):
        # GAP 4 — Section 13 axis 3, and the axis most relevant to
        # quantization stability (Section 11 point 2).
        # ZOH:   A_bar = exp(delta * A);   B_bar = ?
        # EULER: A_bar = I + delta * A;    B_bar = ?
        # Which involves a transcendental function, and what does that cost
        # on an MCU with no FPU (Section 4, RP2040)?
        ...

    def forward(self, x):           # x: (batch, T, d_model)
        # GAP 5 — THE SCAN. This loop becomes C in Phase 4.
        #   h = zeros(batch, d_inner, d_state)
        #   for t in range(T):
        #       h = A_bar[:, t] * h + B_bar[:, t] * u[:, t]
        #       y[t] = (h * C[:, t]).sum(dim=-1)
        # Then answer: which tensors here are materialized at full
        # (batch, T, d_inner, d_state) size, and which of those is the one
        # MambaLite-Micro's operator fusion refuses to materialize?
        # That answer is your Phase 4 starting point.
        ...
```

**Stability check before training anything.** Run a randomly-initialised block on 300 steps of noise, print `h.abs().max()` per timestep. Unbounded growth means GAP 3 is wrong. Minutes now, days later.

---

## 1.4 Backbone assembly — two output modes

Stack `L` blocks with residual connections and pre-norm RMSNorm.

**The backbone needs two output modes**, because of the training objective:

| mode | output | used by |
|---|---|---|
| **sequence** | per-frame outputs, `(batch, T, d_model)` | training — the prediction head attaches here |
| **pooled** | one vector, `(batch, d_model)` | inference — the Option 3 distance head |

The prediction head **cannot** attach to the pooled embedding: pooling collapses time, so there is no "next frame" left to predict. Return the sequence and let the caller pool.

**Pooling choice — a real design decision, not a default.** Candidates: final hidden state, mean over time, max over time, `concat(mean, last)`.

**Question before you pick:** the anomaly is a bearing/motor defect present throughout the clip in some cases, and a transient click in others (master doc Section 8.1's "spike/click vs steady hum"). Which pooling preserves a transient, and which averages it into invisibility? Does the answer change for a 10-second clip?

**Answered (findings/130 §2). mean.** The reasoning in the question — that max should preserve transients that mean averages away — turned out to be the wrong frame, because the deciding factor was stability rather than sensitivity. max pooling on case1 varies by 0.10–0.13 AUC standard deviation across three seeds, swinging from 0.9647 down to 0.7123 on the same configuration. A single-seed measurement made it look like case1's best choice; it is not a good configuration that underperformed once, it is unstable. concat_mean_last produces covariance condition numbers of 6.6e6–1.3e8 in every fold at every seed and scores below chance on case2 under Mahalanobis. mean is the only pooling mode that is stable across seeds on all four folds, and it is also the cheapest.

---

## 1.5 Training objective — full spec

```
input:   x_1 ... x_T          normalized log-mel frames, normal recordings only
target:  x_{t+k} − x_t        residual target, default k = 2
head:    Linear(d_model → n_mels), attached to the SEQUENCE output
loss:    MSE over valid timesteps (t = 1 … T−k)
metric:  skill = 1 − mse_model / mse_persistence
```

Fixed commitments:

- **Normal recordings only, every fold.** Master doc Section 14, non-revisitable.
- **Residual target** and **`k ≥ 2`** per `01_design_decisions.md` §3. Run one cheap pilot on a single fold comparing residual vs absolute target before the ablation — if residual wins as expected, fix it and do not spend a nested-validation axis on it.
- **Log per-tensor activation min/max/percentiles** into `runs/<id>/ranges.json`. This feels premature. It is not — Phase 5 cannot diagnose quantization failures without it. **Done** — `checks/smoke/activation_ranges.py`, all four folds. See `findings/150_ranges_and_activations_for_quantization.md`, which also surfaced findings Phase 5 will want going in: a per-layer activation-range blowup that argues against a shared quantization scale, and a dynamic-range problem in `delta` specific enough to need per-channel or log-scale handling.
- **Early stopping** on a validation signal that never touches the outer held-out case. Decide the split now.
- The prediction head is **dropped at inference** for the distance-only configuration, and **retained** for the fused configuration (`01_design_decisions.md` §4). Keep the checkpoint for both.

**Sanity gate.** If `skill ≤ 0` after training, stop and diagnose before proceeding. Likely causes in order:
1. `k` too small — still copyable.
2. Non-causal leak in GAP 1 — an artificially easy task that then collapsed.
3. Learning rate.
4. **Normalization mismatch between the baseline computation and the training loss** — they must be in identical units. Easiest one to get wrong, hardest to spot.

---

## 1.6 Option 3 head — fitting "normal"

Fit on training-fold normal embeddings only. Four variants, ordered by on-device cost:

| variant | stored state | MCU cost at d_model=64 |
|---|---|---|
| centroid + Euclidean | 1 vector | 64 floats = 256 B |
| centroid + Mahalanobis | vector + inverse covariance | 64×64 = **16 KB** |
| GMM (k components) | k × (mean, cov) | k × above |
| k-NN | k reference embeddings | k × 256 B |

**Question:** Mahalanobis needs a covariance estimated from your training-fold normal embeddings. How many normal clips per fold, and is that enough to estimate a 64×64 covariance without it being singular? If not — shrinkage (Ledoit-Wolf), a diagonal approximation, or a smaller `d_model`? Note the third also helps Phase 5.

**Answered** (findings/130 §2.4, §13.4). With 3,240 training clips against mean pooling's 64 dimensions, the covariance is well-conditioned — 1.5e4 to 7.5e4 across folds, comfortably below the 1e6 instability threshold. No shrinkage needed. concat_mean_last is a different story: its 128-dimensional embedding pairs two correlated halves derived from the same sequence, and every fold exceeds 1e6 at every seed (6.6e6–1.3e8), with case2 scoring below chance. Recommendation: drop concat_mean_last + mahalanobis from the evaluation grid entirely. Reducing the reference set to validation normals only (810 clips) raises conditioning in all four folds but does not cross the threshold under mean pooling; untested under concat_mean_last.

---

## 1.7 Evaluation

- **AUC and pAUC (p=0.1)** — DCASE convention; master doc Section 9.1 commits to these as headline regardless of thresholding.
- **Per-case results, never a single averaged number** (Section 14 small-N caveat).
- Implement all three Section 9.1 threshold methods now: percentile-based (primary), chi-square analytic (cross-check), labeled-anomaly-calibrated (upper bound, flagged in writeup).

Status: complete. AUC and pAUC (src/eval/auc_pauc.py) reported per case, never averaged. Threshold methods and secondary metrics (src/eval/thresholds.py) done for all four folds — see findings/140_thresholds_and_secondary_metrics.md. The chi-square method applies to Mahalanobis only, as originally scoped; the same-machine calibration finding in findings/140 §4 supersedes 01_eval_spec.md §10's original framing with the full numeric picture and a fourth threshold method.

---

## 1.8 GPU autoencoder baseline — DEFERRED

Moved to Phase 6, after everything else is complete (your decision).

**Consequence, recorded so it does not go quiet:** the Phase 1 exit number is **uninterpretable in isolation**. AUC 0.83 is either strong or mediocre and you will not know which until the baseline exists.

*Interim mitigation:* use published DCASE ToyCar baseline AUCs as a rough anchor, with the explicit caveat that their preprocessing will not match yours exactly. **Do not put an unanchored number in any draft.**

---

## 1.9 State-utilisation diagnostics — protecting the headline claim

**The concern:** the model could beat persistence using **only the depthwise conv**, which sees a 4-frame window, leaving the recurrent state `h_t` inert. Master doc Section 1 names state dimension as a primary architectural lever — if the state does nothing, that lever moves nothing and the headline claim quietly dies.

Beating persistence proves the model learned *something*. It does not prove the learning lives in the state.

Three checks, cheapest first:

**(a) `d_state` sweep — free, you are building it anyway.** Master doc Section 13 axis 1 already varies N ∈ {8, 16, 32, 64}. If skill score and AUC are **flat** across all four, the state contributes nothing. That flatness *is* the diagnostic.

**(b) Zero the state at inference.** Force `h_t = 0` before every timestep so the recurrence cannot carry information forward — only the conv path and skip connection survive. Compare skill with and without. Barely changes → the state was inert. ~10 lines, no retraining.

**(c) Read decay time constants off `A` — pure inspection, no forward pass.** After training, compute `Ā = exp(Δ·A)` per channel. Each value is a per-step decay factor, so:

```
half_life_frames = ln(0.5) / ln(Ā)
half_life_ms     = half_life_frames × hop_length / sr × 1000
```

Half-lives clustered at one or two frames → the model forgets immediately and is functionally a short convolution wearing an SSM costume. A meaningful fraction at tens or hundreds of frames → the state genuinely integrates over time.

**Run (c) after your first successful training run.** Highest information-per-minute check in the project, and it produces *direct mechanistic evidence* for the lever Section 1 claims exists — a much stronger paper sentence than a correlation across an ablation sweep.

### Result: Both (b) and (c) run on all four IND folds, both pass decisively.

| Case | Skill, real state | Skill, state zeroed | Half-life max, any layer |
| :---: | :---: | :---: | :---: |
| **1** | 0.533–0.545 | −2.31 to −2.50 | 296–14,985 ms |
| **2** | 0.539–0.555 | −2.90 to −3.25 | 692–6,185 ms |
| **3** | 0.538–0.544 | −2.30 to −2.56 | 710–3,913 ms |
| **4** | 0.534–0.549 | −2.57 to −2.76 | 626–102,561 ms |

- **Zeroing the state:** Does not merely erase the gain — it drives error to roughly $3.5\times$ the persistence baseline. That is stronger evidence than a null result would have been: the model is not merely helped by the recurrence, it actively depends on it.
- **Effective memory horizon:** Every layer in every fold shows half-life tails far beyond the convolution's 128 ms reach.
- **Mechanistic takeaway:** Master doc **§1**'s state-dimension lever is real, providing the exact mechanistic validation required for the paper.
- **Independent corroboration:** Diagnostic (a), the `d_state` sweep, is still free from Phase 2 axis 1 and will corroborate independently.

**Artifacts:** `checks/smoke/zero_state.py`, `checks/smoke/decay_half_life.py`; outputs stored in each respective run directory.

---

## 1.10 Exit gate

Do not start Phase 2 until **all** of:

1. A single ToyCar fold trains end-to-end and produces AUC + pAUC.
2. Same config + seed reproduces identical metrics.
3. State norm bounded across all 300 timesteps.
4. `selective=False` runs without error (Phase 2 axis 2 depends on it).
5. Your scan matches `selective_scan_ref` under `allclose` for the default config.
6. **`skill > 0`** — the model beats the persistence baseline.
7. **At least one state-utilisation diagnostic passes** (1.9).
8. **Measured wall-clock for one training run is recorded.** This single number decides whether Phase 2's nested validation is affordable as designed.

### All Eight Pass

- [x] **End-to-end training & metrics:** All four ToyCar folds train end-to-end and produce AUC + pAUC.
- [x] **Reproducibility:** Same config + seed reproduces train and val skill to within 0.001.
- [x] **Numerical stability:** `checks/integrity/stability.py` — 12 configurations, bounded state over 1,000 timesteps (beyond the 300 required).
- [x] **Non-selective mode:** `selective=False` runs cleanly in both `stability.py` and `parity.py`.
- [x] **Scan parity:** `checks/integrity/parity.py` — $2.3 \times 10^{-10}$ max absolute difference against `selective_scan_ref`, five orders tighter than the $1.7 \times 10^{-5}$ target (Python-to-Python, not cross-language).
- [x] **Skill convergence:** Train skill 0.540–0.550 across folds; val skill within 0.003 of train.
- [x] **Mechanistic validation:** Both (b) and (c) pass on all four folds (see §1.9).
- [x] **Training throughput:** ~80 s/epoch, $40 \pm 10$ epochs/fold (~53 min typical, ~77 min worst) on GTX 1660 Ti.

---

## Handoff manifest → Phase 2

- `manifest.csv` (project root); fold definitions in `src/data/folds.py`
- Frozen `configs/default.yaml` (seed 158)
- `configs/mel_filterbank.npy` — materialized, not parameters
- `runs/baselines/toycar_all_folds_k2_baselines.json` — `mse_persistence`, `mse_climatology` per fold
- Working `SSMBlock` with both selectivity branches and both discretizations, parity- and stability-checked
- `runs/case1/4f3ffab6e66e/` as the run-directory format template
- **Decay-time-constant histograms:** `runs/case{N}/<hash>/half_life_histogram.png`, plus `decay_half_life.csv`
- **Measured per-run wall-clock:** ~53 min/fold (§1.10 item 8)
- **Additional (beyond the original manifest):**
  - Persisted embeddings and scores per pooling $\times$ head
  - Memory-footprint estimate (`footprint.csv`)
  - Three-seed archive under `archive/SEED_*/`
- **`ranges.json`** (§1.5) — `runs/case{N}/<hash>/ranges.json`, all four folds. See `findings/150_ranges_and_activations_for_quantization.md`.
- **Threshold-method outputs** (§1.7) — `runs/case{N}/<hash>/{threshold_metrics, thresholds_same_machine, threshold_diagnostics}.json`, all four folds. See `findings/140_thresholds_and_secondary_metrics.md`.
