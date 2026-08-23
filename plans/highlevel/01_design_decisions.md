# Design Decisions — What Trains the Backbone

**Read before every phase.** Short, but load-bearing: Phases 1, 2, and 5 all depend on it.

---

## 1. The problem that was nearly missed

Master doc Section 10, Option 3, says the distance head needs *"often no extra training beyond the backbone itself."* That is true of the **head**. It says nothing about the **backbone**.

With Option 1 (reconstruction) excluded, Option 2 (classifier) deferred, and no pretrained encoder loaded, nothing in the original plan produced a **loss function** — the single scalar that gradient descent minimises. Feed 1,000 normal clips through a randomly-initialised SSM with no loss defined and *nothing changes*: no error signal, no gradient, no weight update. Option 3 would have been measuring distances between random projections.

**Why it was invisible.** In the thesis autoencoder, one quantity did two jobs: reconstruction error was both the training loss *and* the anomaly score. Because they were the same number, "train on normal data" really was the whole answer. Section 10 excluded the decoder to save memory — but the decoder was not only producing the score, it was producing the **training signal**. Deleting it deleted both.

---

## 2. The resolution — autoregressive next-frame prediction

**Train the backbone to predict the next log-mel frame from the running state.**

Why this over the alternatives:

- **Fully self-supervised.** The label is manufactured from the data itself (frame `t+k` is the target for frame `t`), so no anomaly labels are needed and master doc Section 14's unsupervised commitment is untouched.
- **Enormous supervision density.** Every clip yields `T−1` targets rather than one. ~1,000 clips × ~300 frames ≈ **300,000 supervised examples** from data already on disk.
- **Architecturally native.** Master doc Section 6.2: the state at time *t* is a compressed summary of everything before *t*. "Predict what comes next from that summary" is the most natural possible use of it.
- **Beats the alternative on fold size.** The auxiliary-machine-ID option would be a 3-way classification over 3 cases inside a LOSO fold — a very weak, possibly trivially-solvable signal.
- **Cheap head.** One linear layer, `d_model → n_mels`, dropped at inference (or retained — see §4).

---

## 3. The failure mode that comes with it — the degenerate solution

**This failure is silent.** The loss drops, the curve flattens, everything looks healthy, and the model has learned nothing.

Concretely, at 16 kHz with a 32 ms window (512 samples) and a 16 ms hop (256 samples): frame `t` covers samples 0–511, frame `t+1` covers samples 256–767. **Half the raw audio in those two windows is literally the same samples.** The mel filterbank then averages 257 FFT bins down to 64, smoothing further; the log compresses further still. Add the physics — a toy motor's spectrum barely changes over 16 ms — and consecutive frames are near-duplicates (expect per-bin correlation > 0.95).

So the laziest possible predictor is:

> **Persistence baseline:** `x̂_{t+1} = x_t` — copy the current frame.

No parameters, no training, and it already achieves low MSE. Gradient descent finds it almost immediately because it is the easiest available minimum. A model that converged there has learned an identity mapping, not machine-sound structure, and its embeddings are near-worthless for Option 3.

### Mitigations adopted

Three axes were available. Two are adopted.

| axis | change | why it defeats the copy solution | cost |
|---|---|---|---|
| **Horizon** ✅ | predict `x_{t+k}`, default `k=2` | at hop 256 / window 512, `k=2` means the two windows share **zero raw samples** | too large a `k` makes the target genuinely unpredictable (machine noise is partly stochastic) and the gradient degrades — hence `k` becomes an ablation axis |
| **Target** ✅ | predict the **residual** `x_{t+k} − x_t` | the copy solution becomes "output all zeros", earning no credit; all remaining error requires real understanding | ~one line. Bonus: residuals have narrower dynamic range than raw frames, mildly helpful for Phase 5 quantization |
| **Loss type** ❌ | contrastive / InfoNCE — identify the true future frame among distractors | copying gives no discriminative advantage, so the degenerate solution is structurally unavailable | negative sampling, temperature hyperparameter, projection head, more silent failure modes. **Held in reserve** if skill scores stay near zero |

---

## 4. Consequence — the fusion gap becomes an asset

Master doc Section 17 item 12 worries that Option 3 is latent-distance-only while the published DCASE head fuses reconstruction + latent scores, and that any accuracy gap would need apologising for.

But **per-frame prediction error *is* an autoregressive reconstruction score.** You now get `S_recon` essentially free, from a head you are training anyway, at a fraction of Option 1's cost — no mirrored SSM decoder, no second pass over the sequence.

This converts item 12 from an apology into an **ablation**: distance-only vs. fused `S_total = α·Norm(S_recon) + (1−α)·Norm(S_latent)`, with the on-device memory cost of each measured. Scope added to Phase 5.

---

## 5. Consequence — causality is now a hard constraint

Because the objective is next-frame prediction, any block that sees future frames makes the task trivially cheatable and the loss meaningless. Therefore:

- the depthwise conv **must** be causal (`GAP 1` in Phase 1 is load-bearing, not cosmetic);
- **bidirectionality is forbidden outright**, not merely deprioritized as master doc Section 13 currently has it.

Convenient alignment: Section 13 deprioritized bidirectionality anyway because it is not causally deployable on an MCU. Training objective and deployment target now agree — worth one sentence in the paper.

## 6. Dataset sequencing — ToyCar first, ToyTrain deferred

**Decision:** implement and validate the full pipeline — GPU prototype through MCU deployment — on ToyCar only. ToyTrain, and therefore cross-type LOSO, is deferred until an SSM configuration has actually been evaluated on MCU hardware.

**Why this isn't a scope cut.** Master doc Section 1 already commits to a priority order: deployment feasibility is the *primary* spine, cross-machine generalization is a *supporting* claim. This decision takes that ordering seriously as a sequencing rule, not only a rhetorical framing. Everything generalization-related — cross-type LOSO in particular — depends on a config existing to generalize *from*. Deployability is the harder, more novel unknown (Section 2's "hard gap" framing) and the one nothing else in the paper can substitute for; there is no equivalent fallback if it doesn't pan out. Generalization work spent before that question is answered risks being built on a config that deployment later forces you to abandon.

**What this changes in practice, immediately:**

- **Phase 1 and Phase 2 run entirely on ToyCar.** `get_fold()`, `manifest.csv`, and `fold_norm_stats.json` are already correctly scoped to ToyCar only — this decision requires no code changes, only documentation ones (Phase 2's run-count table and exit gate).
- **Phase 2's near-term scope is within-type LOSO on ToyCar alone**, not the full within-and-cross-type sweep across both machine types. This materially changes Phase 2 §2.2's run-count budgeting — see that file for the corrected arithmetic.
- **Cross-type LOSO (master doc Section 14) moves later** — scheduled after an MCU deployment result exists, not treated as "if time allows" inside Phase 2. It becomes its own checkpoint once ToyTrain is brought in, most likely sitting after Phase 5.

**What doesn't change.** Section 1's claim placeholders (`[X]%`, `[selective / fixed-parameter]`) resolve entirely from ToyCar's state-dimension and selective-vs-fixed axes — both within-type ablations on a single machine type. Nothing about the headline claim is blocked by this deferral; it's purely the *supporting* claim (generalization across machine types) that waits.

**When to revisit.** Once an SSM configuration has a measured MCU deployment result (Phase 4/5 territory), bring in ToyTrain: a second manifest, a `machine_type` parameter threaded through `get_fold()`, and resumption of master doc Section 14's original both-variants LOSO plan.

---

## 7. Data source: CNT dropped, IND-only adopted

**Decision:** train, validate, and test exclusively on IND clips. CNT (`NormalSound_CNT`,
continuous 10-minute-chunked recordings) is dropped from the pipeline entirely — not filtered
further, not stitched, not used at all. This is a full restart of the data path
(`ssm-mcu-asd-ind`), keeping only the backbone/head modules (§1–§6 above, and `src/models/`)
unchanged from the CNT-era project.

**Status of §1–§6 above:** unaffected. The training objective, degenerate-solution mitigations,
fusion-gap consequence, causality constraint, and ToyCar-first sequencing were never
CNT-specific — they apply identically to IND. Nothing here revises them; it only replaces
what data those objectives are trained against.

### 7.1 Why CNT was chosen in the first place

`01_eval_spec.md` §1 originally justified CNT over IND on volume grounds alone: raw continuous
recordings yield roughly 9x more unique normal audio per case than IND's fixed
one-clip-per-run format. More windows, denser supervision, cheaper to reach a large training set.

### 7.2 What went wrong

A manifest-wide RMS scan found 263 of 7,130 files near-silent (RMS < 0.001) — dead air, not
signal. 82% of them (217/263) sat inside the *training* pool, and unevenly: case3 (87 files)
and case4 (77 files) accounted for three-quarters of the contamination, versus ~26 each for
case1/case2. Sequence-number clustering on case3/case4 showed the silent files interleaved at
a near-constant stride through the session (not random dropout, not one contiguous dead
stretch) — consistent with the CNT recording protocol's motor duty cycle plus independent
file-cutting, not a hardware fault, but still real dead signal every LOSO fold's training pool
inherited.

A window-level fix was designed and implemented — `compute_raw_frame_rms` /
`find_valid_windows`, threshold `raw_silence_rms_threshold = 0.0001`, requiring most frames in
a window to clear the floor (`min_frame_fraction`) rather than dropping whole files, since some
files were silent-then-real and file-level exclusion would have thrown away good tail audio. It
worked (`110_dataset_fold_stats.md`, `115_dataset_audio.md`) and materially improved the
CNT-trained model's eval numbers (AUC 0.235→0.807 on the pre-Mahalanobis baseline).

**Separately, a training-stability problem surfaced that looked CNT-related but wasn't.**
`val_mse` oscillated erratically on a fixed, deterministic val set even after the silence fix.
The leading hypothesis was that windows straddling CNT's motor start/stop transitions (allowed
through by the 0.9 alive-frame threshold) were injecting high-variance gradients — plans
`130_cnt_restitch_cached.md` / `131_cnt_restitch_from_raw.md` proposed reconstructing those
transitions inside CNT the way IND already has them. A cheap diagnostic
(`quick_strict_filter_check.py`, tightening the filter to `fraction=1.0`) returned a clean
negative result: window count barely moved (9,246→9,176, 0.76%), and `val_mse` was equally
erratic. Mid-window transitions were already rare — the restitching plans were shelved, and the
real cause turned out to be optimizer step size through a flat loss region, fixed independently
with AdamW + weight decay, gradient clipping, and `ReduceLROnPlateau`. That fix is orthogonal to
this data-source decision and should be re-applied to `train.py` here regardless of CNT vs IND. 
However, `train.py` should first start with the default options (Adam optimizer, no weight decay
or gradient clipping and no `ReduceLROnPlateau`), unless `val_mse` continues to behave erratically
or another decision is made to reintroduce these.

### 7.3 The experiment that decided it

With the silence problem understood but CNT's fix requiring per-case thresholds, regenerated
normalization stats, and a full retrain of every completed run, IND-only was tested directly as
an alternative: same held-out case, same test protocol, no filtering needed (IND's onset/tail
structure is a designed bracket, not noise — every clip is a full valid window by construction).

Held-out-1, single fold, after correcting two population-mismatch bugs in the eval script
(scoring against the CNT fold/stats instead of the IND ones the checkpoint actually trained on;
briefly re-applying the CNT silence filter to IND clips by accident — worth remembering if this
pool design gets reimplemented, since IND should never pass through a silence filter at all):

| | CNT-filtered (held-out-1) | IND-only (held-out-1) |
|---|---|---|
| train skill | 0.5040 | 0.5482 |
| val skill | 0.1346 | 0.5452 |
| **train − val gap** | **0.369** | **0.003** |
| mse_persistence / mse_climatology | 0.304 | 0.071 |

The train/val gap collapsing to near-zero is the headline result — train and val are now drawn
from the same kind of clip (same onset/offset bracket, same duration), so there's no systematic
pattern val contains that train never saw. AUC came out competitive-to-better than the best CNT
result, though not on a strictly matched metric — the CNT comparison figure (AUC 0.929, pAUC
0.692) used mean pooling + **Mahalanobis** distance; the IND-only run was only scored with
centroid/kNN + **Euclidean** distance, since Mahalanobis wasn't re-run on IND before the
decision was made:

| pooling | AUC | pAUC(p=0.1) |
|---|---|---|
| mean | 0.9257 | 0.8429 |
| max | 0.9594 | 0.8450 |
| concat_mean_last | 0.9201 | 0.8061 |

**Re-running mean+Mahalanobis on an IND-only fold is still an open item** — the table above is
Euclidean-only, so it likely *understates* IND-only's ceiling rather than overstates it, but it
hasn't been measured.

### 7.4 The tradeoff, stated honestly

- **Volume drops.** ~3,240 IND training clips (3 cases × ~1,350 normal clips × 80%) vs.
  9,246–11,425 CNT-filtered windows per fold.
- **Less signal per frame.** IND is ~4x more predictable frame-to-frame than filtered CNT by
  the persistence-ratio measure above — a real risk that some of the small train/val gap
  reflects "less to learn" rather than purely "learned it well."
- **Protocol-memorization risk.** Every IND clip shares near-identical onset timing (0.816s ±
  0.072s) and duration. The model could in principle learn "sound starts at frame 25, stops at
  frame 330" rather than the engine itself — the same failure family as the persistence
  baseline in §3, just at a coarser timescale. Nothing has ruled this out yet; worth an
  explicit diagnostic once this project has its own trained checkpoint (e.g. check whether
  embeddings still separate normal/anomaly when onset/offset frames are masked out).

### 7.5 What carries over unchanged, and what doesn't

**Unchanged:** backbone architecture (`SSMBackbone`, `SSMBlock`), training objective (§2),
degenerate-solution mitigations (§3), fusion-gap framing (§4), causality constraint (§5),
ToyCar-first sequencing (§6).

**Superseded, kept here as record:** all CNT-specific plumbing — `is_dead_frame`/
`min_frame_fraction` filtering, the case-specific threshold question, `130`/`131` restitching
plans, CNT-derived `fold_norm_stats.json`. None of it is wrong, none of it gets deleted from
`ssm-mcu-asd` (still there read-only for cross-checking), but none of it is part of this
project's pipeline.

**New, to be implemented fresh in this project:** a fold function in the shape of the old
`get_fold_ind_only()` — except here it's simply *the* fold function, not a variant — plus a
`compute_normalization_stats_ind()`-style stats function (plain two-pass Welford, no filtering,
since every IND frame is real signal by construction). Both were fully drafted in the prior
project's exploration session and are ready to hand over for transcription when we get to that step.