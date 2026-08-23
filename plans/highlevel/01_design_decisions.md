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