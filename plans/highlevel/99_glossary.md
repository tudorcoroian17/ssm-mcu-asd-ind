# Glossary — living document

**How to use this:** ordered roughly by when you meet each term, grouped by area. When a new unfamiliar term shows up in a session, add it here rather than looking it up twice. Terms marked 🔴 are ones that are easy to confuse with each other — check those pairs carefully.

---

## Training machinery

- **Backbone** — the SSM stack plus pooling; the part that turns a sequence of frames into one embedding vector. Everything before it (log-mel) is fixed maths; everything after it (Option 3) stores a centroid but learns nothing.
- **Head** — a small module attached to the backbone for a specific job. You have two: the *prediction head* (training only, one linear layer) and the *anomaly head* (inference, Option 3's distance computation).
- **Embedding** — the fixed-size vector the backbone produces per clip. `d_model` dimensions.
- 🔴 **Feature vector / frame (`x_t`)** — one column of the log-mel spectrogram, `n_mels` values, describing one short time slice. Observable. **Not** the same as *state*.
- 🔴 **State (`h_t`)** — the recurrent hidden vector *inside* the SSM, `d_state` dimensions, updated at every timestep and never observed from outside. This is what the master doc's Section 1 headline claim is about.
- **Loss function** — a single scalar computed from the model's output that gradient descent minimises. Without one, no training happens at all.
- **Gradient descent** — compute the loss, compute how each weight influenced it, nudge every weight in the direction that lowers it, repeat.
- **Self-supervision** — manufacturing training labels from the data itself rather than needing external annotation. Next-frame prediction is self-supervised: the label for frame `t` is simply frame `t+k`.
- **Autoregressive** — predicting the next element of a sequence from the elements before it.

## Objective design

- **Persistence baseline** — the parameter-free predictor `x̂_{t+1} = x_t`. Borrowed from weather forecasting ("tomorrow will be like today"). The degenerate solution you must beat.
- **Climatology baseline** — the parameter-free predictor "always output the training-set average." Weaker than persistence; roughly equals your data's variance.
- **Skill score** — `1 − mse_model / mse_persistence`. Zero means you tied the copy baseline; negative means you lost to it. The interpretable version of your loss.
- **Degenerate solution** — a minimum of the loss that is easy to reach and teaches the model nothing. Dangerous because the loss curve looks healthy while it happens.
- **Horizon (`k`)** — how many frames ahead you predict.
- **Residual target** — predicting the *change* `x_{t+k} − x_t` rather than the absolute value.
- **Contrastive / InfoNCE** — a loss that asks the model to identify the true continuation among distractors rather than reconstruct it numerically. Held in reserve.
- **Causal** — a computation that at time `t` uses only inputs from `≤ t`. Mandatory here: seeing the future makes next-frame prediction meaningless.

## Architecture

- **Depthwise convolution** — each channel filtered independently rather than mixed. Much cheaper than a standard conv; standard in Mamba blocks.
- **Discretization** — converting the SSM's continuous-time formulation into per-step update factors. **ZOH** (zero-order hold) uses `exp(Δ·A)`; **Euler** uses the cheaper linear approximation `I + Δ·A`. Master doc Section 13 axis 3.
- **`d_model` / `d_state` / `d_inner` / `expand`** — embedding width / recurrent state width / internal width after expansion / the expansion factor between them.
- **`Δ` (delta)** — the per-timestep step size. In a *selective* SSM it is computed from the input, which is how the model decides what to retain vs forget.
- **Selective vs fixed** — whether `Δ`, `B`, `C` are computed per-timestep from the input (Mamba-style) or are fixed learned parameters (S4-style). The single most consequential ablation axis in this project.
- **RMSNorm** — a normalization layer; cheaper cousin of LayerNorm, standard in modern SSM stacks.
- **Pooling** — collapsing a sequence into one vector (mean, max, last, or a concatenation).
- **Half-life (of `Ā`)** — how many frames until the state's memory of an input decays to half strength. Your direct measurement of whether the state is doing anything.

## Evaluation and validation

- **AUC / pAUC** — area under the ROC curve, computed by sweeping the anomaly-score threshold across all values; `pAUC` restricts to the low-false-positive region (`p=0.1`). DCASE convention.
- **LOSO** — leave-one-subject-out; here "subject" = machine case, not a person.
- **Nested validation** — an inner selection loop inside each outer test fold, so configs are chosen without ever seeing the outer test data.
- **Leakage** — any path by which held-out data influences training or model selection, inflating reported results.
- **Centroid** — the mean embedding of normal training clips; Option 3's reference point.
- **Mahalanobis distance** — a distance accounting for correlations between embedding dimensions via the inverse covariance matrix. More informative than Euclidean, far more expensive to store.
- **Shrinkage (Ledoit-Wolf)** — a fix for covariance estimates made from too few samples, pulling them toward a simpler well-behaved matrix.
- **Domain shift** — differing recording conditions between train and test for the same machine type. **Confirmed absent** in this dataset (master doc Section 15).

## Deployment

- **Quantization** — representing weights/activations in low-precision integers (typically int8) instead of float32, to cut memory and compute.
- **PTQ / QAT** — post-training quantization (convert an already-trained model) vs quantization-aware training (simulate quantization during training so the model adapts).
- **Parity test** — feeding identical input to two implementations (PyTorch vs C) and checking outputs match within tolerance. Your only defence against silent porting bugs.
- **CMSIS-DSP / CMSIS-NN** — ARM's optimised signal-processing and neural-network libraries for Cortex-M.
- **FPU** — floating-point unit. Present on H7 and ESP32, **absent on RP2040**, which is why that board needs the log lookup table.
- **Operator fusion** — combining operations so a large intermediate tensor is never materialised in memory. MambaLite-Micro's core trick, worth 83% peak memory reduction.
- **Real-time margin** — cycles available per frame vs cycles consumed. If the feature pipeline alone exceeds the hop duration, streaming is impossible regardless of model cost.

---

## Terms to add as we go

> Added Phase 1 close-out, August 2026.

### Evaluation and Validation

* **Zero-state ablation:** A diagnostic that discards the recurrent state $h$ after every timestep, leaving only the convolution path and skip connection. If skill barely changes, the state was inert. Here it collapses skill from $+0.54$ to $-2.5$, proving the opposite.
* **Coefficient of variation (cv):** Standard deviation divided by the mean. Used to compare dispersion across folds whose absolute scales differ.
* **Condition number:** The ratio of a matrix's largest to smallest singular value. For a covariance matrix, a high value (here, $> 10^6$) indicates near-singularity and an untrustworthy inverse — which disqualified `concat_mean_last` from Mahalanobis scoring.
* **Buried clip:** An anomaly ranked below every normal clip in the test set. With 265 clips per class, each buried clip costs exactly $1/265 \approx 0.0038$ of AUC — the quantization behind several repeated AUC values.
* **Discordant pair:** A $(\text{normal}, \text{anomaly})$ pair the scorer ranks the wrong way round. $(1 - \text{AUC}) \times n_{\text{pos}} \times n_{\text{neg}}$ yields the total count; dividing by $n_{\text{neg}}$ recovers the buried-clip count.
* **Reference set:** The population of normal embeddings a distance head scores against. The default here is training embeddings; validation normals and their union were tested as alternatives.
* 🔴 **Rank fusion vs. z-score fusion:** Combining two heads' scores by averaging their ranks within the test set (scale-free, but transductive and impossible per-clip on an MCU) versus standardizing each against calibration constants (deployable, but breaks when the calibration population differs from the test machine).
* **Threshold methods:**
  * **Percentile:** Derived from validation normals (unsupervised, primary).
  * **Chi-square:** Analytic formulation (Mahalanobis only, cross-check).
  * **Labelled-anomaly-calibrated:** Empirical upper bound; leverages label information a deployment would not possess.
* **Clustered $k\text{NN}$ references:** Training embeddings compressed to $k$ representative points via $k$-means, allowing the head to store $16 \times d_{\text{model}}$ floats instead of the entire training set (here: 4 KB vs. 810 KB, at a cost of 0.017 mean AUC).

---

### Deployment

* **Streaming peak RAM:** The memory an MCU requires when processing one frame at a time: persistent recurrent state and convolution history across all layers, plus transient per-timestep scratch memory. Distinct from what a GPU-parallel PyTorch trace reports, which materializes the entire $(T, d_{\text{inner}}, d_{\text{state}})$ tensor for throughput and does not reflect deployed execution.
* **Flash versus RAM footprint:**
  * **Flash:** Stores model weights (fixed, quantizable).
  * **RAM:** Stores state and activations (per-inference).
  * *Context:* For this project, flash represents the more binding constraint on the leanest target board, whereas RAM is directly driven by the SSM recurrence.
