# Finding: IND clips have a consistent, protocol-driven start/stop silence bracket

**Status:** Resolved — mechanism confirmed against the source paper, quantitatively verified.
**Context:** discovered while investigating unexpectedly poor `val_baseline` statistics
(`mse_climatology`, `mu_frame_norm`) during the held-out-1 retrain on silence-filtered CNT data.
This is a *different* phenomenon from the CNT duty-cycle silence documented in
`110_dataset_fold_stats.md` — see Section 5 for the distinction.

---

## 1. The observation that triggered this

While the held-out-1 retrain was in progress, `val_baseline` (computed on the unfiltered IND
validation pool, as designed) showed statistics far outside the pattern every other baseline
computed this session had shown:

```
{'n_clips': 4050, 'mse_persistence': 0.3923, 'mse_climatology': 5.523, 'mu_frame_norm': 8.927}
```

Every CNT-based fold baseline computed to date had `mse_climatology` close to 1.0 and
`mu_frame_norm` close to 0 (correct behavior of per-fold z-scoring). `mse_climatology=5.523`
and `mu_frame_norm=8.927` are large, systematic deviations, suggesting a substantial fraction
of val frames sit far from the training-derived normalization center.

A direct check confirmed this: **3776 of 4050 (93.2%) val clips would fail the same
`min_frame_fraction=0.9` filter now applied to CNT training windows.**

---

## 2. Investigation — onset and tail, both directly measured

### Onset

Five case2 normal IND files (`0001`–`0005`), checking where real audio first exceeds the
raw-waveform silence threshold:

| file | real audio starts at | overall alive fraction |
|---|---|---|
| 0001 | ~0.77s | 89.2% |
| 0002 | ~0.80s | 88.4% |
| 0003 | ~0.77s | 88.4% |
| 0004 | ~0.77s | 88.4% |
| 0005 | ~0.77s | 89.0% |

Extremely tight — onset delay is essentially fixed, not content-dependent.

### Tail

Ten case2 normal IND files (`0001`–`0010`), checking how long before clip-end real audio stops:

```
0.42, 0.42, 0.51, 0.51, 0.45, 0.35, 0.45, 0.45, 0.48, 0.42 (seconds)
```

Mean ≈ 0.446s, similarly tight.

### Arithmetic reconciliation

Onset (mean 0.816s across a broader 10-file sample, see Section 3) + tail (mean 0.446s) ≈
**1.26s dead out of 11s ≈ 11.5%** — matching the originally measured 10.8–11.6% dead
(88.4–89.2% alive) almost exactly. The gap is fully accounted for by two silent brackets,
one at each end; no unexplained residual remains.

---

## 3. Root cause — confirmed against the source paper, not inferred

The DCASE Task 2 dataset description of how it uses ToyADMOS IND data states directly that
IND-type data contains the operating sounds of the entire operation — from start to stop —
within a single recording. The original ToyADMOS paper explains why IND exists as a separate
category at all: IND data collection requires the machine to be started and stopped many
times, unlike CNT, which just records an already-running machine.

**Each IND clip is a deliberately bracketed single event: start the motor, let it run, stop the
motor — all within one 11-second recording.** The onset and tail silence are not artifacts or
defects; they are literally what "start to stop" means, faithfully captured. This is a real,
positive confirmation of the recording protocol, not a data quality problem.

---

## 4. Ruled out: label-correlated timing (would be a real evaluation validity risk)

Before concluding this was benign, checked whether the onset delay differs systematically
between normal and anomalous clips — if damaged-component recordings had a different
trigger/settling delay, a model could learn to separate normal from anomalous using clip-timing
alone rather than genuine acoustic content, invalidating any AUC measured on this fold.

```
normal onset:  mean=0.816s, std=0.072s
anomaly onset: mean=0.768s, std=0.116s
```

The gap (0.048s) is small relative to either group's spread. Anomaly's larger std is itself
plausible on physical grounds — a damaged motor's startup behavior would reasonably be more
variable than a healthy one's. **No evidence of a label-correlated timing shortcut.** This
check should be considered a first pass, not exhaustive (a formal significance test was not
run), but nothing in the data suggests further investigation is urgent.

---

## 5. Why this is a different phenomenon from CNT's silence, and needs a different response

| | CNT silence | IND silence |
|---|---|---|
| cause | 10-min-on/10-min-off motor duty cycle (see `110_dataset_fold_stats.md`) | deliberate start/stop bracket around one operation |
| duration | multi-minute stretches | ~1.3s total, split across both ends |
| position | anywhere in a 10-min file, drifts across a session | fixed, ~0.8s onset + ~0.45s tail, every clip |
| is it real signal loss? | yes — genuinely dead motor, correctly excluded | no — genuinely part of "one complete operation" |
| correct treatment | window-level filtering (implemented, see `cache.py`) | **do not filter** — filtering would discard legitimate bracket audio |

**Applying the CNT-style filter to IND would be actively wrong**, not just unnecessary — IND
clips are exactly one window each (`window_frames` == full clip length), so "filtering" would
mean discarding entire clips wholesale, at a 93.2% rejection rate, destroying most of val/test
in the process. The existing design decision to leave val/test unfiltered was made for other
reasons at the time; this finding gives it a stronger, confirmed justification.

---

## 6. Consequence: a structural train/eval distribution mismatch

Every CNT training window (post-filter) is, by construction, ≥90% real audio — the model has
never been trained on a window that starts or ends mid-transition, silence fading into sound
or sound fading into silence. **Every single IND validation window contains exactly that
shape, at both ends, without exception**, since IND windows are never filtered.

This is a plausible, concrete contributor to the epoch-to-epoch noise observed in `val_mse`
during training (see training log, held-out-1 retrain): the model is evaluated, every epoch,
on a transition pattern it was never given an example of during training. This is distinct
from — and additive to — the checkpoint-selection noise issue and the fold-relative-baseline
issue documented elsewhere.

**Not addressed in this session.** A possible future mitigation: deliberately include a small
fraction of transition-spanning training windows (rather than filtering all of them out) so
the model has *some* exposure to onset/offset shapes. Not implemented; worth a design decision
in Phase 2 rather than adjusting `min_frame_fraction` for CNT, which is a different phenomenon
(see Section 5) and would not create the same exposure.

Superseded by the IND-only restart. This mismatch was real for the CNT pipeline and is now structurally impossible: training, validation, and test all draw from IND clips carrying the same onset/tail bracket. The val_mse instability this section proposed as a contributor does not reproduce on IND (findings/130 §1, sixteen runs with smooth descent). The proposed mitigation — deliberately including transition-spanning training windows — is moot. Retained as record of a correctly-reasoned hypothesis whose premise was removed by a data-source change rather than by being tested.

---

## 7. Open item — not reconciled with an earlier, separate finding

Early in the project (before the raw-waveform filtering work), a whole-file RMS scan flagged
**~30 IND files in case1 (14 normal + 16 anomaly)** as near-total-silence throughout — a much
more severe condition than the ~1.3s bracket documented here, which would not be sufficient to
drag a whole-file average below that earlier threshold.

**Whether these 30 files are simply extreme instances of natural bracket-duration variance, or
a genuinely separate defect (e.g., a failed recording), has not been checked.** Worth a direct
listen/inspection pass on a few of these specific files, same methodology as the CNT silence
investigation (`110_dataset_fold_stats.md`), before assuming either explanation.

Possible connection worth checking. Case1 is the hardest LOSO fold and the only one with flagged near-silent IND files. findings/130 §9.3 shows case1 fails diffusely — essentially no buried clips but 211–257 anomalies overlapping at least one normal — which is what a contaminated normal pool would look like. Untested, and the 30 files split roughly evenly across normal and anomaly, which argues against a one-sided effect. Cheap to check: exclude them and re-score case1.
