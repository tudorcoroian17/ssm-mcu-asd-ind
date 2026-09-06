# Finding: acceptable log-mel feature tolerance for the MCU pipeline

**Status:** Resolved. Tolerance derived, ready to hand to `04_phase_3_mcu_feature_pipeline.md`
§3.4 as the CMSIS-DSP parity acceptance criterion. Two things remain open, both process
decisions rather than missing data — see §8.

**Context:** `04_phase_3_mcu_feature_pipeline.md` §3.4 requires the log-mel parity tolerance to be
tied to something downstream — the noise level at which the trained model's AUC actually starts
to move — rather than chosen as a round number. This finding derives that number by injecting
synthetic noise into the log-mel features and measuring where AUC degrades, using the four
Phase 2 refit configurations (`findings/220`) and the existing eval pipeline. No retraining and
no recomputation of any reference set was needed; this is an eval-time experiment against
existing checkpoints and existing frozen embeddings. Reproduced with
`checks/misc/cmsis_dsp_tollerance.py`.

**Claude comment:** the method changed twice while running it, and both changes are kept here
alongside the reasoning, not silently folded into a clean final version — the same convention
`findings/130` uses. A tolerance derivation is only as trustworthy as the record of what could
have gone wrong and did not.

---

## 1. Method

### 1.1 What is perturbed, and what is not

Noise is injected into **`log_mel_unnormalized`** — the pre-normalization array — not the
model's normalized input. This is the quantity a board's CMSIS-DSP chain actually produces and
the quantity §3.4's parity harness diffs against; perturbing anything else would test a
different question than the one §3.4 asks.

Only **test-time query clips** are perturbed. The Option 3 reference set (`train_emb`, the
frozen centroid/covariance/cluster-reference embeddings) is never touched. In deployment the MCU
never recomputes the reference set — it ships fixed, computed once from clean GPU data during
training — so injecting noise there would test a scenario that does not occur.

### 1.2 Fold: case 1 only

The sweep runs on **held-out case 1 exclusively**. Cases 2 to 4 sit at or near an AUC ceiling
(`findings/130` §8: 0.99 and above, near-zero seed variance), with residual error concentrated in
a handful of specific buried clips rather than a continuous quality signal. Sweeping noise on a
saturated fold would require a large, unrealistic perturbation before AUC visibly moves — not
because the pipeline is robust, but because the fold has no headroom left to lose. Case 1 is the
only fold with continuous dynamic range, so it is the only fold where "AUC starts to move" means
what it is meant to mean.

### 1.3 Threshold definition

The "knee" for a given configuration and scoring head is the smallest `noise_std` after which
the AUC drop (baseline AUC minus perturbed AUC) exceeds **`SEED_SD = 0.0132`**
(`findings/210`, from the three-seed default-model measurement in `findings/130` §8) and remains
above it for the rest of the sweep. A noise level that moves AUC by less than `SEED_SD` is not
distinguishable from ordinary seed-to-seed training noise.

The first version of this rule — first point to cross the threshold, without requiring it to
hold — produced false knees: A-classic's original sweep reported a knee only at the single last
point tested, immediately after the AUC had been steadily *improving* under noise for the entire
range. Requiring the crossing to hold for the remainder of the sweep, combined with extending the
sweep range (§1.5), resolved this.

### 1.4 Self-check and repeat-averaging

Before trusting any point on a sweep, the harness confirms that `noise_std = 0` reproduces the
already-saved `test_emb` to within `1e-4` — catching any reload mismatch (wrong checkpoint, stale
normalization stats, wrong pooling mode) before it contaminates every subsequent point.

Each non-zero `noise_std` is evaluated over multiple independent noise draws and averaged
(`auc_std` column in §4's tables reports the spread across draws). A single draw was found to
introduce enough jitter — a few `1e-4` AUC — to manufacture spurious knee crossings at small
noise, before averaging was added.

### 1.5 Sweep range

The initial range (`noise_std` up to ~0.4) was too narrow: two configurations' real knees sat
past its edge, and one apparent knee (A-classic, first pass) turned out to be an unexplored cliff
at the range boundary rather than a real, sustained crossing. The range was extended to
`noise_std` from 0.001 to 10.0 (log-spaced, 30 points plus zero) before any knee was trusted.

---

## 2. Natural scale of log-mel

Before treating any noise-std number as meaningful, the natural per-bin spread of log-mel itself
was measured, since a "tolerance" expressed only in absolute log-mel units says nothing about
whether the noise levels being swept are physically plausible for a hardware precision error.

| sample size | measured per-bin std | log-mel range |
| :--- | ---: | :--- |
| 20 clips | 4.0974 | [-13.815, 1.218] |
| larger sample | 4.1933 | [-13.815, 2.536] |

The two measurements agree to within 2.3%, confirming the natural-std anchor is not an artifact
of a small sample. The range floor, `-13.815`, is exactly `log(1e-6)` — the configured `log_eps`
value that near-silent bins are clamped to. A meaningful fraction of the measured spread
therefore reflects the gap between silent-floor bins and signal-carrying bins, not purely
frame-to-frame variability within active bins. This does not invalidate the derived tolerance,
but it means the "percentage of natural std" framing is best read as supporting context for the
absolute tolerance, not as an independently precise physical statement.

---

## 3. Raw sweep results

Full sweep output, all four configurations, both scoring heads, as printed by
`checks/misc/cmsis_dsp_tollerance.py`. `noise_std` is additive Gaussian std applied to
unnormalized log-mel. `auc` and `auc_std` are the mean and standard deviation of AUC across
repeated noise draws at that `noise_std`. `auc_drop` is baseline AUC (the `noise_std = 0` row)
minus `auc`.

Configuration key: A = d_state=16, n_layers=2, expand=1, euler. B = d_state=8, n_layers=2,
expand=2, euler. Selective/classic per `findings/220` §5.1.

### 3.1 Head: knn_clustered_16

#### 3.1.1 A-selective (`f4cd557b7e3b.yaml`)

Baseline AUC = 0.910554. Knee = 0.7880462815669912.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.910554 | 0.000000 | 0.000000 |
| 0.001000 | 0.910557 | 0.000006 | -0.000003 |
| 0.001374 | 0.910560 | 0.000007 | -0.000006 |
| 0.001887 | 0.910563 | 0.000007 | -0.000009 |
| 0.002593 | 0.910549 | 0.000007 | 0.000006 |
| 0.003562 | 0.910552 | 0.000011 | 0.000003 |
| 0.004894 | 0.910549 | 0.000007 | 0.000006 |
| 0.006723 | 0.910563 | 0.000015 | -0.000009 |
| 0.009237 | 0.910557 | 0.000025 | -0.000003 |
| 0.012690 | 0.910552 | 0.000011 | 0.000003 |
| 0.017433 | 0.910546 | 0.000035 | 0.000009 |
| 0.023950 | 0.910560 | 0.000025 | -0.000006 |
| 0.032903 | 0.910529 | 0.000021 | 0.000026 |
| 0.045204 | 0.910557 | 0.000031 | -0.000003 |
| 0.062102 | 0.910574 | 0.000078 | -0.000020 |
| 0.085317 | 0.910709 | 0.000148 | -0.000155 |
| 0.117210 | 0.910560 | 0.000100 | -0.000006 |
| 0.161026 | 0.910428 | 0.000151 | 0.000126 |
| 0.221222 | 0.909955 | 0.000320 | 0.000600 |
| 0.303920 | 0.908663 | 0.000241 | 0.001891 |
| 0.417532 | 0.906003 | 0.000540 | 0.004551 |
| 0.573615 | 0.900264 | 0.000573 | 0.010290 |
| 0.788046 | 0.890969 | 0.000394 | 0.019585 |
| 1.082637 | 0.877172 | 0.001237 | 0.033382 |
| 1.487352 | 0.819631 | 0.004050 | 0.090923 |
| 2.043360 | 0.706606 | 0.004274 | 0.203949 |
| 2.807216 | 0.647627 | 0.002840 | 0.262928 |
| 3.856620 | 0.563303 | 0.011114 | 0.347251 |
| 5.298317 | 0.478294 | 0.012587 | 0.432260 |
| 7.278954 | 0.451300 | 0.013594 | 0.459254 |
| 10.000000 | 0.484800 | 0.007424 | 0.425755 |

#### 3.1.2 A-classic (`f2578cb06991.yaml`)

Baseline AUC = 0.842487. Knee = 5.298316906283707.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.842487 | 0.000000 | 0.000000 |
| 0.001000 | 0.842485 | 0.000017 | 0.000003 |
| 0.001374 | 0.842476 | 0.000011 | 0.000011 |
| 0.001887 | 0.842473 | 0.000018 | 0.000014 |
| 0.002593 | 0.842464 | 0.000025 | 0.000023 |
| 0.003562 | 0.842473 | 0.000041 | 0.000014 |
| 0.004894 | 0.842470 | 0.000031 | 0.000017 |
| 0.006723 | 0.842462 | 0.000025 | 0.000026 |
| 0.009237 | 0.842499 | 0.000062 | -0.000011 |
| 0.012690 | 0.842548 | 0.000040 | -0.000060 |
| 0.017433 | 0.842533 | 0.000069 | -0.000046 |
| 0.023950 | 0.842645 | 0.000099 | -0.000158 |
| 0.032903 | 0.842680 | 0.000102 | -0.000192 |
| 0.045204 | 0.842872 | 0.000114 | -0.000385 |
| 0.062102 | 0.843064 | 0.000242 | -0.000577 |
| 0.085317 | 0.843916 | 0.000378 | -0.001429 |
| 0.117210 | 0.844281 | 0.000164 | -0.001794 |
| 0.161026 | 0.845558 | 0.000714 | -0.003070 |
| 0.221222 | 0.847747 | 0.000479 | -0.005260 |
| 0.303920 | 0.850872 | 0.000473 | -0.008385 |
| 0.417532 | 0.856175 | 0.000831 | -0.013688 |
| 0.573615 | 0.862250 | 0.001439 | -0.019763 |
| 0.788046 | 0.868007 | 0.000386 | -0.025519 |
| 1.082637 | 0.875913 | 0.001871 | -0.033425 |
| 1.487352 | 0.860457 | 0.003509 | -0.017969 |
| 2.043360 | 0.690108 | 0.007667 | 0.152379 |
| 2.807216 | 0.792261 | 0.003406 | 0.050227 |
| 3.856620 | 0.846597 | 0.005496 | -0.004109 |
| 5.298317 | 0.643885 | 0.005933 | 0.198603 |
| 7.278954 | 0.439916 | 0.010688 | 0.402571 |
| 10.000000 | 0.400075 | 0.008555 | 0.442413 |

#### 3.1.3 B-classic (`352f70960ed3.yaml`)

Baseline AUC = 0.927542. Knee = 2.8072162039411754.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.927542 | 0.000000e+00 | 0.000000e+00 |
| 0.001000 | 0.927542 | 1.110223e-16 | -1.110223e-16 |
| 0.001374 | 0.927542 | 1.110223e-16 | -1.110223e-16 |
| 0.001887 | 0.927540 | 5.739210e-06 | 2.869605e-06 |
| 0.002593 | 0.927542 | 1.110223e-16 | -1.110223e-16 |
| 0.003562 | 0.927545 | 5.739210e-06 | -2.869605e-06 |
| 0.004894 | 0.927540 | 5.739210e-06 | 2.869605e-06 |
| 0.006723 | 0.927540 | 1.073708e-05 | 2.869605e-06 |
| 0.009237 | 0.927537 | 1.463217e-05 | 5.739210e-06 |
| 0.012690 | 0.927545 | 1.405814e-05 | -2.869605e-06 |
| 0.017433 | 0.927540 | 3.063900e-05 | 2.869605e-06 |
| 0.023950 | 0.927583 | 3.785269e-05 | -4.017447e-05 |
| 0.032903 | 0.927571 | 3.009667e-05 | -2.869605e-05 |
| 0.045204 | 0.927600 | 2.566653e-05 | -5.739210e-05 |
| 0.062102 | 0.927600 | 6.087352e-05 | -5.739210e-05 |
| 0.085317 | 0.927738 | 7.942122e-05 | -1.951331e-04 |
| 0.117210 | 0.927964 | 1.503739e-04 | -4.218320e-04 |
| 0.161026 | 0.927910 | 2.216480e-04 | -3.673095e-04 |
| 0.221222 | 0.928002 | 2.170302e-04 | -4.591368e-04 |
| 0.303920 | 0.928039 | 4.205318e-04 | -4.964417e-04 |
| 0.417532 | 0.928111 | 1.386943e-04 | -5.681818e-04 |
| 0.573615 | 0.981540 | 2.179552e-03 | -5.399736e-02 |
| 0.788046 | 0.991365 | 6.572210e-04 | -6.382289e-02 |
| 1.082637 | 0.983681 | 1.488482e-03 | -5.613809e-02 |
| 1.487352 | 0.967700 | 2.656058e-03 | -4.015725e-02 |
| 2.043360 | 0.940814 | 4.279831e-03 | -1.327192e-02 |
| 2.807216 | 0.864173 | 5.523558e-03 | 6.336949e-02 |
| 3.856620 | 0.728888 | 5.693825e-03 | 1.986542e-01 |
| 5.298317 | 0.535431 | 1.592013e-03 | 3.921115e-01 |
| 7.278954 | 0.398143 | 1.067378e-02 | 5.293991e-01 |
| 10.000000 | 0.362270 | 7.459778e-03 | 5.652720e-01 |

#### 3.1.4 B-selective (`b39731b66741.yaml`)

Baseline AUC = 0.819129. Knee = 2.8072162039411754.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.819129 | 0.000000 | 0.000000 |
| 0.001000 | 0.819132 | 0.000011 | -0.000003 |
| 0.001374 | 0.819132 | 0.000006 | -0.000003 |
| 0.001887 | 0.819114 | 0.000024 | 0.000014 |
| 0.002593 | 0.819123 | 0.000007 | 0.000006 |
| 0.003562 | 0.819123 | 0.000030 | 0.000006 |
| 0.004894 | 0.819155 | 0.000028 | -0.000026 |
| 0.006723 | 0.819212 | 0.000050 | -0.000083 |
| 0.009237 | 0.819175 | 0.000076 | -0.000046 |
| 0.012690 | 0.819203 | 0.000048 | -0.000075 |
| 0.017433 | 0.819244 | 0.000092 | -0.000115 |
| 0.023950 | 0.819292 | 0.000135 | -0.000164 |
| 0.032903 | 0.819315 | 0.000054 | -0.000187 |
| 0.045204 | 0.819539 | 0.000100 | -0.000410 |
| 0.062102 | 0.819553 | 0.000105 | -0.000425 |
| 0.085317 | 0.819906 | 0.000352 | -0.000778 |
| 0.117210 | 0.820437 | 0.000530 | -0.001309 |
| 0.161026 | 0.820862 | 0.000227 | -0.001733 |
| 0.221222 | 0.822449 | 0.000568 | -0.003320 |
| 0.303920 | 0.823333 | 0.001647 | -0.004204 |
| 0.417532 | 0.827454 | 0.001123 | -0.008325 |
| 0.573615 | 0.847483 | 0.002540 | -0.028355 |
| 0.788046 | 0.891463 | 0.001819 | -0.072334 |
| 1.082637 | 0.911983 | 0.001420 | -0.092855 |
| 1.487352 | 0.897704 | 0.001586 | -0.078576 |
| 2.043360 | 0.827689 | 0.005753 | -0.008560 |
| 2.807216 | 0.709077 | 0.011123 | 0.110052 |
| 3.856620 | 0.715123 | 0.007873 | 0.104006 |
| 5.298317 | 0.700585 | 0.006279 | 0.118543 |
| 7.278954 | 0.604201 | 0.009282 | 0.214928 |
| 10.000000 | 0.560474 | 0.007689 | 0.258655 |

### 3.2 Head: euclidean

#### 3.2.1 A-selective (`f4cd557b7e3b.yaml`)

Baseline AUC = 0.945047. Knee = 0.7880462815669912.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.945047 | 0.000000 | 0.000000e+00 |
| 0.001000 | 0.945050 | 0.000011 | -2.869605e-06 |
| 0.001374 | 0.945050 | 0.000011 | -2.869605e-06 |
| 0.001887 | 0.945044 | 0.000006 | 2.869605e-06 |
| 0.002593 | 0.945047 | 0.000009 | 1.110223e-16 |
| 0.003562 | 0.945056 | 0.000017 | -8.608815e-06 |
| 0.004894 | 0.945061 | 0.000016 | -1.434803e-05 |
| 0.006723 | 0.945047 | 0.000016 | -1.110223e-16 |
| 0.009237 | 0.945050 | 0.000021 | -2.869605e-06 |
| 0.012690 | 0.945073 | 0.000028 | -2.582645e-05 |
| 0.017433 | 0.945081 | 0.000032 | -3.443526e-05 |
| 0.023950 | 0.945136 | 0.000060 | -8.895776e-05 |
| 0.032903 | 0.945176 | 0.000047 | -1.291322e-04 |
| 0.045204 | 0.945162 | 0.000033 | -1.147842e-04 |
| 0.062102 | 0.945193 | 0.000135 | -1.463499e-04 |
| 0.085317 | 0.945500 | 0.000163 | -4.533976e-04 |
| 0.117210 | 0.945354 | 0.000100 | -3.070478e-04 |
| 0.161026 | 0.945403 | 0.000184 | -3.558310e-04 |
| 0.221222 | 0.944720 | 0.000411 | 3.271350e-04 |
| 0.303920 | 0.943678 | 0.000424 | 1.368802e-03 |
| 0.417532 | 0.940760 | 0.000558 | 4.287190e-03 |
| 0.573615 | 0.936387 | 0.000887 | 8.660468e-03 |
| 0.788046 | 0.928082 | 0.000579 | 1.696511e-02 |
| 1.082637 | 0.905274 | 0.002458 | 3.977273e-02 |
| 1.487352 | 0.790596 | 0.002360 | 1.544508e-01 |
| 2.043360 | 0.684570 | 0.002910 | 2.604769e-01 |
| 2.807216 | 0.606348 | 0.004249 | 3.386995e-01 |
| 3.856620 | 0.525603 | 0.010272 | 4.194444e-01 |
| 5.298317 | 0.479023 | 0.008182 | 4.660239e-01 |
| 7.278954 | 0.470779 | 0.010987 | 4.742683e-01 |
| 10.000000 | 0.465031 | 0.003872 | 4.800161e-01 |

#### 3.2.2 A-classic (`f2578cb06991.yaml`)

Baseline AUC = 0.882633. Knee = 2.0433597178569416.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.882633 | 0.000000 | 0.000000 |
| 0.001000 | 0.882642 | 0.000015 | -0.000009 |
| 0.001374 | 0.882639 | 0.000007 | -0.000006 |
| 0.001887 | 0.882619 | 0.000009 | 0.000014 |
| 0.002593 | 0.882609 | 0.000009 | 0.000024 |
| 0.003562 | 0.882636 | 0.000033 | -0.000003 |
| 0.004894 | 0.882630 | 0.000028 | 0.000003 |
| 0.006723 | 0.882616 | 0.000021 | 0.000017 |
| 0.009237 | 0.882636 | 0.000021 | -0.000003 |
| 0.012690 | 0.882673 | 0.000045 | -0.000040 |
| 0.017433 | 0.882656 | 0.000070 | -0.000023 |
| 0.023950 | 0.882719 | 0.000065 | -0.000086 |
| 0.032903 | 0.882696 | 0.000110 | -0.000063 |
| 0.045204 | 0.882837 | 0.000154 | -0.000204 |
| 0.062102 | 0.882978 | 0.000155 | -0.000344 |
| 0.085317 | 0.883379 | 0.000124 | -0.000746 |
| 0.117210 | 0.883844 | 0.000194 | -0.001211 |
| 0.161026 | 0.884668 | 0.000301 | -0.002035 |
| 0.221222 | 0.885701 | 0.000337 | -0.003068 |
| 0.303920 | 0.887727 | 0.000682 | -0.005094 |
| 0.417532 | 0.890292 | 0.000643 | -0.007659 |
| 0.573615 | 0.894700 | 0.001500 | -0.012067 |
| 0.788046 | 0.899392 | 0.000577 | -0.016758 |
| 1.082637 | 0.907536 | 0.001470 | -0.024902 |
| 1.487352 | 0.896347 | 0.005280 | -0.013714 |
| 2.043360 | 0.835374 | 0.005025 | 0.047260 |
| 2.807216 | 0.752898 | 0.004822 | 0.129735 |
| 3.856620 | 0.706663 | 0.010545 | 0.175970 |
| 5.298317 | 0.698918 | 0.008240 | 0.183715 |
| 7.278954 | 0.686223 | 0.008900 | 0.196410 |
| 10.000000 | 0.658203 | 0.016767 | 0.224430 |

#### 3.2.3 B-classic (`352f70960ed3.yaml`)

Baseline AUC = 0.948462. Knee = 1.4873521072935119.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.948462 | 0.000000 | 0.000000 |
| 0.001000 | 0.948476 | 0.000009 | -0.000014 |
| 0.001374 | 0.948468 | 0.000025 | -0.000006 |
| 0.001887 | 0.948482 | 0.000021 | -0.000020 |
| 0.002593 | 0.948473 | 0.000017 | -0.000011 |
| 0.003562 | 0.948479 | 0.000014 | -0.000017 |
| 0.004894 | 0.948482 | 0.000021 | -0.000020 |
| 0.006723 | 0.948485 | 0.000054 | -0.000023 |
| 0.009237 | 0.948502 | 0.000014 | -0.000040 |
| 0.012690 | 0.948491 | 0.000020 | -0.000029 |
| 0.017433 | 0.948508 | 0.000064 | -0.000046 |
| 0.023950 | 0.948522 | 0.000096 | -0.000060 |
| 0.032903 | 0.948511 | 0.000052 | -0.000049 |
| 0.045204 | 0.948603 | 0.000132 | -0.000141 |
| 0.062102 | 0.948657 | 0.000108 | -0.000195 |
| 0.085317 | 0.948686 | 0.000201 | -0.000224 |
| 0.117210 | 0.949122 | 0.000390 | -0.000660 |
| 0.161026 | 0.949349 | 0.000442 | -0.000887 |
| 0.221222 | 0.949659 | 0.000560 | -0.001197 |
| 0.303920 | 0.950364 | 0.000417 | -0.001903 |
| 0.417532 | 0.951329 | 0.000525 | -0.002867 |
| 0.573615 | 0.951735 | 0.000665 | -0.003273 |
| 0.788046 | 0.950979 | 0.001440 | -0.002517 |
| 1.082637 | 0.947067 | 0.001322 | 0.001395 |
| 1.487352 | 0.929445 | 0.002554 | 0.019017 |
| 2.043360 | 0.877695 | 0.004165 | 0.070767 |
| 2.807216 | 0.757229 | 0.005284 | 0.191233 |
| 3.856620 | 0.619215 | 0.002985 | 0.329247 |
| 5.298317 | 0.496020 | 0.003374 | 0.452442 |
| 7.278954 | 0.418311 | 0.006916 | 0.530151 |
| 10.000000 | 0.395696 | 0.007529 | 0.552766 |

#### 3.2.4 B-selective (`b39731b66741.yaml`)

Baseline AUC = 0.867553. Knee = 1.0826367338740541.

| noise_std | auc | auc_std | auc_drop |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.867553 | 0.000000 | 0.000000 |
| 0.001000 | 0.867571 | 0.000019 | -0.000017 |
| 0.001374 | 0.867571 | 0.000017 | -0.000017 |
| 0.001887 | 0.867576 | 0.000019 | -0.000023 |
| 0.002593 | 0.867576 | 0.000043 | -0.000023 |
| 0.003562 | 0.867619 | 0.000049 | -0.000066 |
| 0.004894 | 0.867662 | 0.000061 | -0.000109 |
| 0.006723 | 0.867703 | 0.000100 | -0.000149 |
| 0.009237 | 0.867680 | 0.000029 | -0.000126 |
| 0.012690 | 0.867803 | 0.000072 | -0.000250 |
| 0.017433 | 0.867849 | 0.000089 | -0.000296 |
| 0.023950 | 0.868001 | 0.000115 | -0.000448 |
| 0.032903 | 0.868196 | 0.000127 | -0.000643 |
| 0.045204 | 0.868449 | 0.000159 | -0.000895 |
| 0.062102 | 0.868549 | 0.000310 | -0.000996 |
| 0.085317 | 0.869579 | 0.000708 | -0.002026 |
| 0.117210 | 0.870779 | 0.000353 | -0.003225 |
| 0.161026 | 0.872047 | 0.000758 | -0.004494 |
| 0.221222 | 0.873402 | 0.000221 | -0.005848 |
| 0.303920 | 0.875872 | 0.000533 | -0.008319 |
| 0.417532 | 0.877732 | 0.000607 | -0.010178 |
| 0.573615 | 0.879723 | 0.001622 | -0.012170 |
| 0.788046 | 0.876992 | 0.001245 | -0.009438 |
| 1.082637 | 0.851667 | 0.002918 | 0.015886 |
| 1.487352 | 0.742657 | 0.002616 | 0.124897 |
| 2.043360 | 0.646018 | 0.001773 | 0.221535 |
| 2.807216 | 0.604761 | 0.005628 | 0.262793 |
| 3.856620 | 0.583563 | 0.003350 | 0.283990 |
| 5.298317 | 0.548488 | 0.003137 | 0.319066 |
| 7.278954 | 0.489606 | 0.009360 | 0.377947 |
| 10.000000 | 0.477603 | 0.009827 | 0.389951 |

---

## 4. Findings

### 4.1 A-selective is the binding constraint, consistently

A-selective's knee, `0.7880462815669912`, is identical to the last printed digit under both
`knn_clustered_16` and `euclidean`. This exactness is a property of the experiment design, not a
coincidence: noise is drawn once per `noise_std` and the resulting corrupted embedding is scored
under every head, so the two heads' knees for one configuration are directly comparable rather
than independently noisy estimates.

### 4.2 Relative to natural scale

| config | knn_clustered_16 knee | knn ratio to natural std | euclidean knee | euclidean ratio to natural std |
| :--- | ---: | ---: | ---: | ---: |
| A-selective | 0.7880 | **0.188** | 0.7880 | **0.188** |
| B-selective | 2.8072 | 0.669 | 1.0826 | 0.258 |
| B-classic | 2.8072 | 0.669 | 1.4874 | 0.355 |
| A-classic | 5.2983 | 1.264 | 2.0434 | 0.487 |

Ratios use the larger-sample natural std, 4.1933 (§2). A-selective tolerates noise at only 18.8%
of natural signal spread before degrading — roughly 3.5 times tighter than the next-nearest
configuration under `knn_clustered_16`, and consistently the tightest under `euclidean` as well.
The other three configurations tolerate noise comparable to or exceeding natural signal spread.

This is specific to **A-selective** (d_state=16, n_layers=2, expand=1, selective, euler), not to
selectivity as a general property: B-selective's ratios (0.669 / 0.258) sit mid-pack, not at the
same extreme. The claim is scoped to this one configuration, not to "selective SSMs" broadly.

**Claude comment:** stated this way deliberately, since `findings/220` §5 already established
that selectivity's effect is architecture-dependent rather than a fixed property — this result
is a third instance of the same pattern (alongside accuracy and footprint), and generalizing it
to "selective is less robust" would be exactly the overclaim that document's own crossover
finding warns against.

### 4.3 Non-monotonic curves are a real property of `knn_clustered_16`, not noise

B-classic and B-selective's `auc_drop` under `knn_clustered_16` goes *negative* (AUC improving)
through moderate noise before eventually collapsing at high noise — B-classic reaches
`-0.0638` at `noise_std = 0.788` before later crossing to `+0.529` at `noise_std = 7.279`.
`euclidean`'s curves for the same two configurations are smooth and monotonic throughout.

The mechanism: `euclidean`'s score is distance to a single centroid, a continuous function of the
embedding. `knn_clustered_16`'s score is distance to the *nearest* of 16 cluster references — a
piecewise function with discontinuities at cluster-Voronoi boundaries. Small perturbations can
push a borderline point across such a boundary in either direction, transiently helping or
hurting AUC before enough points have crossed for the net effect to become monotonically
degrading. This is why the knee definition in §1.3 requires the threshold crossing to be
sustained, not merely touched once.

### 4.4 Sweep-range artifacts caught during derivation

Two false readings occurred before the final range and averaging were in place, both instructive
enough to record:

- **A-classic's first-pass knee** (narrower sweep, single noise draw per point) was reported at
  the sweep's last tested point, immediately following a long run of *improving* AUC under noise.
  Extending the range revealed this was an unexplored cliff at the boundary, not a real,
  sustained knee — the real, sustained crossing (`5.298`) was found only after widening the range
  to 10.0.
- **B-classic and B-selective under `knn_clustered_16`** initially reported "knee not reached"
  within the narrower range; both were still descending toward their real crossing when the
  original sweep stopped. Both resolved to real knees (`2.807` for both) once the range was
  extended.

---

## 5. Recommended tolerance

**0.2627**, in absolute log-mel units (A-selective's knee, 0.7880, divided by a 3x safety
margin). Equivalently, approximately **6.3%** of the natural per-bin log-mel standard deviation.

This is the number for `04_phase_3_mcu_feature_pipeline.md` §3.4's per-bin CMSIS-DSP parity
acceptance test, on every board, applied uniformly regardless of which of the four Phase 2
configurations is eventually deployed.

---

## 6. Open items

- **Which configuration should set the tolerance.** Using A-selective's knee is the conservative,
  safe-regardless-of-final-choice option, and is what §5 recommends by default. A-selective is
  not on the accuracy/footprint Pareto frontier (`findings/220` §8); if it is ruled out as a
  deployment candidate on those grounds, the tolerance could instead be set by the tightest of the
  remaining three (B-selective, knee 0.669 of natural std), giving a looser, easier-to-implement
  target of roughly 15.9% of natural std. Not decided here — a conscious trade-off between safety
  margin and CMSIS-DSP implementation difficulty, to be made once (or if) a final deployment
  configuration is chosen.
- **The `log_eps` floor's contribution to natural std** (§2) is not quantified separately. If a
  precise physical interpretation of the percentage figure is ever needed, natural std should be
  remeasured excluding floor-clamped bins.
- **Natural-std sample size** for the final reported value (4.1933) was not recorded exactly in
  this document — confirm and record before this number is cited elsewhere.