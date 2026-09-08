# Chip spec diff: Nucleo-H7S3L8 vs MambaLite-Micro's STM32H747XIH6

**Feeds:** `00_master_file.md` §17 item 11; `05_phase_4_backbone_port.md` step 1 ("Diff the chips").

## Result

The two chips are close in raw CPU capability. The Nucleo-H7S3L8 is not weaker than
MambaLite-Micro's target chip on any compute axis; the one real structural difference is
where code and weights execute from.

| Spec | STM32H7S3L8 (Nucleo, this project) | STM32H747XIH6 (Portenta H7, MambaLite-Micro) |
| :--- | ---: | ---: |
| Core | Cortex-M7 only | Cortex-M7 + Cortex-M4 (M7 used by MambaLite-Micro) |
| Clock (M7) | 600 MHz | 480 MHz |
| FPU | Double-precision | Double-precision |
| L1 cache | 32 KB + 32 KB (I/D) | 16 KB + 16 KB (I/D) |
| Performance | 1,284 DMIPS, 3,196 CoreMark | 1,027 DMIPS |
| Internal user flash | 64 KB (+128 KB system flash, not usable for application code) | Up to 2 MB, dual-bank, read-while-write |
| RAM | 620 KB (456 KB AXI SRAM + 32 KB AHB SRAM + 64 KB ITCM + 64 KB DTCM + 4 KB backup) | 1 MB (864 KB user SRAM + 192 KB TCM [64 KB ITCM + 128 KB DTCM] + 4 KB backup) |
| External memory interface | Octo-SPI/Hexa-SPI, XiP, up to 200 MHz | Quad-SPI, up to 133 MHz |

## What transfers, and what doesn't

**Compute and RAM transfer.** The H7S3L8 runs at a higher clock, has a larger cache, and a
comparable, though smaller, RAM budget. MambaLite-Micro's 83% peak-memory reduction
(`00_master_file.md` §3) is a RAM result: it avoids materializing the full-sequence
`A_bar`/`B_bar` tensors. Nothing about the H7S3L8's RAM (620 KB, versus their 1 MB) makes that
fusion trick inapplicable. The refit configs in this project (36-80 KB of weights) are small
enough that RAM is unlikely to be the binding constraint on either chip.

**Flash does not transfer, and this was already flagged.** MambaLite-Micro's chip has 2 MB of
internal flash and ran entirely from it. The Nucleo-H7S3L8's internal flash is 64 KB, smaller
than even the smallest refit config's weights alone, before code. `findings/310` already
resolved the storage question for the feature pipeline (use external XSPI flash, sparse
filterbank format); the same decision now extends to the backbone weights, since none of the
four refit configs (31-62 KB) fit in 64 KB internal flash either, and internal flash is needed
for the runtime code itself.

**The consequence for Phase 4 is latency, not feasibility.** MambaLite-Micro measured their
throughput executing from generous internal flash, where instruction and weight fetches are
effectively free. The H7S3L8 build executes weights, and per the linker scripts already in
`Appli/`, likely code too, from external XSPI in execute-in-place mode. XiP access latency is
real, even with the H7S3L8's larger cache and higher clock working against it.
`04_phase_3_mcu_feature_pipeline.md` §3.6 already named this as an open question for the
feature pipeline; it now applies with equal force to the backbone. Their "fp32 throughput
comparable to quantized int8 models" claim (`00_master_file.md` §3) shouldn't be assumed to
transfer to this board without measurement. That's exactly what step 5 of
`05_phase_4_backbone_port.md` ("measure flash/RAM/cycles per board") is for.

**Single-core versus dual-core doesn't change anything here.** MambaLite-Micro used only the
Cortex-M7 core of a dual-core chip; the H7S3L8 has no companion M4 core at all. This project's
existing Phase 3 code (`feature_pipeline.c`, `audio_ingest.c`) already runs single-core, so
there's no M4 offload path being lost.

## Bottom line

Nothing in this diff argues against porting the selective, data-dependent recurrence to the
H7S3L8. The risk MambaLite-Micro's own paper leaves untested (`00_master_file.md` §3, open item
1), whether their fusion trick holds up outside their own chip and outside fp32, isn't made
worse by anything found here. The new, chip-specific risk this diff surfaces is XiP latency
eating into the real-time margin, which was already open from Phase 3 and now needs answering
for the backbone too, not a reason to expect the port itself to fail.

**Claude comment:** the flash story here almost reproduces the CNT-versus-IND lesson from
`01_design_decisions.md` §7 in miniature: a spec that looks worse on paper (64 KB against 2 MB)
turns out to be a solved problem elsewhere in the project (external XSPI, already decided), and
the thing actually worth worrying about is a second-order effect (access latency) a comparison
table alone won't reveal. Worth measuring early, per step 5, rather than assuming either way.