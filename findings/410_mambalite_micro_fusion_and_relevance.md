# MambaLite-Micro's fusion mechanism, and whether it matches this project's port target

**Feeds:** `05_phase_4_backbone_port.md` step 2 ("Read their C engine").

## Their fusion, precisely

The original Mamba recurrence discretizes for the whole sequence up front: it builds
`A_bar` and `B_bar` as (batch, d_inner, length, d_state) tensors, then scans over them one
timestep at a time. MambaLite-Micro fuses the discretization into the scan itself: at each
timestep `i`, it computes `A_bar_i = exp(delta_i * A)` and the input-scaled term
`delta_i * (B * u_i)` on the fly, immediately folding them into the state update, and never
materializes the full-sequence tensor. This drops memory from `O(batch*d_inner*length*d_state)`
to `O(batch*d_inner*d_state)` and, as a side effect, makes the recurrence streamable frame by
frame rather than requiring the whole clip in memory first.

## This already matches `ssm_block.py`, exactly

`_scan_streaming()` and `discretize()` implement the identical pattern: `discretize()` is called
once per timestep inside the loop rather than once for the whole sequence, and nothing about it
changes when called that way, since every operation inside it is elementwise with no
cross-timestep reduction. `findings/220` section 9 already validated this against
`parity_vectors.npz` at `0.000e+00` difference for all four refit configs across all four folds.
Step 2's actual question -- does the winning config share the tensor structure MambaLite-Micro's
fusion avoids materializing -- is already answered: yes, and it's already been verified
byte-exact, before this session started.

## One precise nuance: which discretization formula

MambaLite-Micro's fused equation uses `A_bar = exp(delta*A)` (exact) for the state-decay term,
but `delta*(B*u)` (a first-order approximation, not the exact ZOH coefficient) for the input
term. This is the standard Mamba discretization, and it matches neither of this project's two
`discretization` modes exactly: `zoh` here computes both terms exactly (via the numerically
stable `expm1` form), and `euler` approximates both terms to first order, including the `A`
term. The config being ported, `f4cd557b7e3b`, uses `euler`.

This doesn't affect whether the fusion technique transfers -- fusion is about *when* `A_bar`/
`B_bar` get computed, not *which* formula computes them, and `discretize()` is called identically
regardless of `self.discretization`. It only means a direct numerical comparison against
MambaLite-Micro's own reported error figures would be comparing two different approximation
schemes, not the same formula on different hardware.

## Their hardware and latency numbers

Their STM32H7 target was the Portenta H7's `STM32H747XIH6 @ 480 MHz`, with `511.35 KB RAM` and
`768 KB flash` **available inside their PlatformIO/mbed toolchain** -- smaller than the chip's
full 1 MB/2 MB capacity, since the framework consumes some of both. Their KWS model (a single
Mamba layer, `d_model=64` -- matching this project's own `d_model` exactly, `d_state` and
`expand` unspecified in the paper) processed 100 timesteps in 934.9 ms fp32 on that chip: roughly
9.3 ms per timestep for one layer, unquantized, no SIMD.

**Claude comment:** I want to flag this number as a loose anchor, not a prediction. Their model's
`d_state`/`expand` aren't given, so I can't scale it precisely to `f4cd557b7e3b`'s two layers.
But if 9.3 ms/timestep/layer is even roughly the right order of magnitude, two stacked layers
land around 18-19 ms/frame on similar hardware -- against a **32 ms per-frame budget**, not 16 ms:
`hop_length=512` in `configs/default.yaml` is 32 ms at 16 kHz, matching the 344-frames-per-11s-clip
figure `findings/324` reports. (`01_design_decisions.md` §3's illustrative "512-sample window,
256-sample hop" example uses different numbers than the config that was actually built --
harmless for that section's own argument, since the "zero shared samples at k=2" conclusion holds
either way, but worth knowing if you go looking for where 16 ms came from.) A ~19 ms-of-32 ms
estimate leaves real margin, but not so much that it can be waved off before step 5 actually
measures it -- especially once the feature pipeline's own per-frame cost (not yet quantified
in the findings I've read) is added on top.

## Bottom line

Nothing here changes the recommendation. The fusion technique this project needs is already
implemented and already validated; the config choice doesn't need to be revisited on the strength
of anything in MambaLite-Micro's repo. The two things worth carrying forward are the corrected
128.25 KB fp32 flash figure (see `findings/225`) and a loose expectation that real-time margin,
not flash, is the more likely place for Phase 4 to run into trouble on this board.