"""
Activation quantizer for the Phase 5 rung-2 diagnostic.

A duck-typed hook the model calls as quantizer.apply(name, tensor) -> tensor,
mirroring the read-only range_recorder already threaded through ssm_block.py
and backbone.py. It quantize-dequantizes the tensor to int8 when its name is
in the selected group, and returns it unchanged otherwise -- so passing None
leaves the model bit-for-bit fp32.

Names match the recorder's exactly (e.g. 'block0.delta', 'final_norm_output'),
which are also the keys findings/150's ranges.json uses, so static scales line
up 1:1 with calibration.

Two axes, matching the weight scripts:
  scale_source  'onthefly'  scale from the tensor's own runtime range (a
                            best-case probe -- not deployable, the MCU cannot
                            recompute a scale per frame).
                'ranges'    fixed scale precomputed from calibration
                            (findings/150). Deployment-realistic. Requires a
                            scales dict; per-tensor only (ranges.json holds no
                            per-channel stats -- findings/150 section 10).
  granularity   'per-tensor'   one scale for the whole tensor.
                'per-channel'  one scale per SSM channel (d_inner). For most
                              quantized tensors d_inner is already the last
                              axis. A_bar, B_bar, and h are the exception:
                              their shape is (..., d_inner, d_state), so
                              d_inner sits second-to-last -- see
                              CHANNEL_AXIS_MINUS2_SUFFIXES. Quantizing those
                              three along the true last axis would scale per
                              d_state slot (16 values, the per-channel
                              recurrence's internal state size) instead of
                              per channel (128 values, the axis actually
                              analogous to the weight scripts' per-channel
                              scaling) -- a different, less meaningful grouping.
"""
import torch

# Activation groups, by name suffix. Matched with endswith so '.B' catches
# 'block0.B' but not 'block0.B_bar'.
#   scan        the recurrence internals + what feeds them -- findings/150's
#               four flagged tensors (delta, A_bar, B_bar, h) plus B, C, and
#               the scan output y_scan.
#   boundaries  layer-boundary activations a simpler scheme could reach.
ACTIVATION_GROUP_SUFFIXES = {
    'scan': ('.delta', '.A_bar', '.B_bar', '.h', '.B', '.C', '.y_scan'),
    'boundaries': ('.u_post_conv_silu', '.z_gate', '.y_gated', '.block_output', 'final_norm_output'),
    # Classic-only. A strict subset of 'scan', but used for something
    # different: on the selective branch A_bar/B_bar are genuine per-frame
    # activations, while on the classic branch they are CONSTANTS that the
    # qabar deploy schemes ship as int8 weights. There is no state-dict entry
    # for them, so this is how the fake-quant path quantizes them -- with
    # scale_source='onthefly' the per-tensor max of a constant IS that
    # constant's max-abs, and _channel_axis already returns ndim-2 (d_inner)
    # for both names, so the result is identical to per-tensor/per-channel
    # weight quantization. Not part of 'all' (both suffixes already are).
    'discretized_const': ('.A_bar', '.B_bar'),
}
ACTIVATION_GROUP_SUFFIXES['all'] = (
    ACTIVATION_GROUP_SUFFIXES['scan'] + ACTIVATION_GROUP_SUFFIXES['boundaries']
)

ACTIVATION_GROUPS = tuple(ACTIVATION_GROUP_SUFFIXES)
GRANULARITIES = ('per-tensor', 'per-channel')
SCALE_SOURCES = ('onthefly', 'ranges')

# Tensors whose real channel axis (d_inner) sits second-to-last rather than
# last. A_bar and B_bar are (batch, T, d_inner, d_state) after discretize();
# h is (batch, d_inner, d_state). See the granularity note above.
CHANNEL_AXIS_MINUS2_SUFFIXES = ('.A_bar', '.B_bar', '.h')


def _channel_axis(name, ndim):
    if any(name.endswith(s) for s in CHANNEL_AXIS_MINUS2_SUFFIXES):
        return ndim - 2
    return ndim - 1


class ActivationQuantizer:
    def __init__(self, group='scan', granularity='per-tensor',
                 scale_source='onthefly', scales=None):
        self.suffixes = ACTIVATION_GROUP_SUFFIXES[group]
        self.granularity = granularity
        self.scale_source = scale_source
        self.scales = scales or {}   # name -> fp32 scalar, required for 'ranges'
        self.applied_names = set()   # which tensors were actually quantized

        if scale_source == 'ranges' and granularity == 'per-channel':
            raise ValueError(
                "static 'ranges' scales are per-tensor only (findings/150 "
                "section 10 did not measure per-channel ranges); use "
                "granularity='per-tensor' with scale_source='ranges'."
            )

    def _should(self, name):
        return any(name.endswith(s) for s in self.suffixes)

    def apply(self, name, tensor):
        if not self._should(name):
            return tensor
        self.applied_names.add(name)

        if self.scale_source == 'onthefly':
            if self.granularity == 'per-channel' and tensor.dim() >= 2:
                axis = _channel_axis(name, tensor.dim())
                reduce_dims = tuple(d for d in range(tensor.dim()) if d != axis)
                max_abs = tensor.abs().amax(dim=reduce_dims, keepdim=True)
            else:
                max_abs = tensor.abs().max()
            scale = torch.clamp(max_abs / 127.0, min=1e-12)
        else:  # 'ranges'
            if name not in self.scales:
                raise KeyError(f"no static scale for '{name}' in scales dict")
            scale = torch.as_tensor(self.scales[name], dtype=tensor.dtype,
                                    device=tensor.device)

        q = torch.clamp(torch.round(tensor / scale), -127, 127)
        return q * scale