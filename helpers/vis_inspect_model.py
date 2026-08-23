"""
Netron-style inspector for SSMBackbone + PredictionHead.

    python inspect_model.py                      # architecture + params, random init
    python inspect_model.py --ckpt runs/case1/ckpt.pt
    python inspect_model.py --onnx               # also write runs/graph/model_T8.onnx

Why the ONNX export uses T=8 and not T=344: _scan is a Python for-loop,
so ONNX tracing unrolls it -- one copy of the scan body per timestep.
The graph TOPOLOGY is identical either way (same layers, same widths);
T=8 just keeps it small enough for Netron to draw legibly.
"""
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

from src.config import load_config, PROJECT_ROOT
from src.models.backbone import SSMBackbone
from src.models.heads import PredictionHead

import contextlib


@contextlib.contextmanager
def _onnx_op_shims():
    """
    ONNX has no Expm1 op, and the TorchScript exporter has no symbolic
    to decompose it, so tracing dies on ssm_block's exact-ZOH branch.
    expm1(x) == exp(x) - 1 mathematically; expm1 exists purely for
    precision near x=0, which is irrelevant for a graph we're drawing
    rather than running. Scoped to the export call only -- training and
    eval keep the real expm1, untouched.
    """
    original = torch.expm1
    torch.expm1 = lambda x: torch.exp(x) - 1.0
    try:
        yield
    finally:
        torch.expm1 = original


def fmt_shape(x):
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, (tuple, list)):
        parts = [fmt_shape(i) for i in x if isinstance(i, (torch.Tensor, tuple, list))]
        return parts[0] if len(parts) == 1 else "(" + ", ".join(parts) + ")"
    return "-"


def collect_io_shapes(model, dummy_input):
    """
    Forward hooks on every module -- records the ACTUAL tensor shapes
    flowing in and out, rather than what the constructor args imply.
    This is what catches the transposes inside SSMBlock (the conv sees
    (batch, d_inner, T), everything else sees (batch, T, d_inner)).
    """
    shapes = OrderedDict()
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            shapes[name] = (fmt_shape(inputs), fmt_shape(output))
        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in handles:
        h.remove()
    return shapes


def print_module_tree(model, shapes, title):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(f"{'layer (type)':<46}{'output shape':<22}{'params':>12}")
    print("-" * 100)

    for name, module in model.named_modules():
        depth = name.count('.')
        label = name.split('.')[-1] if name else '(root)'
        indent = "  " * depth
        cls = module.__class__.__name__

        # params owned DIRECTLY by this module, not by its children --
        # otherwise every container would double-count its subtree
        own = sum(p.numel() for p in module.parameters(recurse=False))
        own_str = f"{own:,}" if own else "-"

        out_shape = shapes.get(name, ("-", "-"))[1]
        print(f"{indent + label + ' (' + cls + ')':<46}{out_shape:<22}{own_str:>12}")

        # direct nn.Parameters (A_log, D, B, C, dt) aren't submodules,
        # so named_modules() never surfaces them -- list them explicitly
        for pname, p in module.named_parameters(recurse=False):
            if pname in dict(module.named_children()):
                continue
            leaf_indent = "  " * (depth + 1)
            print(f"{leaf_indent + '· ' + pname:<46}{str(tuple(p.shape)):<22}{p.numel():>12,}")


def print_param_table(modules):
    print(f"\n{'=' * 100}")
    print("  PARAMETER INVENTORY")
    print(f"{'=' * 100}")
    print(f"{'parameter':<52}{'shape':<22}{'numel':>12}{'KB fp32':>12}")
    print("-" * 100)

    total = 0
    for prefix, m in modules:
        for name, p in m.named_parameters():
            full = f"{prefix}.{name}"
            kb = p.numel() * 4 / 1024
            print(f"{full:<52}{str(tuple(p.shape)):<22}{p.numel():>12,}{kb:>12.1f}")
            total += p.numel()
    print("-" * 100)
    print(f"{'TOTAL':<52}{'':<22}{total:>12,}{total * 4 / 1024:>12.1f}")
    return total


def print_memory_summary(total_params, cfg):
    """
    The number the thesis actually cares about -- deployability is the
    primary claim, so weight footprint is a headline result, not trivia.
    Activation memory is NOT included here (that needs Phase 5's
    per-tensor range work); this is weights only.
    """
    print(f"\n{'=' * 100}")
    print("  WEIGHT FOOTPRINT (weights only -- excludes activations & scoring head)")
    print(f"{'=' * 100}")
    for label, nbytes in [("fp32", 4), ("fp16", 2), ("int8", 1)]:
        kb = total_params * nbytes / 1024
        print(f"  {label:<8}{total_params:>12,} params{kb:>12.1f} KB{kb / 1024:>10.2f} MB")

    d_model = cfg['model']['d_model']
    print(f"\n  Option 3 scoring head, on top of the above:")
    print(f"    centroid (Euclidean)     {d_model:>6} floats{d_model * 4 / 1024:>10.1f} KB")
    print(f"    + inverse covariance     {d_model * d_model:>6} floats{d_model * d_model * 4 / 1024:>10.1f} KB  (Mahalanobis)")
    print(f"    or 16 k-means references {16 * d_model:>6} floats{16 * d_model * 4 / 1024:>10.1f} KB  (clustered kNN)")


def export_onnx(model, head, cfg, T_small=8):
    class BackboneWithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x, mode='sequence'))

    out_dir = PROJECT_ROOT / 'runs' / 'graph'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'model_T{T_small}.onnx'

    wrapped = BackboneWithHead(model, head).eval()
    dummy = torch.randn(1, T_small, cfg['model']['n_mels'])

    with _onnx_op_shims():
        torch.onnx.export(
            wrapped, (dummy,), str(path),
            input_names=['logmel'], output_names=['pred_residual'],
            opset_version=14, do_constant_folding=False,
            dynamo=False,
        )
    print(f"\n  ONNX graph written to {path}")
    print(f"  Open it at https://netron.app -- scan is unrolled {T_small}x; "
          f"at the real T=344 the topology is the same, just {344 // T_small}x longer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--T', type=int, default=None, help='sequence length for the shape trace')
    args = parser.parse_args()

    cfg = load_config()
    n_mels = cfg['model']['n_mels']
    d_model = cfg['model']['d_model']

    model = SSMBackbone(**cfg['model'])
    head = PredictionHead(d_model, n_mels)

    if args.ckpt:
        ckpt = torch.load(PROJECT_ROOT / args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        head.load_state_dict(ckpt['head'])
        print(f"loaded weights from {args.ckpt}")
    else:
        print("no --ckpt given: showing randomly initialised model (shapes are identical either way)")

    # 344 frames is the real clip length; shorter just to keep the
    # shape-trace forward pass quick, since the scan is sequential
    T = args.T if args.T else 32
    dummy = torch.randn(1, T, n_mels)

    print(f"\ninput: (batch=1, T={T}, n_mels={n_mels})    [real clips are T=344]")

    backbone_shapes = collect_io_shapes(model, dummy)
    print_module_tree(model, backbone_shapes, "SSMBackbone")

    with torch.no_grad():
        seq_out = model(dummy, mode='sequence')
    head_shapes = collect_io_shapes(head, seq_out)
    print_module_tree(head, head_shapes, "PredictionHead")

    total = print_param_table([('backbone', model), ('head', head)])
    print_memory_summary(total, cfg)

    print(f"\n  pooling mode: '{model.pooling}'  ->  pooled embedding "
          f"({2 * d_model if model.pooling == 'concat_mean_last' else d_model},) per clip")

    if args.onnx:
        export_onnx(model, head, cfg, T_small=T)