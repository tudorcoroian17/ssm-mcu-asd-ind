"""
Netron-style inspector for SSMBackbone + PredictionHead.

    python -m helpers.vis_inspect_model                         # architecture + params, random init
    python -m helpers.vis_inspect_model --ckpt runs/case1/ckpt.pt
    python -m helpers.vis_inspect_model --onnx                  # also write an ONNX graph
    python -m helpers.vis_inspect_model --onnx --onnx-T 1 --single-layer --no-head --simplify --hide-ops

Why the ONNX export defaults to a short T: _scan is a Python for-loop, so ONNX
tracing unrolls it -- one copy of the scan body per timestep. The graph
TOPOLOGY is identical at any T (same layers, same widths); a short T just keeps
it small enough for Netron to draw legibly. At T=1 there is no h_{t-1} -> h_t
edge to trace, so the recurrence itself renders as a flat chain rather than a
loop -- T=1 is for seeing the non-recurrent wiring cleanly (in_proj -> chunk ->
conv -> x_proj -> discretize), not for seeing the recurrence, which no ONNX
export can show regardless of T.

RMSNorm is exported as an ONNX local function (export_modules_as_functions),
so Netron collapses it to a single labelled box instead of its six primitive
ops, double-click to expand. Needs opset >= 15. SSMBlock is deliberately NOT
collapsed -- that internal wiring is the thing worth seeing; only RMSNorm's
clutter is worth hiding.

Two further, optional cleanup passes on the exported .onnx file:

  --simplify   runs onnxsim, which folds the Shape/Gather/Unsqueeze/Cast chains
               the tracer inserts to compute dimensions defensively, even
               though T is fixed at export time here. REAL simplification --
               the output stays a valid, executable graph. Try this first.

  --hide-ops   strips whatever Squeeze/Unsqueeze/Gather/Cast/Identity nodes
               remain after --simplify, by rewiring around them and deleting
               them. DIAGRAM ONLY: unlike --simplify, this can remove a Gather
               that does real indexing, which changes what the graph computes.
               Saved to a separate _diagram_only.onnx file. Never load this in
               onnxruntime, never use it for parity checking or as a C-port
               reference -- parity_vectors.npz is the only artifact that plays
               that role.
"""
import argparse
import contextlib
from collections import OrderedDict

import torch
import torch.nn as nn

from src.config import load_config, PROJECT_ROOT
from src.models.backbone import SSMBackbone, RMSNorm
from src.models.ssm_block import SSMBlock
from src.models.heads import PredictionHead

# opset for a plain export; bumped to at least 15 automatically when
# collapse_modules=True, since ONNX local functions need opset >= 15.
BASE_OPSET = 14

COSMETIC_OPS = ('Squeeze', 'Unsqueeze', 'Gather', 'Cast', 'Identity')


@contextlib.contextmanager
def _onnx_op_shims():
    """
    ONNX has no Expm1 op, and the TorchScript exporter has no symbolic to
    decompose it, so tracing dies on ssm_block's exact-ZOH branch.
    expm1(x) == exp(x) - 1 mathematically; expm1 exists purely for precision
    near x=0, which is irrelevant for a graph we're drawing rather than
    running. Scoped to the export call only -- training and eval keep the real
    expm1, untouched.
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
    Forward hooks on every module -- records the ACTUAL tensor shapes flowing
    in and out, rather than what the constructor args imply. This is what
    catches the transposes inside SSMBlock (the conv sees (batch, d_inner, T),
    everything else sees (batch, T, d_inner)).
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

        # direct nn.Parameters (A_log, D, B, C, dt) aren't submodules, so
        # named_modules() never surfaces them -- list them explicitly
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
    The number the thesis actually cares about -- deployability is the primary
    claim, so weight footprint is a headline result, not trivia. Activation
    memory is NOT included here (that needs the streaming RAM estimate); this
    is weights only.
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


def export_onnx(model, head, cfg, T_small=4, no_head=False, single_layer=False,
                collapse_modules=True):
    class BackboneWithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x, mode='sequence'))

    class BackboneWithoutHead(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            return self.backbone(x, mode='sequence')

    out_dir = PROJECT_ROOT / 'runs' / 'graph'
    out_dir.mkdir(parents=True, exist_ok=True)

    if single_layer:
        # One block, same widths, for a legible figure. The other n_layers-1
        # blocks are structurally identical, so showing one is faithful; note
        # that in the caption. Random init is fine -- topology, not weights,
        # is the point of this export.
        export_backbone = SSMBackbone(**dict(cfg['model'], n_layers=1)).eval()
    else:
        export_backbone = model

    tag = f"T{T_small}"
    tag += "_1layer" if single_layer else ""
    tag += "_no_head" if no_head else ""
    path = out_dir / f"model_{tag}.onnx"

    wrapped = (BackboneWithoutHead(export_backbone).eval() if no_head
              else BackboneWithHead(export_backbone, head).eval())
    dummy = torch.randn(1, T_small, cfg['model']['n_mels'])

    # ONNX local functions keep a chosen nn.Module's forward pass as ONE
    # collapsed node instead of flattening it to primitive ops. RMSNorm only:
    # SSMBlock's internals (conv, x_proj, discretize, scan, gate) are the
    # wiring worth seeing, so it is left expanded. Needs opset >= 15.
    export_kwargs = {}
    opset = BASE_OPSET
    if collapse_modules:
        export_kwargs['export_modules_as_functions'] = {RMSNorm}
        opset = max(BASE_OPSET, 15)

    with _onnx_op_shims():
        torch.onnx.export(
            wrapped, (dummy,), str(path),
            input_names=['logmel'], output_names=['pred_residual'],
            opset_version=opset, do_constant_folding=False,
            dynamo=False,
            **export_kwargs,
        )

    print(f"\n  ONNX graph written to {path}")
    print(f"  opset {opset}"
          + (", RMSNorm collapsed as a local function" if collapse_modules else ""))
    print(f"  Open it at https://netron.app -- scan is unrolled {T_small}x; "
          f"at the real T=344 the topology is the same, just {344 // T_small}x longer.")
    if T_small == 1:
        print("  T=1: no h_{t-1} -> h_t edge exists to trace, so the recurrence "
              "renders as a flat chain, not a loop. Good for seeing the "
              "non-recurrent wiring (in_proj/chunk/conv/x_proj/discretize); "
              "the recurrence itself needs a hand-drawn diagram, not ONNX.")
    return path


def simplify_onnx(path):
    """
    Runs onnxsim, which folds the Shape/Gather/Unsqueeze/Cast chains PyTorch's
    tracer inserts to compute dimensions defensively, even when the shape is
    fixed at trace time -- which it is here (T_small is a concrete int). This
    is a REAL simplification: the output stays a valid, executable graph, so
    unlike hide_cosmetic_ops below there is no correctness caveat. Try this
    before reaching for the diagram-only strip.

    Will not remove a genuine indexing Gather (real selection, not
    bookkeeping) -- only redundant shape computation.
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("\n  onnxsim not installed -- skipping simplification.")
        print("  pip install onnxsim   (then re-run with --simplify)")
        return path

    model = onnx.load(str(path))
    before = len(model.graph.node)
    simplified, ok = simplify(model)
    if not ok:
        print("\n  onnxsim could not verify the simplified graph -- keeping original.")
        return path

    out_path = path.with_name(path.stem + '_simplified.onnx')
    onnx.save(simplified, str(out_path))
    after = len(simplified.graph.node)
    print(f"\n  onnxsim: {before} -> {after} nodes")
    print(f"  {out_path}")
    return out_path


def hide_cosmetic_ops(path, op_types=COSMETIC_OPS):
    """
    Rewires around nodes of the given op types and deletes them, purely to
    shrink the picture further than --simplify can. For each matched node,
    every downstream consumer of its output is repointed at its first input,
    then the node is dropped. Runs to a fixed point, since removing one node
    can expose another target node immediately upstream (e.g. a Cast feeding
    a Squeeze).

    DIAGRAM ONLY. Unlike simplify_onnx, this is not guaranteed to preserve
    what the graph computes -- Gather in particular can do real indexing, and
    splicing it out changes that. The result must never be loaded by
    onnxruntime, used for parity checking, or treated as a C-port reference;
    parity_vectors.npz is the only artifact that plays that role. Saved with a
    _diagram_only suffix so it can't be mistaken for a real export.
    """
    import onnx

    model = onnx.load(str(path))
    graph = model.graph
    graph_outputs = {o.name for o in graph.output}

    removed_total = 0
    while True:
        rewrite = {}
        keep = []
        for node in graph.node:
            if (node.op_type in op_types
                    and len(node.output) == 1
                    and node.output[0] not in graph_outputs
                    and len(node.input) >= 1):
                rewrite[node.output[0]] = node.input[0]
                continue
            keep.append(node)

        if not rewrite:
            break

        def resolve(name, _seen=None):
            _seen = _seen or set()
            while name in rewrite and name not in _seen:
                _seen.add(name)
                name = rewrite[name]
            return name

        for node in keep:
            for i, inp in enumerate(node.input):
                node.input[i] = resolve(inp)

        del graph.node[:]
        graph.node.extend(keep)
        removed_total += len(rewrite)

    # drop initializers nothing references any more (axes/indices constants
    # that fed a now-deleted Squeeze/Gather)
    used = {inp for n in graph.node for inp in n.input} | graph_outputs
    kept_init = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_init)

    out_path = path.with_name(path.stem + '_diagram_only.onnx')
    onnx.save(model, str(out_path))
    print(f"\n  {removed_total} cosmetic nodes stripped ({', '.join(op_types)})")
    print(f"  DIAGRAM ONLY, not a valid graph: {out_path}")
    print(f"  Never load this in onnxruntime or use it for parity/C-port work --")
    print(f"  parity_vectors.npz is the only artifact that plays that role.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--T', type=int, default=None,
                        help='sequence length for the printed shape trace (not the ONNX export)')
    parser.add_argument('--onnx-T', type=int, default=4,
                        help='sequence length for the ONNX export only; small keeps '
                             'the unrolled scan legible')
    parser.add_argument('--no-head', action='store_true')
    parser.add_argument('--single-layer', action='store_true',
                        help='export a 1-layer backbone (same widths) instead of the full stack')
    parser.add_argument('--no-collapse', action='store_true',
                        help='disable RMSNorm local-function collapsing; export fully flattened')
    parser.add_argument('--simplify', action='store_true',
                        help='run onnxsim on the exported graph (real simplification, '
                             'stays valid/executable) -- try this first')
    parser.add_argument('--hide-ops', action='store_true',
                        help='strip Squeeze/Unsqueeze/Gather/Cast/Identity for a cleaner '
                             'picture -- DIAGRAM ONLY, saves a separate file, never use '
                             'it for parity checks')
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

    # 344 frames is the real clip length; shorter just to keep the shape-trace
    # forward pass quick, since the scan is sequential
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
        onnx_path = export_onnx(model, head, cfg, T_small=args.onnx_T, no_head=args.no_head,
                                single_layer=args.single_layer,
                                collapse_modules=not args.no_collapse)
        if args.simplify:
            onnx_path = simplify_onnx(onnx_path)
        if args.hide_ops:
            hide_cosmetic_ops(onnx_path)