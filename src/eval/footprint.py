import pandas as pd
import torch

from runs.compute_hash import train_config_hash
from src.config import load_config, PROJECT_ROOT, load_config_by_name
from src.models.backbone import SSMBackbone


def profile_activation_trace(model, input_shape):
    """
    Hook-based trace. Only sees registered nn.Module boundaries -- in_proj,
    conv, x_proj, dt_proj, out_proj, RMSNorm. Cannot and will not see inside
    SSMBlock._scan()/discretize(), which are plain methods. This number is
    a lower bound on module-level activation size, useful as a sanity check,
    NOT a peak-RAM estimate -- see estimate_streaming_ram() for that.
    """
    activation_sizes = []
    hooks = []

    def hook_fn(module, input_layer, output_layer):
        if isinstance(output_layer, torch.Tensor):
            in_bytes = sum(i.numel() for i in input_layer if isinstance(i, torch.Tensor))
            out_bytes = output_layer.numel()
            activation_sizes.append(in_bytes + out_bytes)
        elif isinstance(output_layer, tuple):
            in_bytes = sum(i.numel() for i in input_layer if isinstance(i, torch.Tensor))
            out_bytes = sum(o.numel() for o in output_layer if isinstance(o, torch.Tensor))
            activation_sizes.append(in_bytes + out_bytes)

    for layer in model.modules():
        if len(list(layer.children())) == 0:
            hooks.append(layer.register_forward_hook(hook_fn))

    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        model(dummy_input, mode='pooled')

    for hook in hooks:
        hook.remove()

    return max(activation_sizes) if activation_sizes else 0


def estimate_streaming_ram(cfg, n_scratch_buffers=4, reuse_across_layers=True):
    """
    Analytical peak-RAM estimate for STREAMING, per-timestep MCU execution --
    not a trace of this repo's GPU-parallel forward(). forward()'s own
    comment names the gap: on MCU, only A_bar_t/B_bar_t (per timestep)
    should exist, not the full (T, d_inner, d_state) tensor this code
    materializes for GPU throughput.

    Caveat: assumes NAIVE (unfused) per-timestep scratch -- n_scratch_buffers
    separate (d_inner, d_state) arrays. MambaLite-Micro's 83% peak-memory
    reduction (master doc Section 3) comes from fusing exactly this scratch
    via operator fusion. This function reports the PRE-fusion ceiling --
    the honest Phase 1 number, and the one Phase 4/5 should improve on, not
    match.

    reuse_across_layers=True assumes scratch is freed and reused between
    layers within one frame's pass (correct embedded practice, not yet
    verified against an actual Phase 4 implementation). Set False for a
    worst-case bound if buffers are allocated per-layer instead.
    """
    m = cfg['model']
    d_model = m['d_model']
    d_inner = m['expand'] * d_model
    d_state = m['d_state']
    d_conv = m['d_conv']
    n_layers = m['n_layers']
    selective = m['selective']
    dt_rank = max(1, d_model // 16)

    report = {}
    for dtype, nbytes in [('fp32', 4), ('int8', 1)]:
        state_bytes = n_layers * d_inner * d_state * nbytes
        conv_buf_bytes = n_layers * (d_conv - 1) * d_inner * nbytes

        one_layer_scratch = n_scratch_buffers * d_inner * d_state * nbytes
        scan_scratch_bytes = one_layer_scratch if reuse_across_layers else one_layer_scratch * n_layers

        proj_scratch_bytes = (
            2 * d_inner +
            (dt_rank + 2 * d_state if selective else 0) +
            d_inner +
            d_model
        ) * nbytes

        peak = state_bytes + conv_buf_bytes + scan_scratch_bytes + proj_scratch_bytes

        report[dtype] = {
            'state_kb': state_bytes / 1024,
            'conv_buffer_kb': conv_buf_bytes / 1024,
            'scan_scratch_kb': scan_scratch_bytes / 1024,
            'projection_scratch_kb': proj_scratch_bytes / 1024,
            'peak_ram_kb': peak / 1024,
        }
    return report


def run_profiler(held_out_case: int, cfg: dict, dir_name):
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name

    model = SSMBackbone(**cfg['model'])
    ckpt = torch.load(base_dir / 'ckpt.pt')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # State/scan/conv costs are architecture-level, not pooling-level --
    # pooling only changes the final accumulator (d_model vs 2*d_model
    # floats), which is negligible next to the KB-scale scan buffers.
    # Computed once, not re-derived per pooling mode.
    streaming = estimate_streaming_ram(cfg)

    results = []
    for pooling in ['mean', 'max', 'concat_mean_last']:
        x = torch.randn(1, 344, cfg['features']['n_mels'])
        model.pooling = pooling

        traced_activation_bytes = profile_activation_trace(model, x.shape)
        pooled_out_floats = 2 * cfg['model']['d_model'] if pooling == 'concat_mean_last' else cfg['model']['d_model']

        total_weights = sum(p.numel() for p in model.parameters())
        flash_footprint_int8 = total_weights

        results.append({
            'pooling': pooling,
            'flash_footprint_fp32_kb': flash_footprint_int8 * 4 / 1024,
            'flash_footprint_int8_kb': flash_footprint_int8 / 1024,
            'traced_module_activation_fp32_kb': traced_activation_bytes * 4 / 1024,
            'streaming_peak_ram_fp32_kb': streaming['fp32']['peak_ram_kb'] + pooled_out_floats * 4 / 1024,
            'streaming_peak_ram_int8_kb': streaming['int8']['peak_ram_kb'] + pooled_out_floats / 1024,
        })

    df = pd.DataFrame(results)
    df.to_csv(base_dir / 'footprint.csv', index=False)
    print(df.to_string(index=False))
    print(f"\n  streaming RAM breakdown (fp32): {streaming['fp32']}")
    print(f"  (int8 scan/state figures are speculative -- see docstring)")


if __name__ == '__main__':
    configs_root = PROJECT_ROOT / 'configs' / 'ablation'
    runs_root = PROJECT_ROOT / 'runs'
    configs_manifest = pd.read_csv(configs_root / '000_config_manifest.csv')

    for index, row in configs_manifest.iterrows():
        case = int(row['held_out_case'])
        model_hash = row['model_hash']
        ckpt_path = runs_root / f'case{case}' / model_hash / 'ckpt.pt'

        if not ckpt_path.exists():
            print(f'model {model_hash} not cached')
            continue

        config_file = load_config_by_name(row['config_name'])

        print(f'\n=== generating embeddings for case {case} -> model {model_hash} ===')
        run_profiler(case, config_file, model_hash)