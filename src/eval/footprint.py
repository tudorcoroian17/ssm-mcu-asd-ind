import pandas as pd
import torch

from runs.compute_hash import train_config_hash
from src.config import load_config, PROJECT_ROOT
from src.models.backbone import SSMBackbone


def profile_mcu_memory(model, input_shape):
    total_weights = sum(p.numel() for p in model.parameters())
    total_biases = sum(b.numel() for b in model.parameters())
    flash_footprint_int8 = total_weights + total_biases

    activation_sizes = []
    hooks = []

    def hook_fn(module, input_layer, output_layer):
        if isinstance(output_layer, torch.Tensor):
            # RAM must hold both input and output tensors during layer execution
            in_bytes = sum(i.numel() for i in input_layer if isinstance(i, torch.Tensor))
            out_bytes = output_layer.numel()
            activation_sizes.append(in_bytes + out_bytes)
        elif isinstance(output_layer, tuple):
            in_bytes = sum(i.numel() for i in input_layer if isinstance(i, torch.Tensor))
            out_bytes = sum(o.numel() for o in output_layer if isinstance(o, torch.Tensor))
            activation_sizes.append(in_bytes + out_bytes)

    # Register hooks across all operational layers
    for layer in model.modules():
        # Skip wrapper modules to avoid double-counting
        if len(list(layer.children())) == 0:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Run a single dummy inference pass to trigger hooks
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        model(dummy_input, mode='pooled')

    # Remove hooks to clean up the model
    for hook in hooks:
        hook.remove()

    # Peak RAM is dictated by the single bottleneck layer
    peak_ram_bytes = max(activation_sizes) if activation_sizes else 0

    return flash_footprint_int8, peak_ram_bytes

def run_profiler(held_out_case: int, config: dict):
    dir_name = train_config_hash(config, held_out_case)
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name

    results = []

    model = SSMBackbone(**cfg['model'])
    ckpt = torch.load(base_dir / 'ckpt.pt')
    model.load_state_dict(ckpt['model'])
    model.eval()
    for pooling in ['mean', 'max', 'concat_mean_last']:
        x = torch.randn(1, 344, cfg['features']['n_mels'])  # (batch, T, n_mels) -- 344 is a real clip
        model.pooling = pooling

        flash_footprint_int8, peak_ram_bytes = profile_mcu_memory(model, x.shape)
        results.append({
            'pooling': pooling,
            'flash_footprint_fp32': flash_footprint_int8 * 4,
            'flash_footprint_int8': flash_footprint_int8,
            'peak_ram_fp32_bytes': peak_ram_bytes * 4,
            'peak_ram_int8_bytes': peak_ram_bytes,
        })

    data = [(r['pooling'],
             r['flash_footprint_fp32'] // 1024,
             r['flash_footprint_int8'] // 1024,
             r['peak_ram_fp32_bytes'] // 1024,
             r['peak_ram_int8_bytes'] // 1024)
            for r in results]
    df = pd.DataFrame(data, columns=['pooling',
                                     'flash_footprint_fp32_kb',
                                     'flash_footprint_int8_kb',
                                     'peak_ram_fp32_kb',
                                     'peak_ram_int8_kb'])
    df.to_csv(base_dir / 'footprint.csv', index=False)

if __name__ == '__main__':
    cfg = load_config()

    cases = [1, 2, 3, 4]
    for case in cases:
        print(f'\n=== held_out_case {case} ===')
        run_profiler(case, cfg)