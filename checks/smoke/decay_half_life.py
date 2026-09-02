"""
Measures the effective memory horizon of each SSM block.

Half-life is how many frames it takes a state contribution to decay to half its
magnitude, converted to milliseconds. It answers whether the model is using
long-range state at all or behaving like a short convolution: if half-lives sit
below the depthwise conv window (d_conv frames, 128 ms at the default hop), the
recurrence is not contributing memory the conv could not supply.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.config import PROJECT_ROOT, load_config_by_name
from src.data.folds import get_fold
from src.features.baselines import load_fold_clips
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone


def mean_delta(block, captured, layer_key):
    """
    Returns the mean per-channel delta, shape (d_inner,).

    Softplus is applied before averaging, not after. The model applies softplus
    per timestep and discretizes with the result, so the mean of the deltas
    actually used is mean(softplus(x)). Applying softplus to the mean
    pre-activation gives a different number, because softplus is convex, and
    it corresponds to nothing the model computes.
    """
    if block.selective:
        return F.softplus(captured[layer_key]).mean(dim=(0, 1))
    # Fixed branch: dt is a learned per-channel constant with no data
    # dependence, so there is no hook output and nothing to average over.
    return F.softplus(block.dt)


def half_life_ms(block, delta, hop, sr):
    """
    Converts a per-channel delta into per-(channel, state) half-life in ms.

    Calls block.discretize() rather than reimplementing A_bar, so the ZOH and
    Euler formulas cannot drift from the model. Euler's A_bar can be negative
    (clamped at -0.9), so the decay rate is the magnitude; log of the raw value
    would be NaN for every Euler config.
    """
    A = -torch.exp(block.A_log)
    B_dummy = torch.zeros(block.d_state, device=A.device, dtype=A.dtype)
    A_bar, _ = block.discretize(delta, A, B_dummy)

    decay = A_bar.abs().clamp(min=1e-12, max=0.999999)
    frames = torch.log(torch.tensor(0.5, device=decay.device)) / torch.log(decay)
    return frames * hop / sr * 1000


def run_check(held_out_case, cfg, device, dir_name):
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name

    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(base_dir / 'ckpt.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels)
    val_normal_set = fold['val'][fold['val']['label'] == 'normal']
    x_val = load_fold_clips(val_normal_set, mean, std)
    x_val_batch = torch.from_numpy(x_val[:32]).float().to(device)

    # Hook dt_proj to capture the real, data-dependent per-timestep delta.
    # Only selective blocks have a dt_proj; fixed blocks are handled by
    # mean_delta() without a forward pass.
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = out.detach()
        return hook

    handles = [block.dt_proj.register_forward_hook(make_hook(f'layer{i}'))
               for i, block in enumerate(model.blocks) if block.selective]
    if handles:
        with torch.no_grad():
            model(x_val_batch, mode='sequence')
    for h in handles:
        h.remove()

    hop, sr = cfg['features']['hop_length'], cfg['features']['sample_rate']
    conv_window_ms = cfg['model']['d_conv'] * hop / sr * 1000

    # One pass produces both the table and the figure, so they cannot disagree.
    n_blocks = len(model.blocks)
    fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 4), squeeze=False)
    axes = axes.flatten()

    results = []
    for i, block in enumerate(model.blocks):
        with torch.no_grad():
            delta = mean_delta(block, captured, f'layer{i}')
            hl = half_life_ms(block, delta, hop, sr)

        flat = hl.flatten().cpu().numpy()
        print(f'layer {i}: half-life {flat.min():.1f}-{flat.max():.1f} ms '
              f'(median {np.median(flat):.1f} ms)')
        results.append((
            i,
            bool(block.selective),
            block.discretization,
            float(flat.min()),
            float(flat.max()),
            float(np.median(flat)),
            float(flat.mean()),
            float(flat.std()),
            float((flat < conv_window_ms).mean()),
        ))

        # Floor near-zero and underflowed values so log-scale bins can show
        # them, rather than silently dropping them: log(0) is undefined.
        plotted = np.clip(flat, 0.01, None)
        hi = max(float(plotted.max()), 0.02)
        bins = np.logspace(np.log10(0.01), np.log10(hi), 40)

        ax = axes[i]
        ax.hist(plotted, bins=bins, color='#2a78d6', edgecolor='white', linewidth=0.3)
        ax.set_xscale('log')
        ax.axvline(conv_window_ms, color='red', linestyle='--', linewidth=1,
                   label=f'conv window ({conv_window_ms:.0f}ms)')
        ax.set_title(f'layer {i}  (median={np.median(flat):.1f}ms)')
        ax.set_xlabel('half-life (ms, log scale)')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(base_dir / 'half_life_histogram.png'), dpi=150)
    plt.close(fig)

    df = pd.DataFrame(results, columns=[
        'layer', 'selective', 'discretization', 'min_ms', 'max_ms',
        'median_ms', 'mean_ms', 'std_ms', 'frac_below_conv_window'])
    df.to_csv(base_dir / 'decay_half_life.csv', index=False)


if __name__ == "__main__":
    configs_root = PROJECT_ROOT / 'configs' / 'ablation'
    runs_root = PROJECT_ROOT / 'runs'
    configs_manifest = pd.read_csv(configs_root / '000_config_manifest.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for index, row in configs_manifest.iterrows():
        case = int(row['held_out_case'])
        model_hash = row['model_hash']
        ckpt_path = runs_root / f'case{case}' / model_hash / 'ckpt.pt'

        if not ckpt_path.exists():
            continue

        config_file = load_config_by_name(row['config_name'])
        print(f'\n=== decay half-life for case {case} -> model {model_hash} ===')
        run_check(case, config_file, device, model_hash)