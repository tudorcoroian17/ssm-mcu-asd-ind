import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from runs.compute_hash import train_config_hash
from src.config import PROJECT_ROOT, load_config
from src.data.folds import get_fold
from src.features.baselines import load_fold_clips
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone


def run_check(held_out_case, cfg, device):
    dir_name = train_config_hash(cfg, held_out_case)
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

    # hook dt_proj on every layer to capture real, data-dependent per-timestep delta
    captured = {}
    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = out.detach()
        return hook

    handles = [block.dt_proj.register_forward_hook(make_hook(f'layer{i}'))
               for i, block in enumerate(model.blocks) if block.selective]
    with torch.no_grad():
        model(x_val_batch, mode='sequence')

    for h in handles:
        h.remove()

    hop, sr = cfg['features']['hop_length'], cfg['features']['sample_rate']

    results = []
    for i, block in enumerate(model.blocks):
        delta = F.softplus(captured[f'layer{i}'].mean(dim=(0, 1)))
        A = -torch.exp(block.A_log)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
        half_life_frames = torch.log(torch.tensor(0.5, device=device)) / torch.log(A_bar.clamp(max=0.999999))
        half_life_ms = half_life_frames * hop / sr * 1000
        print(f"layer {i}: half-life {half_life_ms.min():.1f}-{half_life_ms.max():.1f} ms "
              f"(median {half_life_ms.median():.1f} ms)")
        results.append((
            i,
            half_life_ms.min().item(),
            half_life_ms.max().item(),
            half_life_ms.median().item(),
            half_life_ms.mean().item(),
            half_life_ms.std().item()
        ))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, block in enumerate(model.blocks):
        with torch.no_grad():
            delta = F.softplus(captured[f'layer{i}']).mean(dim=(0, 1))
            A = -torch.exp(block.A_log)
            A_bar = torch.exp(delta.unsqueeze(-1) * A)
            half_life_frames = torch.log(torch.tensor(0.5, device=device)) / torch.log(A_bar.clamp(max=0.999999))
            half_life_ms = (half_life_frames * hop / sr * 1000).flatten().cpu().numpy()

            # floor near-zero/underflowed values so log-scale bins can still show them,
            # rather than silently dropping them (log(0) is undefined)
            plotted = np.clip(half_life_ms, 0.01, None)
            bins = np.logspace(np.log10(0.01), np.log10(plotted.max()), 40)

            ax = axes[i]
            ax.hist(plotted, bins=bins, color='#2a78d6', edgecolor='white', linewidth=0.3)
            ax.set_xscale('log')
            ax.axvline(128, color='red', linestyle='--', linewidth=1, label='conv window (128ms)')
            ax.set_title(f'layer {i}  (median={np.median(half_life_ms):.1f}ms)')
            ax.set_xlabel('half-life (ms, log scale)')
            ax.set_ylabel('count')
            ax.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(str(base_dir / 'half_life_histogram.png'), dpi=150)
    df = pd.DataFrame(results, columns=['layer', 'min_ms', 'max_ms', 'median_ms', 'mean_ms', 'std_ms'])
    df.to_csv(base_dir / 'decay_half_life.csv', index=False)

if __name__ == "__main__":
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cases = [1, 2, 3, 4]
    for case in cases:
        print(f'\n=== held_out_case {case} ===')
        run_check(case, cfg, device)