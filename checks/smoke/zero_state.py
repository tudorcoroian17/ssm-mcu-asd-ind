import functools

import pandas as pd
import torch

from runs.compute_hash import train_config_hash
from src.config import PROJECT_ROOT, load_config
from src.data.folds import get_fold
from src.features.baselines import load_fold_clips, compute_baselines
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone
from src.models.heads import PredictionHead
from src.train import compute_loss


def scan_zero_state(block, A_bar, B_bar, C, u, return_h_trace=False):
    """
            Same math as SSMBlock._scan, except h is discarded after every
            step -- removes all cross-timestep memory. Diagnostic only; never
            used in training or in the real, parity-validated _scan().
        """
    batch, T = u.shape[0], u.shape[1]
    h = torch.zeros(batch, block.d_inner, block.d_state, device=u.device, dtype=u.dtype)
    A_bar_ts, B_bar_ts, u_ts = A_bar.unbind(1), B_bar.unbind(1), u.unbind(1)
    C_ts = C.unbind(1) if block.selective else None
    ys = []
    for t in range(T):
        Bu_t = B_bar_ts[t] * u_ts[t].unsqueeze(-1)
        h = torch.addcmul(Bu_t, A_bar_ts[t], h)
        C_t = C_ts[t].unsqueeze(1) if block.selective else C
        ys.append((h * C_t).sum(dim=-1))
        h = torch.zeros_like(h)  # <--- forget everything before next step
    y = torch.stack(ys, dim=1)
    return (y, None) if return_h_trace else y

def run_check(held_out_case, cfg, device):
    dir_name = train_config_hash(cfg, held_out_case)
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name

    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(base_dir / 'ckpt.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    fold = get_fold(held_out_case)
    k = cfg['training']['horizon_k']

    n_mels = cfg['features']['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels)
    val_normal_set = fold['val'][fold['val']['label'] == 'normal']
    x_val = load_fold_clips(val_normal_set, mean, std)
    x_val_batch_1 = torch.from_numpy(x_val[:32]).float().to(device)
    x_val_batch_2 = torch.from_numpy(x_val[256:288]).float().to(device)
    x_val_batch_3 = torch.from_numpy(x_val[288:320]).float().to(device)

    head = PredictionHead(cfg['model']['d_model'], cfg['model']['n_mels']).to(device)
    head.load_state_dict(ckpt['head'])
    head.eval()

    with torch.no_grad():
        real_loss_1 = compute_loss(model, head, x_val_batch_1, k).item()
        real_loss_2 = compute_loss(model, head, x_val_batch_2, k).item()
        real_loss_3 = compute_loss(model, head, x_val_batch_3, k).item()

    # swap in the zero-state scan on every block
    originals = [block._scan for block in model.blocks]
    for block in model.blocks:
        block._scan = functools.partial(scan_zero_state, block)

    with torch.no_grad():
        zeroed_loss_1 = compute_loss(model, head, x_val_batch_1, k).item()
        zeroed_loss_2 = compute_loss(model, head, x_val_batch_2, k).item()
        zeroed_loss_3 = compute_loss(model, head, x_val_batch_3, k).item()

    # restore the real scan
    for block, orig in zip(model.blocks, originals):
        block._scan = orig

    # same batch, so this is the exact right baseline to compare both against
    persistence_1 = compute_baselines(x_val_batch_1.cpu().numpy(), k=k)["mse_persistence"]
    skill_real_1 = 1 - real_loss_1 / persistence_1
    skill_zeroed_1 = 1 - zeroed_loss_1 / persistence_1

    persistence_2 = compute_baselines(x_val_batch_2.cpu().numpy(), k=k)["mse_persistence"]
    skill_real_2 = 1 - real_loss_2 / persistence_2
    skill_zeroed_2 = 1 - zeroed_loss_2 / persistence_2

    persistence_3 = compute_baselines(x_val_batch_3.cpu().numpy(), k=k)["mse_persistence"]
    skill_real_3 = 1 - real_loss_3 / persistence_3
    skill_zeroed_3 = 1 - zeroed_loss_3 / persistence_3

    batch_1 = (1, f'[0:32]', real_loss_1, zeroed_loss_1, persistence_1, skill_real_1, skill_zeroed_1)
    batch_2 = (9, f'[256:288]', real_loss_2, zeroed_loss_2, persistence_2, skill_real_2, skill_zeroed_2)
    batch_3 = (10, f'[288:320]', real_loss_3, zeroed_loss_3, persistence_3, skill_real_3, skill_zeroed_3)

    df = pd.DataFrame([batch_1, batch_2, batch_3], columns=['#batch', 'limits', 'real_loss', 'zeroed_loss', 'persistence', 'skill_real', 'skill_zeroed'])
    df.to_csv(base_dir / 'zero_state.csv', index=False)
    print(df)

if __name__ == "__main__":
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cases = [1, 2, 3, 4]
    for case in cases:
        print(f'\n=== held_out_case {case} ===')
        run_check(case, cfg, device)