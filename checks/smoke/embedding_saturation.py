import numpy as np
import torch
from scipy.spatial.distance import pdist

from src.config import load_config, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from src.eval.auc_pauc import get_embeddings, score_euclidean

TOP_N = 25


def run(held_out_case, checkpoint, cfg, device):
    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels=n_mels)

    X_train = load_fold_clips(fold['train'], mean, std)
    X_test = load_fold_clips(fold['test'], mean, std)
    test_labels = (fold['test']['label'].values == 'anomaly').astype(int)

    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(PROJECT_ROOT / 'runs' / f'case{held_out_case}' / f'{checkpoint}', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.pooling = 'mean'

    train_emb = get_embeddings(model, X_train, device)
    test_emb = get_embeddings(model, X_test, device)
    scores, _ = score_euclidean(train_emb, test_emb, cfg['seed'])
    clip_rms = np.sqrt((X_test ** 2).mean(axis=(1, 2)))

    order = np.argsort(-scores)
    top = order[:TOP_N]
    rest = order[TOP_N:]

    print(f"\n=== case{held_out_case} ===")
    print(f"  score vs clip RMS, all test clips:  r={np.corrcoef(scores, clip_rms)[0, 1]:+.4f}")

    # If these top clips occupy ONE point, their pairwise distances collapse
    # toward zero while the rest of the test set stays spread out.
    print(f"  mean pairwise distance, top-{TOP_N} embeddings:   {pdist(test_emb[top]).mean():.4f}")
    print(f"  mean pairwise distance, remaining embeddings:  {pdist(test_emb[rest]).mean():.4f}")

    print(f"  embedding L2 norm, top-{TOP_N}:   {np.linalg.norm(test_emb[top], axis=1).mean():.4f}")
    print(f"  embedding L2 norm, remaining:  {np.linalg.norm(test_emb[rest], axis=1).mean():.4f}")

    print(f"  top-{TOP_N} clips: {test_labels[top].sum()} anomaly, {TOP_N - test_labels[top].sum()} normal")
    print(f"  their clip RMS: mean={clip_rms[top].mean():.4f} vs rest={clip_rms[rest].mean():.4f}")


if __name__ == "__main__":
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cases = [1, 2, 3, 4]
    checkpoints = ['4f3ffab6e66e.pt', 'ec6602b5742c.pt', '554da113910d.pt', '8ffeae10cff6.pt']

    for case in cases:
        run(case, checkpoints[case - 1], cfg, device)