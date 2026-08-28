import numpy as np
import torch

from src.config import load_config, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from src.eval.auc_pauc import get_embeddings, score_euclidean


def describe(name, x):
    print(f"    {name:10s} n={len(x):4d}  mean={x.mean():.4f}  std={x.std():.4f}  "
          f"cv={x.std() / x.mean():.4f}  min={x.min():.4f}  max={x.max():.4f}")


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

    print(f"\n=== case{held_out_case} ===")
    print("  embedding-space scores (mean/euclidean):")
    describe('normal', scores[test_labels == 0])
    describe('anomaly', scores[test_labels == 1])

    # model-independent: does case N's own raw normal-clip pool vary less
    # than another case's, before the model ever sees it?
    normal_rows = fold['test'][fold['test']['label'] == 'normal']
    X_normal_raw = load_fold_clips(normal_rows, mean, std)
    clip_rms = np.sqrt((X_normal_raw ** 2).mean(axis=(1, 2)))
    print("  raw per-clip RMS energy (normalized features, model-independent):")
    describe('clip_rms', clip_rms)

    # which clips produced the extreme high-score outliers?
    order = np.argsort(-scores)
    print("  5 highest-scoring test clips:")
    for i in order[:5]:
        label = 'anomaly' if test_labels[i] else 'normal'
        path = fold['test'].iloc[i]['path']
        print(f"    score={scores[i]:.3f}  label={label:8s}  {path}")


if __name__ == "__main__":
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cases = [1, 2, 3, 4]
    checkpoints = ['4f3ffab6e66e.pt', 'ec6602b5742c.pt', '554da113910d.pt', '8ffeae10cff6.pt']

    for case in cases:
        run(case, checkpoints[case - 1], cfg, device)