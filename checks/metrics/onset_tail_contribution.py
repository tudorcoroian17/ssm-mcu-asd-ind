"""
If scoring on the onset/tail silence bracket ALONE still gets a high AUC,
the model (or the distance head) is reading clip timing, not the engine.
Onset ~0.816s +/- 0.072s, tail ~0.45s -- padded generously below since the
exact bracket length varies per clip.
"""
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from checks.metrics.eval_auc_pauc import get_embeddings, score_euclidean

ONSET_FRAMES = 40  # ~1.28s, generous padding over the ~0.816s +/- 0.072s bracket
TAIL_FRAMES = 25   # ~0.8s, generous padding over the ~0.45s bracket


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

    variants = {
        'full_clip':   (X_train, X_test),
        'onset_only':  (X_train[:, :ONSET_FRAMES, :], X_test[:, :ONSET_FRAMES, :]),
        'tail_only':   (X_train[:, -TAIL_FRAMES:, :], X_test[:, -TAIL_FRAMES:, :]),
        'middle_only': (X_train[:, ONSET_FRAMES:-TAIL_FRAMES, :], X_test[:, ONSET_FRAMES:-TAIL_FRAMES, :]),
    }

    print(f"=== case{held_out_case}: protocol-memorization check ===")
    for name, (Xtr, Xte) in variants.items():
        train_emb = get_embeddings(model, Xtr, device)
        test_emb = get_embeddings(model, Xte, device)
        scores, _ = score_euclidean(train_emb, test_emb, cfg['seed'])
        auc = roc_auc_score(test_labels, scores)
        print(f"  {name:12s} T={Xtr.shape[1]:4d}  AUC={auc:.4f}")


if __name__ == "__main__":
    config = load_config()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    cases = [1, 2, 3, 4]
    checkpoints = ['4f3ffab6e66e.pt', 'ec6602b5742c.pt', '554da113910d.pt', '8ffeae10cff6.pt']

    for case in cases:
        print(f'\n=== held_out_case {case} ===')
        run(case, checkpoints[case - 1], config, dev)