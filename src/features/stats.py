import numpy as np

def compute_normalization_stats(cache_paths, n_mels):
    n = 0
    mean = np.zeros(n_mels, dtype=np.float64)
    m2 = np.zeros(n_mels, dtype=np.float64)
    for path in cache_paths:
        batch = np.load(path).astype(np.float64)
        n_b = batch.shape[0]
        mean_b = batch.mean(axis=0)
        m2_b = ((batch - mean_b) ** 2).sum(axis=0)
        n_new = n + n_b
        delta = mean_b - mean
        mean = mean + delta * n_b / n_new
        m2 = m2 + m2_b + delta ** 2 * n * n_b / n_new
        n = n_new
    std = np.sqrt(m2 / n)
    return mean.astype(np.float32), std.astype(np.float32), n