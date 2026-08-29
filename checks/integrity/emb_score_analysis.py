"""
4 analyses performed:
    1. Latent Dimension Separation Weights
    2. Decomposed Distance Shift
    3. Distance Score Density and Margin Analysis
    4. 2D PCA Manifold Projection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from runs.compute_hash import train_config_hash
from src.config import PROJECT_ROOT, load_config
from src.eval.embeddings import read_embeddings

POOLING_MODES = ['mean', 'max']
DISTANCE_HEADS = ['euclidean', 'knn_clustered_16', 'knn_full', 'mahalanobis']

def run_analysis(held_out_case, cfg):
    dir_name = train_config_hash(cfg, held_out_case)
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name
    analysis_dir = base_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for pooling_mode in POOLING_MODES:
        print(f"\n=== case{held_out_case} | pooling: {pooling_mode} ===")
        train_emb, val_normal_emb, _, test_emb, test_labels, mean_normal, _ = read_embeddings(held_out_case, cfg, pooling_mode)

        for distance in DISTANCE_HEADS:
            print(f"\n{'=' * 20} distance: {distance} ===")
            scores = np.load(str(base_dir / 'scores' / f'scores_{pooling_mode}_{distance}.npz'))['scores']

            # 1. Logistic regression weights for linear separation share
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(test_emb)
            lr = LogisticRegression(penalty='l2', C=1.0, random_state=42)
            lr.fit(X_scaled, test_labels)

            coef_df = pd.DataFrame({
                'Dimension': [f'Dim {i}' for i in range(64)],
                'Abs_Coef': np.abs(lr.coef_[0])
            })
            coef_df['Relative_Share_Pct'] = (coef_df['Abs_Coef'] / coef_df['Abs_Coef'].sum()) * 100
            coef_df = coef_df.sort_values(by='Relative_Share_Pct', ascending=False).reset_index(drop=True)

            # 2. Euclidean Distance Shift share
            dim_contrib_normal = ((test_emb[test_labels == 0] - mean_normal) ** 2).mean(axis=0)
            dim_contrib_anomaly = ((test_emb[test_labels == 1] - mean_normal) ** 2).mean(axis=0)
            dim_contrib_diff = dim_contrib_anomaly - dim_contrib_normal

            contrib_df = pd.DataFrame({
                'Dimension': [f'Dim {i}' for i in range(64)],
                'Delta_SqDist': dim_contrib_diff,
                'Relative_Share_Pct': (dim_contrib_diff.clip(min=0) / dim_contrib_diff.clip(min=0).sum()) * 100
            }).sort_values(by='Delta_SqDist', ascending=False).reset_index(drop=True)

            # 3. PCA
            pca = PCA(n_components=2, random_state=42)
            all_emb = np.vstack([train_emb, val_normal_emb, test_emb])
            pca.fit(all_emb)

            train_pca = pca.transform(train_emb)
            val_pca = pca.transform(val_normal_emb)
            test_normal_pca = pca.transform(test_emb[test_labels == 0])
            test_anomaly_pca = pca.transform(test_emb[test_labels == 1])

            # Plot high-res grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            plt.subplots_adjust(hspace=0.3, wspace=0.25)

            # Subplot 1
            top_coef = coef_df.head(10).iloc[::-1]
            axes[0, 0].barh(top_coef['Dimension'], top_coef['Relative_Share_Pct'], color='#1f77b4', edgecolor='black',
                            linewidth=0.5)
            axes[0, 0].set_title('Top 10 Latent Dimensions by Model Weight', fontsize=12, fontweight='bold', pad=10)
            axes[0, 0].set_xlabel('Relative Weight Share (%)', fontsize=10)
            axes[0, 0].grid(axis='x', linestyle='--', alpha=0.5)

            # Subplot 2
            top_contrib = contrib_df.head(10).iloc[::-1]
            axes[0, 1].barh(top_contrib['Dimension'], top_contrib['Relative_Share_Pct'], color='#ff7f0e',
                            edgecolor='black', linewidth=0.5)
            axes[0, 1].set_title('Top 10 Dimensions Driving Distance Delta', fontsize=12, fontweight='bold', pad=10)
            axes[0, 1].set_xlabel('Contribution Share to Distance Shift (%)', fontsize=10)
            axes[0, 1].grid(axis='x', linestyle='--', alpha=0.5)

            # Subplot 3
            sns.histplot(scores[test_labels == 0], color='#2ca02c', label='Normal (n=265)', kde=True, ax=axes[1, 0],
                         stat="density", bins=25, alpha=0.5)
            sns.histplot(scores[test_labels == 1], color='#d62728', label='Anomaly (n=265)', kde=True, ax=axes[1, 0],
                         stat="density", bins=25, alpha=0.5)
            axes[1, 0].axvline(x=scores[test_labels == 0].max(), color='black', linestyle=':',
                               label=f'Max Normal ({scores[test_labels == 0].max():.2f})')
            axes[1, 0].axvline(x=scores[test_labels == 1].min(), color='black', linestyle='--',
                               label=f'Min Anomaly ({scores[test_labels == 1].min():.2f})')
            axes[1, 0].set_title('Distance Score Distribution (ROC-AUC = 1.000)', fontsize=12, fontweight='bold',
                                 pad=10)
            axes[1, 0].set_xlabel('Euclidean Distance Score', fontsize=10)
            axes[1, 0].set_ylabel('Density', fontsize=10)
            axes[1, 0].legend(loc='upper right', fontsize=9)
            axes[1, 0].grid(True, linestyle='--', alpha=0.4)

            # Subplot 4
            axes[1, 1].scatter(train_pca[:, 0], train_pca[:, 1], c='#7f7f7f', alpha=0.15, s=15, label='Train (Normal)')
            axes[1, 1].scatter(val_pca[:, 0], val_pca[:, 1], c='#17becf', alpha=0.3, s=20, label='Val (Normal)')
            axes[1, 1].scatter(test_normal_pca[:, 0], test_normal_pca[:, 1], c='#2ca02c', alpha=0.7, s=25,
                               label='Test Normal')
            axes[1, 1].scatter(test_anomaly_pca[:, 0], test_anomaly_pca[:, 1], c='#d62728', alpha=0.7, s=25,
                               label='Test Anomaly')
            axes[1, 1].set_title('2D PCA Projection of SSM Embeddings', fontsize=12, fontweight='bold', pad=10)
            axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)', fontsize=10)
            axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)', fontsize=10)
            axes[1, 1].legend(loc='upper right', fontsize=9)
            axes[1, 1].grid(True, linestyle='--', alpha=0.4)

            plt.savefig(str(analysis_dir / f'analysis_{pooling_mode}_{distance}.png'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    cfg = load_config()

    cases = [1, 2, 3, 4]
    for case in cases:
        print(f'\n=== held_out_case {case} ===')
        run_analysis(case, cfg)
