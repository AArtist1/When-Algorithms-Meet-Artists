"""
PCA vs UMAP for 2D Visualization of the Consensus Semantic Space

This script compares two approaches for reducing the 8D consensus UMAP
coordinates to 2D for visualization:

    Approach A (current): UMAP 8D → 2D
    Approach B (proposed): PCA 8D → 2D (first 2 principal components)

Pipeline:
1. Load precomputed 1024-dim embeddings (e5-large-v2)
2. Run consensus UMAP (11 seeds) to get stable 8D coordinates
3. Train MLP projection head → project artist probes into 8D space
4. Apply both PCA and UMAP to go from 8D → 2D
5. Generate comparative visualizations and metrics
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus_umap import (
    distance_matrix_consensus,
    l2_normalize,
    run_umap_multi_seed,
    umap_from_precomputed_distances,
)
from src.clustering import run_hdbscan, run_kmeans, get_cluster_sizes
from src.projection import train_projection_head, project_to_consensus_space
from src.data_loading import load_public_discourse, load_artist_perspectives

# ============================================================
# Config
# ============================================================
SEEDS = [  # Full 31 seeds — identical to manuscript pipeline (src/models.py UMAPConfig)
    137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
    69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
    46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
]
N_COMPONENTS_CONSENSUS = 8
N_NEIGHBORS = 27
MIN_DIST = 0.1
OUTPUT_DIR = Path(__file__).parent.parent / "figures" / "pca_vs_umap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load precomputed embeddings
# ============================================================
print("=" * 60)
print("STEP 1: Loading precomputed embeddings")
print("=" * 60)

EDIT_DIR = Path(__file__).parent.parent / "When-Algorithms-Meet-Artists-EDIT"
X_public = np.load(EDIT_DIR / "embeddings" / "Chunks_of_Public_250words_with_25word_overlap_e5_embeddings.npy")
X_artist = np.load(EDIT_DIR / "embeddings" / "Artists_Perspectives_e5_embeddings.npy")

df_public = load_public_discourse(Path(__file__).parent.parent / "data")
df_artist = load_artist_perspectives(Path(__file__).parent.parent / "data")

print(f"Public embeddings: {X_public.shape}")
print(f"Artist embeddings: {X_artist.shape}")

# ============================================================
# 2. Consensus UMAP → 8D reference map
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 2: Consensus UMAP ({len(SEEDS)} seeds → {N_COMPONENTS_CONSENSUS}D)")
print("=" * 60)

t0 = time.time()
umap_embeddings = run_umap_multi_seed(
    X_public, seeds=SEEDS,
    n_components=N_COMPONENTS_CONSENSUS,
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    metric="cosine",
)
print(f"Multi-seed UMAP: {time.time() - t0:.1f}s")

t0 = time.time()
D_consensus = distance_matrix_consensus(umap_embeddings, metric="euclidean")
consensus_8d, _ = umap_from_precomputed_distances(
    D_consensus, n_components=N_COMPONENTS_CONSENSUS,
    n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
)
print(f"Distance consensus + final UMAP: {time.time() - t0:.1f}s")
print(f"Consensus 8D shape: {consensus_8d.shape}")

# ============================================================
# 3. Cluster in 8D space
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Clustering in 8D consensus space")
print("=" * 60)

labels_hdbscan, _ = run_hdbscan(consensus_8d, min_cluster_size=10, min_samples=5)
sizes = get_cluster_sizes(labels_hdbscan)
n_clusters = len([k for k in sizes if k >= 0])
n_noise = sizes.get(-1, 0)
print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(labels_hdbscan):.1f}%)")

# Also try KMeans with 22 clusters (manuscript value) for consistent comparison
labels_km, _ = run_kmeans(consensus_8d, n_clusters=22, metric="euclidean")

# ============================================================
# 4. Project artist probes into 8D consensus space
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Projecting artist probes via MLP")
print("=" * 60)

proj = train_projection_head(X_public, consensus_8d, random_state=42)
print(f"Projection head R² train: {proj['r2_train']:.4f} | val: {proj['r2_val']:.4f}")

artist_8d = project_to_consensus_space(X_artist, proj['model'], proj['scaler_X'], proj['scaler_Y'])
print(f"Artist probes projected to 8D: {artist_8d.shape}")

# Combined data for 2D reduction
X_combined_8d = np.vstack([consensus_8d, artist_8d])
source_labels = np.array(["public"] * len(consensus_8d) + ["artist"] * len(artist_8d))
print(f"Combined 8D data: {X_combined_8d.shape}")

# ============================================================
# 5. Approach A: UMAP 8D → 2D
# ============================================================
print("\n" + "=" * 60)
print("STEP 5A: UMAP 8D → 2D")
print("=" * 60)

from umap import UMAP

t0 = time.time()
reducer_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
X_umap_2d = reducer_2d.fit_transform(X_combined_8d)
print(f"UMAP 2D: {time.time() - t0:.1f}s, shape: {X_umap_2d.shape}")

# ============================================================
# 6. Approach B: PCA 8D → 2D
# ============================================================
print("\n" + "=" * 60)
print("STEP 5B: PCA 8D → 2D")
print("=" * 60)

pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_combined_8d)
print(f"PCA 2D shape: {X_pca_2d.shape}")
print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.1f}%)")
print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.1f}%)")
print(f"Total variance (PC1+PC2): {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")

# Full PCA for cumulative variance
pca_full = PCA(n_components=min(8, X_combined_8d.shape[1]))
pca_full.fit(X_combined_8d)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print(f"\nCumulative variance by PC:")
for i, (var, cum) in enumerate(zip(pca_full.explained_variance_ratio_, cumvar)):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)  cumulative: {cum:.4f} ({cum*100:.1f}%)")

# ============================================================
# 7. Compute comparison metrics
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Computing comparison metrics")
print("=" * 60)

n_public = len(consensus_8d)
n_artist = len(artist_8d)

# -- Trustworthiness (how well local structure is preserved from 8D to 2D) --
tw_umap = trustworthiness(X_combined_8d, X_umap_2d, n_neighbors=15)
tw_pca = trustworthiness(X_combined_8d, X_pca_2d, n_neighbors=15)
print(f"Trustworthiness (k=15): UMAP={tw_umap:.4f}, PCA={tw_pca:.4f}")

# -- k-NN preservation (do points keep the same neighbors in 2D?) --
def knn_preservation(X_high, X_low, k=15):
    nn_h = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    nn_l = NearestNeighbors(n_neighbors=k+1).fit(X_low)
    _, idx_h = nn_h.kneighbors(X_high)
    _, idx_l = nn_l.kneighbors(X_low)
    idx_h, idx_l = idx_h[:, 1:], idx_l[:, 1:]
    overlaps = sum(len(set(idx_h[i]) & set(idx_l[i])) for i in range(len(X_high)))
    return overlaps / (len(X_high) * k)

knn_umap = knn_preservation(X_combined_8d, X_umap_2d, k=15)
knn_pca = knn_preservation(X_combined_8d, X_pca_2d, k=15)
print(f"k-NN preservation (k=15): UMAP={knn_umap:.4f}, PCA={knn_pca:.4f}")

# -- Cluster separation in 2D (silhouette using KMeans labels from 8D) --
combined_km_labels = np.concatenate([labels_km, np.full(n_artist, -1)])
public_mask = source_labels == "public"

sil_umap = silhouette_score(X_umap_2d[public_mask], labels_km, metric="euclidean")
sil_pca = silhouette_score(X_pca_2d[public_mask], labels_km, metric="euclidean")
print(f"Silhouette (public, KMeans-22 labels): UMAP={sil_umap:.4f}, PCA={sil_pca:.4f}")

# -- Source separation (can you visually distinguish public vs artist?) --
# Centroid distance between public and artist in 2D
pub_centroid_umap = X_umap_2d[:n_public].mean(axis=0)
art_centroid_umap = X_umap_2d[n_public:].mean(axis=0)
dist_umap = np.linalg.norm(pub_centroid_umap - art_centroid_umap)

pub_centroid_pca = X_pca_2d[:n_public].mean(axis=0)
art_centroid_pca = X_pca_2d[n_public:].mean(axis=0)
dist_pca = np.linalg.norm(pub_centroid_pca - art_centroid_pca)
print(f"Public-Artist centroid distance: UMAP={dist_umap:.4f}, PCA={dist_pca:.4f}")

# -- Same-source k-NN rate --
def source_knn_rate(X_2d, sources, k=15):
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_2d)
    _, idx = nn.kneighbors(X_2d)
    same = 0
    for i in range(len(X_2d)):
        neighbors = idx[i, 1:]
        same += np.sum(sources[neighbors] == sources[i])
    return same / (len(X_2d) * k)

ssr_umap = source_knn_rate(X_umap_2d, source_labels, k=15)
ssr_pca = source_knn_rate(X_pca_2d, source_labels, k=15)
print(f"Same-source k-NN rate (k=15): UMAP={ssr_umap:.4f}, PCA={ssr_pca:.4f}")

# ============================================================
# 8. Summary table
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

summary = pd.DataFrame({
    "Metric": [
        "Variance explained (PC1+PC2)",
        "Trustworthiness (k=15)",
        "k-NN preservation (k=15)",
        "Silhouette (22 clusters)",
        "Public-Artist centroid distance",
        "Same-source k-NN rate (k=15)",
    ],
    "UMAP 2D": [
        "N/A",
        f"{tw_umap:.4f}",
        f"{knn_umap:.4f}",
        f"{sil_umap:.4f}",
        f"{dist_umap:.4f}",
        f"{ssr_umap:.4f}",
    ],
    "PCA 2D": [
        f"{sum(pca.explained_variance_ratio_)*100:.1f}%",
        f"{tw_pca:.4f}",
        f"{knn_pca:.4f}",
        f"{sil_pca:.4f}",
        f"{dist_pca:.4f}",
        f"{ssr_pca:.4f}",
    ],
    "Winner": [
        "PCA (interpretable)",
        "UMAP" if tw_umap > tw_pca else "PCA",
        "UMAP" if knn_umap > knn_pca else "PCA",
        "UMAP" if sil_umap > sil_pca else "PCA",
        "UMAP" if dist_umap > dist_pca else "PCA",
        "-",
    ],
})
print(summary.to_string(index=False))
summary.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)

# ============================================================
# 9. Visualizations
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Generating comparative visualizations")
print("=" * 60)

# --- Figure 1: Side-by-side comparison colored by source ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, X_2d, title in [
    (axes[0], X_umap_2d, "UMAP 8D → 2D"),
    (axes[1], X_pca_2d, f"PCA 8D → 2D ({sum(pca.explained_variance_ratio_)*100:.1f}% var.)"),
]:
    pub = source_labels == "public"
    art = source_labels == "artist"
    ax.scatter(X_2d[pub, 0], X_2d[pub, 1], c="#4C72B0", alpha=0.4, s=10, label=f"Public ({pub.sum()})", edgecolors="none")
    ax.scatter(X_2d[art, 0], X_2d[art, 1], c="#DD8452", alpha=0.4, s=10, label=f"Artist ({art.sum()})", edgecolors="none")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=11, markerscale=3)

plt.suptitle("PCA vs UMAP: Public Discourse + Artist Probes in 2D", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "comparison_source_overlay.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'comparison_source_overlay.png'}")
plt.close()

# --- Figure 2: Side-by-side colored by cluster (public only) ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
cmap = plt.cm.tab20(np.linspace(0, 1, 22))

for ax, X_2d, title in [
    (axes[0], X_umap_2d[:n_public], "UMAP 8D → 2D (clusters)"),
    (axes[1], X_pca_2d[:n_public], f"PCA 8D → 2D (clusters)"),
]:
    for ci in range(22):
        mask = labels_km == ci
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap[ci]], alpha=0.6, s=15, label=f"C{ci}", edgecolors="none")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

plt.suptitle("PCA vs UMAP: 22-Cluster Structure in 2D", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "comparison_clusters.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'comparison_clusters.png'}")
plt.close()

# --- Figure 3: Side-by-side colored by artist question_group (artist probes only) ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
theme_colors = {"threat": "#e74c3c", "utility": "#2ecc71", "ownership": "#3498db", "transparency": "#9b59b6", "compensation": "#f39c12"}
theme_labels = df_artist["question_group"].str.strip().str.lower().values

for ax, X_2d, title in [
    (axes[0], X_umap_2d[n_public:], "UMAP: Artist Probes by Theme"),
    (axes[1], X_pca_2d[n_public:], "PCA: Artist Probes by Theme"),
]:
    for theme, color in theme_colors.items():
        mask = theme_labels == theme
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.5, s=12, label=theme.capitalize(), edgecolors="none")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=11, markerscale=3)

plt.suptitle("PCA vs UMAP: Artist Probes Colored by Concern Theme", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "comparison_artist_themes.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'comparison_artist_themes.png'}")
plt.close()

# --- Figure 4: PCA explained variance scree plot ---
fig, ax = plt.subplots(figsize=(8, 5))
pcs = range(1, len(pca_full.explained_variance_ratio_) + 1)
ax.bar(pcs, pca_full.explained_variance_ratio_ * 100, color="#4C72B0", alpha=0.7, label="Individual")
ax.plot(pcs, cumvar * 100, "o-", color="#DD8452", linewidth=2, label="Cumulative")
ax.set_xlabel("Principal Component", fontsize=12)
ax.set_ylabel("Variance Explained (%)", fontsize=12)
ax.set_title("PCA Scree Plot: Variance in 8D Consensus Space", fontsize=14, fontweight="bold")
ax.set_xticks(list(pcs))
ax.set_xticklabels([f"PC{i}" for i in pcs])
ax.legend(fontsize=11)
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "pca_scree_plot.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'pca_scree_plot.png'}")
plt.close()

# --- Figure 5: Combined overlay with artist concentration highlighted ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, X_2d, title in [
    (axes[0], X_umap_2d, "UMAP 8D → 2D"),
    (axes[1], X_pca_2d, f"PCA 8D → 2D"),
]:
    pub = source_labels == "public"
    art = source_labels == "artist"
    # Public as gray background
    ax.scatter(X_2d[pub, 0], X_2d[pub, 1], c="lightgray", alpha=0.3, s=8, label="Public discourse", edgecolors="none", zorder=1)
    # Artist colored by theme
    for theme, color in theme_colors.items():
        mask_t = np.array([False] * n_public + list(theme_labels == theme))
        if mask_t.sum() > 0:
            ax.scatter(X_2d[mask_t, 0], X_2d[mask_t, 1], c=color, alpha=0.6, s=15, label=theme.capitalize(), edgecolors="none", zorder=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=10, markerscale=3, loc="upper right")

plt.suptitle("Artist Probe Concentration: PCA vs UMAP", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "comparison_concentration.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'comparison_concentration.png'}")
plt.close()

print("\n" + "=" * 60)
print("DONE — all outputs saved to figures/pca_vs_umap/")
print("=" * 60)
