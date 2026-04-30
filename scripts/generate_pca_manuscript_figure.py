"""
Generate publication-quality PCA 2D visualization for manuscript submission.

Creates a figure showing:
- Panel A: Public discourse reference map (891 chunks) colored by 22 clusters/macro-themes
- Panel B: Same map with artist probes overlaid, colored by concern theme
- Panel C: Same map with public probes overlaid (if available)

Uses PCA on the 8D consensus UMAP coordinates (PC1: 38.1%, PC2: 30.2%, total: 68.2%).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus_umap import (
    distance_matrix_consensus,
    run_umap_multi_seed,
    umap_from_precomputed_distances,
)
from src.clustering import run_kmeans
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
N_COMPONENTS = 8
N_NEIGHBORS = 27
MIN_DIST = 0.1
OUTPUT_DIR = Path(__file__).parent.parent / "figures"
EDIT_DIR = Path(__file__).parent.parent / "When-Algorithms-Meet-Artists-EDIT"

# Macro-theme assignments (from Table S2)
MACRO_THEMES = {
    "Institutions & Markets": [1, 2, 4, 11, 13, 14, 19],
    "Governance & Rights": [8, 9, 17, 18, 20],
    "Technical Genealogy": [0, 3, 7, 12],
    "Practice & Pedagogy": [15, 16, 21],
    "Philosophy of Creativity": [5, 6],
}

MACRO_COLORS = {
    "Institutions & Markets": "#4C72B0",
    "Governance & Rights": "#DD8452",
    "Technical Genealogy": "#55A868",
    "Practice & Pedagogy": "#C44E52",
    "Philosophy of Creativity": "#8172B3",
}

THEME_COLORS = {
    "threat": "#e74c3c",
    "utility": "#2ecc71",
    "ownership": "#3498db",
    "transparency": "#9b59b6",
    "compensation": "#f39c12",
}

# ============================================================
# Load and process
# ============================================================
print("Loading data...")
X_public = np.load(EDIT_DIR / "embeddings" / "Chunks_of_Public_250words_with_25word_overlap_e5_embeddings.npy")
X_artist = np.load(EDIT_DIR / "embeddings" / "Artists_Perspectives_e5_embeddings.npy")
df_public = load_public_discourse(Path(__file__).parent.parent / "data")
df_artist = load_artist_perspectives(Path(__file__).parent.parent / "data")

print("Running consensus UMAP...")
umap_embeddings = run_umap_multi_seed(
    X_public, seeds=SEEDS, n_components=N_COMPONENTS,
    n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric="cosine",
)
D_consensus = distance_matrix_consensus(umap_embeddings, metric="euclidean")
consensus_8d, _ = umap_from_precomputed_distances(
    D_consensus, n_components=N_COMPONENTS,
    n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
)

print("Clustering...")
labels_km, _ = run_kmeans(consensus_8d, n_clusters=28, metric="euclidean")

print("Training projection head...")
proj = train_projection_head(X_public, consensus_8d, random_state=42)
artist_8d = project_to_consensus_space(X_artist, proj['model'], proj['scaler_X'], proj['scaler_Y'])

print("PCA 8D → 2D...")
X_combined = np.vstack([consensus_8d, artist_8d])
pca = PCA(n_components=2)
X_pca_all = pca.fit_transform(X_combined)

X_public_2d = X_pca_all[:len(consensus_8d)]
X_artist_2d = X_pca_all[len(consensus_8d):]

# Assign macro-themes
cluster_to_macro = {}
for macro, clusters in MACRO_THEMES.items():
    for c in clusters:
        cluster_to_macro[c] = macro

macro_labels = [cluster_to_macro.get(int(l), "Unknown") for l in labels_km]
theme_labels = df_artist["question_group"].str.strip().str.lower().values

# Assign artist probes to nearest cluster
nn = NearestNeighbors(n_neighbors=1).fit(consensus_8d)
_, idx = nn.kneighbors(artist_8d)
artist_cluster_labels = labels_km[idx.flatten()]

# ============================================================
# Figure: 2-panel manuscript figure
# ============================================================
print("Generating manuscript figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel A: Reference map colored by macro-theme ---
ax = axes[0]
for macro, color in MACRO_COLORS.items():
    mask = np.array([m == macro for m in macro_labels])
    if mask.sum() > 0:
        ax.scatter(
            X_public_2d[mask, 0], X_public_2d[mask, 1],
            c=color, alpha=0.6, s=20, edgecolors="none", zorder=2,
        )

handles = [mpatches.Patch(color=c, label=f"{m} ({sum(1 for x in macro_labels if x==m)})")
           for m, c in MACRO_COLORS.items()]
ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
ax.set_title(f"A. Public Discourse Semantic Map (n={len(consensus_8d)})", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)

# --- Panel B: Reference map (gray) + artist probes by theme ---
ax = axes[1]
ax.scatter(
    X_public_2d[:, 0], X_public_2d[:, 1],
    c="lightgray", alpha=0.25, s=8, edgecolors="none", zorder=1,
    label=f"Public discourse (n={len(consensus_8d)})",
)

for theme, color in THEME_COLORS.items():
    mask = theme_labels == theme
    if mask.sum() > 0:
        ax.scatter(
            X_artist_2d[mask, 0], X_artist_2d[mask, 1],
            c=color, alpha=0.7, s=25, edgecolors="white", linewidths=0.3, zorder=3,
            label=f"{theme.capitalize()} (n={mask.sum()})",
        )

ax.legend(fontsize=8, loc="lower left", framealpha=0.9, markerscale=1.5)
ax.set_title(f"B. Artist Probes Projected into Discourse Space (n={len(artist_8d)})", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)

plt.suptitle(
    "Semantic Compression: Artist Concerns in Public AI-Art Discourse",
    fontsize=15, fontweight="bold", y=1.02,
)
plt.tight_layout()

save_path = OUTPUT_DIR / "figure_manuscript_pca_2d.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path}")

save_path_pdf = OUTPUT_DIR / "figure_manuscript_pca_2d.pdf"
fig.savefig(save_path_pdf, bbox_inches="tight")
print(f"Saved: {save_path_pdf}")
plt.close()

# ============================================================
# Print summary stats for the figure caption
# ============================================================
print("\n" + "=" * 60)
print("FIGURE CAPTION DATA")
print("=" * 60)
print(f"PCA variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%, total={sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"Public discourse chunks: {len(consensus_8d)}")
print(f"Artist probes: {len(artist_8d)}")
print(f"Projection head R² val: {proj['r2_val']:.4f}")
print(f"\nArtist probe cluster distribution:")
for c in sorted(set(artist_cluster_labels)):
    count = (artist_cluster_labels == c).sum()
    pct = 100 * count / len(artist_cluster_labels)
    macro = cluster_to_macro.get(int(c), "?")
    if count > 10:
        print(f"  Cluster {c:2d} ({macro[:20]:20s}): {count:4d} probes ({pct:.1f}%)")

# Top 4 clusters
top4 = sorted(set(artist_cluster_labels), key=lambda c: -(artist_cluster_labels == c).sum())[:4]
top4_total = sum((artist_cluster_labels == c).sum() for c in top4)
print(f"\nTop 4 clusters capture: {top4_total}/{len(artist_cluster_labels)} = {100*top4_total/len(artist_cluster_labels):.1f}%")

# Clusters with zero artist probes
zero_clusters = [c for c in range(22) if (artist_cluster_labels == c).sum() == 0]
zero_mass = sum((labels_km == c).sum() for c in zero_clusters) / len(labels_km)
print(f"Clusters with 0 artist probes: {len(zero_clusters)} ({zero_mass*100:.1f}% of public discourse)")

print("\nDone.")
