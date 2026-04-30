"""
3-Way PCA 2D Comparison: Public Discourse + Artist Probes + Public Probes

Generates comprehensive visualizations and statistical comparisons between
all three datasets in the PCA 2D projection of the consensus semantic space.

Public probes: The 250 Likert anchor phrases are embedded and projected into
the consensus space as the style-matched control. These are the stimuli used
to extract the 379 NN-based public probes described in the manuscript.

All runs use the full 31 seeds from the manuscript pipeline.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus_umap import (
    distance_matrix_consensus,
    run_umap_multi_seed,
    umap_from_precomputed_distances,
)
from src.clustering import run_kmeans, get_cluster_sizes
from src.projection import train_projection_head, project_to_consensus_space
from src.analysis import (
    safe_probabilities, js_divergence, kl_divergence, topic_counts,
    centroid_distance, cramers_v, cohens_d,
)
from src.data_loading import load_public_discourse, load_artist_perspectives, load_likert_anchors
from src.embeddings import embed_chunks

# ============================================================
# Config — full 31 seeds, identical to manuscript
# ============================================================
SEEDS = [
    137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
    69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
    46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
]
N_COMPONENTS = 8
N_NEIGHBORS = 27
MIN_DIST = 0.1
OUTPUT_DIR = Path(__file__).parent.parent / "figures" / "three_way_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDIT_DIR = Path(__file__).parent.parent / "When-Algorithms-Meet-Artists-EDIT"
DATA_DIR = Path(__file__).parent.parent / "data"

THEME_COLORS = {
    "threat": "#e74c3c",
    "utility": "#2ecc71",
    "ownership": "#3498db",
    "transparency": "#9b59b6",
    "compensation": "#f39c12",
}

# ============================================================
# 1. Load data
# ============================================================
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

X_public = np.load(EDIT_DIR / "embeddings" / "Chunks_of_Public_250words_with_25word_overlap_e5_embeddings.npy")
X_artist = np.load(EDIT_DIR / "embeddings" / "Artists_Perspectives_e5_embeddings.npy")

df_public = load_public_discourse(DATA_DIR)
df_artist = load_artist_perspectives(DATA_DIR)
df_likert = load_likert_anchors(DATA_DIR)

print(f"Public discourse: {X_public.shape}")
print(f"Artist probes: {X_artist.shape}")
print(f"Likert anchors (public probes): {len(df_likert)} rows")

# ============================================================
# 2. Embed Likert anchor phrases (public probes)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Embedding Likert anchor phrases")
print("=" * 70)

t0 = time.time()
X_likert = embed_chunks(df_likert, text_col="text", model_name="intfloat/e5-large-v2", batch_size=32)
print(f"Likert embeddings: {X_likert.shape} ({time.time()-t0:.1f}s)")

# ============================================================
# 3. Consensus UMAP → 8D (31 seeds)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 3: Consensus UMAP ({len(SEEDS)} seeds → {N_COMPONENTS}D)")
print("=" * 70)

t0 = time.time()
umap_embeddings = run_umap_multi_seed(
    X_public, seeds=SEEDS, n_components=N_COMPONENTS,
    n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric="cosine",
)
D_consensus = distance_matrix_consensus(umap_embeddings, metric="euclidean")
consensus_8d, _ = umap_from_precomputed_distances(
    D_consensus, n_components=N_COMPONENTS,
    n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
)
print(f"Consensus UMAP: {time.time()-t0:.1f}s, shape: {consensus_8d.shape}")

# ============================================================
# 4. Cluster + Project
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Clustering and projection")
print("=" * 70)

labels_km, _ = run_kmeans(consensus_8d, n_clusters=22, metric="euclidean")

proj = train_projection_head(X_public, consensus_8d, random_state=42)
print(f"Projection head R² train: {proj['r2_train']:.4f} | val: {proj['r2_val']:.4f}")

artist_8d = project_to_consensus_space(X_artist, proj['model'], proj['scaler_X'], proj['scaler_Y'])
likert_8d = project_to_consensus_space(X_likert, proj['model'], proj['scaler_X'], proj['scaler_Y'])

print(f"Public 8D: {consensus_8d.shape}")
print(f"Artist 8D: {artist_8d.shape}")
print(f"Public probes 8D: {likert_8d.shape}")

# ============================================================
# 5. PCA 8D → 2D (all three datasets combined)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: PCA 8D → 2D (combined)")
print("=" * 70)

n_pub = len(consensus_8d)
n_art = len(artist_8d)
n_lik = len(likert_8d)

X_all = np.vstack([consensus_8d, artist_8d, likert_8d])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)

X_pub_2d = X_pca[:n_pub]
X_art_2d = X_pca[n_pub:n_pub+n_art]
X_lik_2d = X_pca[n_pub+n_art:]

source = np.array(["public"]*n_pub + ["artist"]*n_art + ["public_probe"]*n_lik)

print(f"PCA variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%, total={sum(pca.explained_variance_ratio_)*100:.1f}%")

# ============================================================
# 6. Assign cluster labels to probes via NN
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Nearest-neighbor cluster assignment")
print("=" * 70)

nn = NearestNeighbors(n_neighbors=1).fit(consensus_8d)
_, idx_art = nn.kneighbors(artist_8d)
_, idx_lik = nn.kneighbors(likert_8d)
artist_cluster_labels = labels_km[idx_art.flatten()]
likert_cluster_labels = labels_km[idx_lik.flatten()]

print(f"Artist probes assigned to {len(set(artist_cluster_labels))} unique clusters")
print(f"Public probes assigned to {len(set(likert_cluster_labels))} unique clusters")

# ============================================================
# 7. Comprehensive 3-way statistical comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: 3-Way Statistical Comparison")
print("=" * 70)

all_cluster_labels = sorted(set(labels_km))

pub_counts = topic_counts(pd.DataFrame({"c": labels_km}), "c", all_cluster_labels)
art_counts = topic_counts(pd.DataFrame({"c": artist_cluster_labels}), "c", all_cluster_labels)
lik_counts = topic_counts(pd.DataFrame({"c": likert_cluster_labels}), "c", all_cluster_labels)

pub_frac = safe_probabilities(pub_counts)
art_frac = safe_probabilities(art_counts)
lik_frac = safe_probabilities(lik_counts)

# Pairwise comparisons
pairs = [
    ("Public vs Artist", pub_frac, art_frac, pub_counts, art_counts,
     consensus_8d, artist_8d, X_pub_2d, X_art_2d),
    ("Public vs Public Probes", pub_frac, lik_frac, pub_counts, lik_counts,
     consensus_8d, likert_8d, X_pub_2d, X_lik_2d),
    ("Artist vs Public Probes", art_frac, lik_frac, art_counts, lik_counts,
     artist_8d, likert_8d, X_art_2d, X_lik_2d),
]

comparison_rows = []
for name, frac_a, frac_b, cnt_a, cnt_b, X_a_8d, X_b_8d, X_a_2d, X_b_2d in pairs:
    # JSD
    jsd = js_divergence(cnt_a, cnt_b)

    # Cramér's V from contingency table (filter out zero-sum columns)
    ct = np.vstack([cnt_a, cnt_b])
    ct_nonzero = ct[:, ct.sum(axis=0) > 0]  # drop clusters absent from both
    try:
        cv = cramers_v(ct_nonzero)
        chi2, p_chi2, dof, _ = stats.chi2_contingency(ct_nonzero)
    except ValueError:
        cv = float("nan")
        chi2, p_chi2, dof = float("nan"), float("nan"), 0

    # Centroid distance (8D)
    centroid_a = X_a_8d.mean(axis=0)
    centroid_b = X_b_8d.mean(axis=0)
    cd_8d = float(np.linalg.norm(centroid_a - centroid_b))

    # Centroid distance (2D)
    cd_2d = float(np.linalg.norm(X_a_2d.mean(axis=0) - X_b_2d.mean(axis=0)))

    # k-NN same-source rate
    X_combined = np.vstack([X_a_8d, X_b_8d])
    labels_src = np.array(["a"]*len(X_a_8d) + ["b"]*len(X_b_8d))
    nn_k = NearestNeighbors(n_neighbors=16).fit(X_combined)
    _, idx_nn = nn_k.kneighbors(X_combined)
    same_src = sum(np.sum(labels_src[idx_nn[i, 1:]] == labels_src[i]) for i in range(len(X_combined)))
    ssr = same_src / (len(X_combined) * 15)

    # Per-dimension Cohen's d (8D)
    d_values = [cohens_d(X_a_8d[:, dim], X_b_8d[:, dim]) for dim in range(X_a_8d.shape[1])]
    mean_abs_d = float(np.mean(np.abs(d_values)))

    row = {
        "Comparison": name,
        "JSD": jsd,
        "Cramér's V": cv,
        "Chi² p-value": p_chi2,
        "Centroid Dist (8D)": cd_8d,
        "Centroid Dist (2D)": cd_2d,
        "Same-Source kNN (k=15)": ssr,
        "Mean |Cohen's d| (8D)": mean_abs_d,
    }
    comparison_rows.append(row)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in row.items():
        if k == "Comparison":
            continue
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        else:
            print(f"  {k:30s}: {v}")

df_stats = pd.DataFrame(comparison_rows)
df_stats.to_csv(OUTPUT_DIR / "three_way_statistics.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'three_way_statistics.csv'}")

# ============================================================
# 8. Cluster concentration analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Cluster concentration analysis")
print("=" * 70)

for label, counts, name in [
    ("Artist probes", artist_cluster_labels, "artist"),
    ("Public probes", likert_cluster_labels, "public_probe"),
]:
    n_total = len(counts)
    cluster_dist = pd.Series(counts).value_counts().sort_index()

    top4 = cluster_dist.nlargest(4)
    top4_pct = 100 * top4.sum() / n_total
    zero_clusters = [c for c in all_cluster_labels if (counts == c).sum() == 0]
    zero_mass = 100 * sum((labels_km == c).sum() for c in zero_clusters) / len(labels_km)

    print(f"\n{label} (n={n_total}):")
    print(f"  Top 4 clusters capture: {top4.sum()}/{n_total} ({top4_pct:.1f}%)")
    print(f"  Clusters with 0 probes: {len(zero_clusters)} ({zero_mass:.1f}% of public discourse)")
    print(f"  Unique clusters used: {len(cluster_dist)}")

# Theme-specific salience comparison
print("\n--- Theme-Specific Salience Ratios ---")
artist_themes = df_artist["question_group"].str.strip().str.lower().values
likert_themes = df_likert["theme"].str.strip().str.lower().values

for theme in sorted(THEME_COLORS.keys()):
    art_mask = artist_themes == theme
    lik_mask = likert_themes == theme

    art_theme_clusters = artist_cluster_labels[art_mask]
    lik_theme_clusters = likert_cluster_labels[lik_mask]

    art_theme_counts = topic_counts(pd.DataFrame({"c": art_theme_clusters}), "c", all_cluster_labels)
    lik_theme_counts = topic_counts(pd.DataFrame({"c": lik_theme_clusters}), "c", all_cluster_labels)

    art_sr_vals = safe_probabilities(art_theme_counts) / safe_probabilities(pub_counts)
    lik_sr_vals = safe_probabilities(lik_theme_counts) / safe_probabilities(pub_counts)

    art_max = float(art_sr_vals.max())
    lik_max = float(lik_sr_vals.max())

    print(f"  {theme:15s}: Artist max SR = {art_max:6.2f}x | Public probe max SR = {lik_max:6.2f}x")

# ============================================================
# 9. Visualizations
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Generating visualizations")
print("=" * 70)

# --- Figure 1: 3-way source overlay ---
fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(X_pub_2d[:, 0], X_pub_2d[:, 1], c="lightgray", alpha=0.3, s=10,
           label=f"Public discourse (n={n_pub})", edgecolors="none", zorder=1)
ax.scatter(X_lik_2d[:, 0], X_lik_2d[:, 1], c="#2ca02c", alpha=0.6, s=30, marker="s",
           label=f"Public probes (n={n_lik})", edgecolors="white", linewidths=0.3, zorder=2)
ax.scatter(X_art_2d[:, 0], X_art_2d[:, 1], c="#d62728", alpha=0.4, s=15,
           label=f"Artist probes (n={n_art})", edgecolors="none", zorder=3)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
ax.set_title("3-Way Comparison: Public Discourse + Artist Probes + Public Probes", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, markerscale=2, loc="lower left")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "three_way_source_overlay.png", dpi=300, bbox_inches="tight")
print(f"Saved: three_way_source_overlay.png")
plt.close()

# --- Figure 2: 3-panel by source, colored by theme ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# Panel A: Public discourse (gray + clusters)
ax = axes[0]
ax.scatter(X_pub_2d[:, 0], X_pub_2d[:, 1], c="steelblue", alpha=0.4, s=12, edgecolors="none")
ax.set_title(f"A. Public Discourse (n={n_pub})", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

# Panel B: Artist probes by theme
ax = axes[1]
ax.scatter(X_pub_2d[:, 0], X_pub_2d[:, 1], c="lightgray", alpha=0.15, s=5, edgecolors="none", zorder=1)
for theme, color in THEME_COLORS.items():
    mask = artist_themes == theme
    ax.scatter(X_art_2d[mask, 0], X_art_2d[mask, 1], c=color, alpha=0.6, s=20,
               label=f"{theme.capitalize()} ({mask.sum()})", edgecolors="white", linewidths=0.2, zorder=3)
ax.set_title(f"B. Artist Probes by Theme (n={n_art})", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.legend(fontsize=9, markerscale=2, loc="lower left")

# Panel C: Public probes by theme
ax = axes[2]
ax.scatter(X_pub_2d[:, 0], X_pub_2d[:, 1], c="lightgray", alpha=0.15, s=5, edgecolors="none", zorder=1)
for theme, color in THEME_COLORS.items():
    mask = likert_themes == theme
    ax.scatter(X_lik_2d[mask, 0], X_lik_2d[mask, 1], c=color, alpha=0.7, s=35, marker="s",
               label=f"{theme.capitalize()} ({mask.sum()})", edgecolors="white", linewidths=0.3, zorder=3)
ax.set_title(f"C. Public Probes by Theme (n={n_lik})", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.legend(fontsize=9, markerscale=1.5, loc="lower left")

plt.suptitle("Artist Probes vs Public Probes: Spatial Distribution by Concern Theme",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "three_way_by_theme.png", dpi=300, bbox_inches="tight")
print(f"Saved: three_way_by_theme.png")
plt.close()

# --- Figure 3: Manuscript-ready 2-panel (reference map + all probes overlaid) ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Panel A: Reference map
ax = axes[0]
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
cluster_to_macro = {}
for macro, clusters in MACRO_THEMES.items():
    for c in clusters:
        cluster_to_macro[c] = macro
macro_labels = [cluster_to_macro.get(int(l), "Unknown") for l in labels_km]

for macro, color in MACRO_COLORS.items():
    mask = np.array([m == macro for m in macro_labels])
    if mask.sum() > 0:
        ax.scatter(X_pub_2d[mask, 0], X_pub_2d[mask, 1], c=color, alpha=0.6, s=20, edgecolors="none")
handles = [mpatches.Patch(color=c, label=m) for m, c in MACRO_COLORS.items()]
ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
ax.set_title(f"A. Public Discourse Semantic Map (n={n_pub})", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)

# Panel B: Gray background + artist probes + public probes
ax = axes[1]
ax.scatter(X_pub_2d[:, 0], X_pub_2d[:, 1], c="lightgray", alpha=0.2, s=6,
           edgecolors="none", zorder=1, label=f"Public discourse (n={n_pub})")
for theme, color in THEME_COLORS.items():
    art_mask = artist_themes == theme
    ax.scatter(X_art_2d[art_mask, 0], X_art_2d[art_mask, 1], c=color, alpha=0.5, s=18,
               edgecolors="white", linewidths=0.2, zorder=3,
               label=f"Artist: {theme.capitalize()} ({art_mask.sum()})")
for theme, color in THEME_COLORS.items():
    lik_mask = likert_themes == theme
    ax.scatter(X_lik_2d[lik_mask, 0], X_lik_2d[lik_mask, 1], c=color, alpha=0.8, s=40,
               marker="s", edgecolors="black", linewidths=0.4, zorder=4)

# Custom legend
legend_elements = [
    mpatches.Patch(facecolor="lightgray", label=f"Public discourse (n={n_pub})"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label=f"Artist probes (n={n_art})"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label=f"Public probes (n={n_lik})"),
]
for theme, color in THEME_COLORS.items():
    legend_elements.append(mpatches.Patch(color=color, label=theme.capitalize()))
ax.legend(handles=legend_elements, fontsize=7.5, loc="lower left", framealpha=0.9)
ax.set_title(f"B. All Probes in Discourse Space", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)

plt.suptitle("Semantic Compression: 3-Way Comparison in PCA Space", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "manuscript_three_way.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "manuscript_three_way.pdf", bbox_inches="tight")
print(f"Saved: manuscript_three_way.png + .pdf")
plt.close()

# ============================================================
# 10. Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nPCA: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%, total={sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"Projection head R² val: {proj['r2_val']:.4f}")
print(f"\n{'Comparison':30s} {'JSD':>8s} {'Cramér V':>10s} {'Centroid 8D':>13s} {'Same-src kNN':>14s}")
print("-" * 80)
for _, row in df_stats.iterrows():
    cv_val = row["Cramér's V"]
    print(f"{row['Comparison']:30s} {row['JSD']:8.4f} {cv_val:10.4f} {row['Centroid Dist (8D)']:13.4f} {row['Same-Source kNN (k=15)']:14.4f}")

print("\n" + "=" * 70)
print("DONE — all outputs in figures/three_way_comparison/")
print("=" * 70)
