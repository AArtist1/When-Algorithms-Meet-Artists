"""Regenerate all manuscript figures with the final pipeline configuration (Config A).

Uses:
    - Config A: k=20, n_neighbors=53, min_dist=0.01, n_components=5
    - 30 unique UMAP seeds (consensus distance-matrix averaging)
    - 4-layer MLP projection head (1024 -> 512 -> 256 -> 128 -> 5)
    - 1,736 public discourse chunks (125 articles)
    - 1,259 artist probes
    - 906 public (keyword) probes

Generates:
    Figure 1: Semantic density (PCA projection with density contours)
    Figure 2: Artist probe concentration (horizontal bar chart)
    Figure 3: Macro-thematic distribution (grouped bar chart)
    Figure 4: Compression metrics (two-panel: entropy + coverage)
    Figure S1: Macro-theme regions (95% convex hulls)
    Figure S2: Artist concern themes (95% convex hulls)

Side effects:
    Writes PNG and PDF files to figures/manuscript/, NMS2026/figures/,
    and NatureCollection2026/figures/. Prints progress to stdout.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
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
from src.data_loading import (
    load_clean_public_discourse,
    load_artist_perspectives,
    load_public_probes,
)

# ---------------------------------------------------------------------------
# Config A parameters
# ---------------------------------------------------------------------------
N_COMPONENTS = 5
N_NEIGHBORS = 53
MIN_DIST = 0.01
K = 20
HIDDEN_LAYERS = (512, 256, 128)

SEEDS = [
    137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
    69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
    46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
]

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "figures" / "manuscript"
NMS_FIG_DIR = ROOT / "jan_2026_manuscript" / "NMS2026" / "figures"
NATURE_FIG_DIR = ROOT / "jan_2026_manuscript" / "NatureCollection2026" / "figures"

# ---------------------------------------------------------------------------
# Macro themes for k=20
# ---------------------------------------------------------------------------
MACRO_THEMES = {
    "Philosophy of Creativity": [1, 4, 11, 13, 17],
    "Practice & Pedagogy": [3, 7, 14, 16],
    "Technical Genealogy": [2, 5, 15, 19],
    "Governance & Rights": [8, 10, 18],
    "Institutions & Markets": [0, 6, 9, 12],
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
THEME_DISPLAY = {
    "threat": "Threat",
    "utility": "Utility",
    "ownership": "Ownership",
    "transparency": "Transparency",
    "compensation": "Compensation",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def density_threshold(H: np.ndarray, pct: float) -> float:
    """Return the density level enclosing *pct*% of the probability mass.

    Args:
        H: 2-D histogram (density grid).
        pct: Percentage of mass to enclose (0-100).

    Returns:
        Density level threshold.
    """
    flat = H.flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return 0.0
    idx = np.argsort(flat)[::-1]
    cs = np.cumsum(flat[idx])
    cs /= cs[-1]
    ci = min(int(np.searchsorted(cs, pct / 100.0)), len(flat) - 1)
    return float(flat[idx[ci]])


def percentile_hull(
    pts: np.ndarray, center: np.ndarray, pct: float
) -> np.ndarray | None:
    """Compute a convex hull around the closest *pct*% of points to *center*.

    Args:
        pts: Array of shape (n, 2).
        center: Centroid, shape (2,).
        pct: Percentage of points to include (0-100).

    Returns:
        Closed polygon vertices (n_vertices+1, 2) or None if too few points.
    """
    dists = np.linalg.norm(pts - center, axis=1)
    k = max(3, int(len(pts) * pct / 100.0))
    k = min(k, len(pts))
    idx = np.argsort(dists)[:k]
    p = pts[idx]
    if len(p) < 3:
        return None
    try:
        hull = ConvexHull(p)
        v = p[hull.vertices]
        return np.vstack([v, v[0]])
    except Exception:
        return None


def density_grid(
    pts: np.ndarray,
    xr: tuple[float, float],
    yr: tuple[float, float],
    bins: int = 80,
    sigma: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a smoothed 2-D density grid.

    Args:
        pts: Array of shape (n, 2).
        xr: X-axis range (min, max).
        yr: Y-axis range (min, max).
        bins: Number of histogram bins per axis.
        sigma: Gaussian smoothing sigma.

    Returns:
        (H, x_centers, y_centers) where H is shape (bins, bins).
    """
    H, xe, ye = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=bins, range=[list(xr), list(yr)]
    )
    H = gaussian_filter(H.T, sigma=sigma)
    return H, 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])


def save_figure(
    fig: matplotlib.figure.Figure, name: str, dirs: list[Path]
) -> None:
    """Save a figure as PNG (300 dpi) and PDF to all target directories.

    Args:
        fig: Matplotlib figure.
        name: Base filename without extension.
        dirs: List of output directories.

    Side effects:
        Writes PNG and PDF files to each directory.
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full figure-generation pipeline.

    Side effects:
        Loads data, trains models, writes figure files, prints progress.
    """
    out_dirs = [FIG_DIR, NMS_FIG_DIR, NATURE_FIG_DIR]

    # ------------------------------------------------------------------
    # 1. Load data and embeddings
    # ------------------------------------------------------------------
    print("Loading data...")
    X_pub = np.load(ROOT / "figures" / "prefix_grid_search" / "prefix_embeddings_public.npy")
    X_art = np.load(ROOT / "figures" / "prefix_grid_search" / "prefix_embeddings_artist.npy")
    X_pp = np.load(ROOT / "figures" / "prefix_grid_search" / "prefix_embeddings_probes.npy")
    df_pub = load_clean_public_discourse(ROOT / "data")
    df_art = load_artist_perspectives(ROOT / "data")
    df_pp = load_public_probes(ROOT / "data")

    print(f"  Public discourse: {X_pub.shape[0]} chunks, {df_pub['article_name'].nunique()} articles")
    print(f"  Artist probes:    {X_art.shape[0]} rows")
    print(f"  Public probes:    {X_pp.shape[0]} rows")

    # ------------------------------------------------------------------
    # 2. Consensus UMAP (30 seeds)
    # ------------------------------------------------------------------
    print(f"Consensus UMAP ({len(SEEDS)} seeds, nc={N_COMPONENTS}, nn={N_NEIGHBORS}, md={MIN_DIST})...")
    umap_embeddings = run_umap_multi_seed(
        X_pub,
        seeds=SEEDS,
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
    )
    print("  Computing distance-matrix consensus...")
    D = distance_matrix_consensus(umap_embeddings, metric="euclidean")
    consensus_coords, _ = umap_from_precomputed_distances(
        D,
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
    )
    print(f"  Consensus coordinates: {consensus_coords.shape}")

    # ------------------------------------------------------------------
    # 3. Clustering
    # ------------------------------------------------------------------
    print(f"KMeans clustering (k={K})...")
    labels_pub, _ = run_kmeans(consensus_coords, n_clusters=K, metric="euclidean")
    print(f"  Cluster sizes: {np.bincount(labels_pub)}")

    # ------------------------------------------------------------------
    # 4. Projection head
    # ------------------------------------------------------------------
    print(f"Training MLP projection head {HIDDEN_LAYERS}...")
    proj = train_projection_head(
        X_pub,
        consensus_coords,
        hidden_layer_sizes=HIDDEN_LAYERS,
        random_state=42,
    )
    print(f"  R2 train={proj['r2_train']:.4f}, R2 val={proj['r2_val']:.4f}")

    art_coords = project_to_consensus_space(
        X_art, proj["model"], proj["scaler_X"], proj["scaler_Y"]
    )
    pp_coords = project_to_consensus_space(
        X_pp, proj["model"], proj["scaler_X"], proj["scaler_Y"]
    )

    # ------------------------------------------------------------------
    # 5. PCA for 2-D visualization (from 5-D consensus space)
    # ------------------------------------------------------------------
    print("PCA (5D -> 2D)...")
    X_all = np.vstack([consensus_coords, art_coords, pp_coords])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    n_pub = len(consensus_coords)
    n_art = len(art_coords)
    pub2d = X_pca[:n_pub]
    art2d = X_pca[n_pub : n_pub + n_art]
    # pp2d available at X_pca[n_pub + n_art :] if needed
    var_pct = pca.explained_variance_ratio_ * 100
    print(f"  Variance explained: PC1={var_pct[0]:.1f}%, PC2={var_pct[1]:.1f}%")

    # ------------------------------------------------------------------
    # 6. Derived labels and assignments
    # ------------------------------------------------------------------
    # Artist theme labels
    theme_labels_art = df_art["question_group"].str.strip().str.lower().values

    # Assign artist probes to nearest public cluster
    nn_model = NearestNeighbors(n_neighbors=1).fit(consensus_coords)
    _, idx_art = nn_model.kneighbors(art_coords)
    art_cluster = labels_pub[idx_art.flatten()]

    # Assign public probes to nearest public cluster
    _, idx_pp = nn_model.kneighbors(pp_coords)
    pp_cluster = labels_pub[idx_pp.flatten()]
    theme_labels_pp = df_pp["theme"].str.strip().str.lower().values

    # ===================================================================
    # FIGURE 1: Semantic Density
    # ===================================================================
    print("Figure 1: Semantic density...")
    x_all = np.concatenate([pub2d[:, 0], art2d[:, 0]])
    y_all = np.concatenate([pub2d[:, 1], art2d[:, 1]])
    pad = 0.8
    xr = (float(x_all.min() - pad), float(x_all.max() + pad))
    yr = (float(y_all.min() - pad), float(y_all.max() + pad))

    H_pub, xe, ye = density_grid(pub2d, xr, yr)
    H_art, _, _ = density_grid(art2d, xr, yr)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7))

    # Light contour rings
    for p in [30, 50, 70, 85, 95]:
        lv = density_threshold(H_pub, p)
        if lv > 0:
            if p == 30:
                ax.contourf(
                    xe, ye, H_pub, levels=[lv, H_pub.max()],
                    colors=["#4C72B0"], alpha=0.04,
                )
            ax.contour(
                xe, ye, H_pub, levels=[lv],
                colors="#4C72B0", alpha=0.22, linewidths=0.5,
            )
    for p in [30, 50, 70, 85, 95]:
        lv = density_threshold(H_art, p)
        if lv > 0:
            if p == 30:
                ax.contourf(
                    xe, ye, H_art, levels=[lv, H_art.max()],
                    colors=["#e74c3c"], alpha=0.06,
                )
            ax.contour(
                xe, ye, H_art, levels=[lv],
                colors="#e74c3c", alpha=0.32, linewidths=0.5,
            )

    # Scatter points
    ax.scatter(
        pub2d[:, 0], pub2d[:, 1],
        c="#4C72B0", alpha=0.06, s=2.5, edgecolors="none", zorder=1,
    )
    ax.scatter(
        art2d[:, 0], art2d[:, 1],
        c="#e74c3c", alpha=0.25, s=5, edgecolors="none", zorder=2,
    )

    # Bold 50% HDR contours
    pub50 = density_threshold(H_pub, 50)
    art50 = density_threshold(H_art, 50)
    if pub50 > 0:
        ax.contour(
            xe, ye, H_pub, levels=[pub50],
            colors="#2c5aa0", linewidths=1.4, linestyles="--", zorder=5,
        )
    if art50 > 0:
        ax.contour(
            xe, ye, H_art, levels=[art50],
            colors="#c0392b", linewidths=2.0, linestyles="-", zorder=6,
        )

    # Annotation for 50% artist HDR
    mask_art = H_art >= art50
    if mask_art.any():
        rows, cols = np.where(mask_art)
        w = H_art[mask_art]
        acx = float(np.average(xe[cols], weights=w))
        acy = float(np.average(ye[rows], weights=w))
        ax.annotate(
            "50% of Artist Concerns",
            xy=(acx + 2.2, acy + 2.0),
            fontsize=9.5,
            color="#c0392b",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#e74c3c",
                alpha=0.92,
                linewidth=0.8,
            ),
            ha="center",
            va="center",
            zorder=8,
        )

    handles = [
        mpatches.Patch(color="#4C72B0", alpha=0.4, label="Public discourse (n=1,736)"),
        mpatches.Patch(color="#e74c3c", alpha=0.6, label="Artist probes (n=1,259)"),
    ]
    ax.legend(
        handles=handles, fontsize=9, loc="lower right",
        framealpha=0.92, edgecolor="#cccccc",
    )
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)", fontsize=11)
    ax.set_title(
        "Semantic Density: Public Discourse vs Artist Concerns",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, "figure_1_semantic_density", out_dirs)
    plt.close()
    print("  Saved figure_1_semantic_density")

    # ===================================================================
    # FIGURE 2: Artist Probe Concentration (horizontal bar chart)
    # ===================================================================
    print("Figure 2: Artist probe concentration...")

    # Count public chunks and artist probes per cluster
    pub_counts = np.bincount(labels_pub, minlength=K)
    art_counts = np.bincount(art_cluster, minlength=K)

    # Identify top-4 clusters where artist probes concentrate
    top4_idx = np.argsort(art_counts)[::-1][:4]
    top4_total = art_counts[top4_idx].sum()
    top4_pct = 100.0 * top4_total / art_counts.sum()

    # Sort clusters by artist probe count (descending)
    sort_order = np.argsort(art_counts)  # ascending for horizontal barh
    cluster_ids_sorted = sort_order

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    y_pos = np.arange(K)

    # Color bars: top-4 clusters get red, others get gray
    bar_colors = []
    for cid in cluster_ids_sorted:
        if cid in top4_idx:
            bar_colors.append("#e74c3c")
        else:
            bar_colors.append("#b0b0b0")

    # Public discourse bars (background, light blue)
    ax.barh(
        y_pos, pub_counts[cluster_ids_sorted],
        color="#4C72B0", alpha=0.25, height=0.7, label="Public discourse",
    )
    # Artist probe bars (overlay)
    ax.barh(
        y_pos, art_counts[cluster_ids_sorted],
        color=bar_colors, alpha=0.85, height=0.7,
    )

    # Labels
    ylabels = []
    for cid in cluster_ids_sorted:
        n_art_c = art_counts[cid]
        if n_art_c == 0:
            ylabels.append(f"Topic {cid}  (0 artist probes)")
        else:
            ylabels.append(f"Topic {cid}  ({n_art_c:,} artist probes)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("Number of chunks/probes", fontsize=11)

    # Annotation for top-4 concentration
    ax.text(
        0.97, 0.03,
        f"Top 4 clusters: {top4_pct:.1f}% of all artist probes",
        transform=ax.transAxes, fontsize=9.5, ha="right", va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#fff3f3",
            edgecolor="#e74c3c", alpha=0.92, linewidth=0.8,
        ),
    )

    handles = [
        mpatches.Patch(color="#4C72B0", alpha=0.25, label="Public discourse chunks"),
        mpatches.Patch(color="#e74c3c", alpha=0.85, label="Artist probes (top 4)"),
        mpatches.Patch(color="#b0b0b0", alpha=0.85, label="Artist probes (other)"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", framealpha=0.92)
    ax.set_title(
        "Artist Probe Concentration Across Topics",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, "figure_2_artist_probe_concentration", out_dirs)
    plt.close()
    print("  Saved figure_2_artist_probe_concentration")

    # ===================================================================
    # FIGURE 3: Macro-Thematic Distribution (grouped bar chart)
    # ===================================================================
    print("Figure 3: Macro-thematic distribution...")

    macro_order = [
        "Philosophy of Creativity",
        "Practice & Pedagogy",
        "Technical Genealogy",
        "Governance & Rights",
        "Institutions & Markets",
    ]

    # Public discourse: fraction per macro-theme
    pub_macro_counts = {}
    for macro in macro_order:
        cids = MACRO_THEMES[macro]
        pub_macro_counts[macro] = int(np.isin(labels_pub, cids).sum())
    pub_total = sum(pub_macro_counts.values())
    pub_macro_pct = {m: 100.0 * c / pub_total for m, c in pub_macro_counts.items()}

    # Artist probes: fraction per macro-theme
    art_macro_counts = {}
    for macro in macro_order:
        cids = MACRO_THEMES[macro]
        art_macro_counts[macro] = int(np.isin(art_cluster, cids).sum())
    art_total = sum(art_macro_counts.values())
    art_macro_pct = {m: 100.0 * c / art_total for m, c in art_macro_counts.items()}

    x = np.arange(len(macro_order))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars_pub = ax.bar(
        x - width / 2,
        [pub_macro_pct[m] for m in macro_order],
        width,
        color="#4C72B0",
        alpha=0.75,
        label=f"Public discourse (n={pub_total:,})",
    )
    bars_art = ax.bar(
        x + width / 2,
        [art_macro_pct[m] for m in macro_order],
        width,
        color="#e74c3c",
        alpha=0.75,
        label=f"Artist probes (n={art_total:,})",
    )

    # Value labels on bars
    for bar in bars_pub:
        h = bar.get_height()
        if h > 1:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
            )
    for bar in bars_art:
        h = bar.get_height()
        if h > 1:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(macro_order, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Percentage of total", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax.set_title(
        "Macro-Thematic Distribution: Public Discourse vs Artist Probes",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, "figure_3_macrotheme_distributions", out_dirs)
    plt.close()
    print("  Saved figure_3_macrotheme_distributions")

    # ===================================================================
    # FIGURE 4: Compression Metrics (two-panel)
    # ===================================================================
    print("Figure 4: Compression metrics...")

    # Load canonical H3 metrics from the pipeline output (single source of truth)
    h3_csv_path = ROOT / "figures" / "final_pipeline" / "h3_table.csv"
    if h3_csv_path.exists():
        h3_df = pd.read_csv(h3_csv_path)
        print("  Using canonical H3 metrics from h3_table.csv")
    else:
        raise FileNotFoundError(f"h3_table.csv not found at {h3_csv_path}. Run final_pipeline.py first.")

    themes_ordered = ["Ownership", "Utility", "Transparency", "Threat", "Compensation"]
    theme_key_map = {
        "Ownership": "ownership",
        "Utility": "utility",
        "Transparency": "transparency",
        "Threat": "threat",
        "Compensation": "compensation",
    }

    UNIQUE_FRAMES = {"Ownership": 24, "Utility": 3, "Transparency": 3, "Threat": 3, "Compensation": 37}
    theme_entropy: dict[str, float] = {}
    theme_coverage: dict[str, int] = {}
    theme_frame_counts: dict[str, int] = {}

    for display_name in themes_ordered:
        key = theme_key_map[display_name]
        row = h3_df[h3_df["theme"] == key].iloc[0]
        theme_entropy[display_name] = max(float(row["entropy_norm"]), 0.0)
        theme_coverage[display_name] = int(row["topics_any"])
        theme_frame_counts[display_name] = UNIQUE_FRAMES[display_name]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: Distributional entropy
    y_pos = np.arange(len(themes_ordered))
    entropy_vals = [theme_entropy[t] for t in themes_ordered]
    frame_counts = [theme_frame_counts[t] for t in themes_ordered]
    bar_labels = [f"{t} ({UNIQUE_FRAMES[t]} frames)" for t in themes_ordered]

    colors_a = [THEME_COLORS[theme_key_map[t]] for t in themes_ordered]
    ax1.barh(y_pos, entropy_vals, color=colors_a, alpha=0.8, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(bar_labels, fontsize=9.5)
    ax1.set_xlabel("Normalized distributional entropy", fontsize=10)
    ax1.set_title("A. Distributional Entropy by Theme", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 1.05)

    # Value labels
    for i, val in enumerate(entropy_vals):
        ax1.text(val + 0.02, i, f"{abs(val):.3f}", va="center", fontsize=9)

    # Panel B: Topic coverage
    coverage_vals = [theme_coverage[t] for t in themes_ordered]
    ax2.barh(y_pos, coverage_vals, color=colors_a, alpha=0.8, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bar_labels, fontsize=9.5)
    ax2.set_xlabel(f"Number of topics (out of {K})", fontsize=10)
    ax2.set_title("B. Topic Coverage by Theme", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, K + 1)

    # Value labels
    for i, val in enumerate(coverage_vals):
        ax2.text(val + 0.2, i, str(val), va="center", fontsize=9)

    plt.tight_layout()
    save_figure(fig, "figure_4_compression_metrics", out_dirs)
    plt.close()
    print("  Saved figure_4_compression_metrics")

    # ===================================================================
    # FIGURE S1: Macro-theme regions (50% density contours per macro-theme)
    # ===================================================================
    print("Figure S1: Macro-theme density contours...")
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.5))

    # Use the same axis range as Figure 1 for consistency
    s1_xr = xr
    s1_yr = yr

    for macro in macro_order:
        color = MACRO_COLORS[macro]
        cids = MACRO_THEMES[macro]
        mask = np.isin(labels_pub, cids)
        pts = pub2d[mask]
        if len(pts) < 10:
            continue

        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=color, alpha=0.22, s=7, edgecolors="none", zorder=1,
        )

        H_m, xe_m, ye_m = density_grid(pts, s1_xr, s1_yr)
        lv50 = density_threshold(H_m, 50)
        if lv50 > 0:
            ax.contourf(
                xe_m, ye_m, H_m, levels=[lv50, H_m.max()],
                colors=[color], alpha=0.22, zorder=2,
            )
            ax.contour(
                xe_m, ye_m, H_m, levels=[lv50],
                colors=[color], linewidths=1.4, alpha=0.85, zorder=3,
            )

    handles = [
        mpatches.Patch(color=MACRO_COLORS[m], alpha=0.55, label=m)
        for m in macro_order
    ]
    ax.legend(
        handles=handles, fontsize=8.5, loc="lower right",
        framealpha=0.92, edgecolor="#cccccc",
        title="Macro-Theme Density (50% HDR)",
        title_fontsize=9,
    )
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)", fontsize=11)
    ax.set_title(
        "Macro-Theme Regions in Public Discourse",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, "figure_S1_macro_theme_regions", out_dirs)
    plt.close()
    print("  Saved figure_S1_macro_theme_regions")

    # ===================================================================
    # FIGURE S2: Artist theme hulls (95% convex hulls)
    # ===================================================================
    print("Figure S2: Artist theme hulls...")
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.5))

    # Gray background: public discourse
    ax.scatter(
        pub2d[:, 0], pub2d[:, 1],
        c="#b0b0b0", alpha=0.30, s=8, edgecolors="none", zorder=1,
    )

    for theme, color in THEME_COLORS.items():
        mask = theme_labels_art == theme
        pts = art2d[mask]
        if len(pts) < 4:
            continue

        center = pts.mean(axis=0)
        hp = percentile_hull(pts, center, 95)
        if hp is not None:
            ax.fill(hp[:, 0], hp[:, 1], alpha=0.10, color=color, zorder=2)
            ax.plot(hp[:, 0], hp[:, 1], color=color, alpha=0.55, linewidth=1.0, zorder=3)
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=color, alpha=0.70, s=20, edgecolors="white", linewidths=0.3, zorder=4,
        )

    handles = [
        mpatches.Patch(
            color=c, alpha=0.6,
            label=f"{THEME_DISPLAY[t]} (n={int((theme_labels_art == t).sum())})",
        )
        for t, c in THEME_COLORS.items()
    ]
    handles.append(
        mpatches.Patch(
            color="#b0b0b0", alpha=0.45,
            label=f"Public discourse (n={len(pub2d):,})",
        )
    )
    ax.legend(
        handles=handles, fontsize=8.5, loc="lower right",
        framealpha=0.92, edgecolor="#cccccc",
        title="Artist Concern Themes (95% hulls)",
        title_fontsize=9,
    )
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)", fontsize=11)
    ax.set_title(
        "Artist Concern Themes in Public Discourse",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, "figure_S2_artist_concern_themes", out_dirs)
    plt.close()
    print("  Saved figure_S2_artist_concern_themes")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nAll figures saved to:")
    for d in out_dirs:
        print(f"  {d}")
    print(f"\nPipeline config: k={K}, nn={N_NEIGHBORS}, md={MIN_DIST}, nc={N_COMPONENTS}")
    print(f"Public discourse: {n_pub} chunks | Artist probes: {n_art} | Public probes: {len(df_pp)}")
    print("Done.")


if __name__ == "__main__":
    main()
