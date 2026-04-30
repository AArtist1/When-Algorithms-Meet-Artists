"""Generate 5 variants of Figure 1 (semantic density) with different visual approaches.

Uses canonical pipeline data (5D consensus coords, KMeans k=20, MLP projection).
All variants share the same PCA(2D) projection; they differ in how points are
colored and whether density contours, topic regions, or centroids are shown.

Variant A: Topic assignment only (20 colors, no density)
Variant B: Topic assignment + artist probe density contours
Variant C: Topic convex hulls + public-vs-artist coloring
Variant D: Topic assignment + dual density contours (public + artist)
Variant E: Topic centroids with labels + density + artist overlay (minimalist)

Outputs:
    figures/manuscript/figure_1_variant_A_topics_only.png
    figures/manuscript/figure_1_variant_B_topics_plus_artist_density.png
    figures/manuscript/figure_1_variant_C_hulls_plus_source.png
    figures/manuscript/figure_1_variant_D_topics_plus_dual_density.png
    figures/manuscript/figure_1_variant_E_centroids_density_minimalist.png

Side effects:
    Reads .npy and .csv files. Trains MLP projection head.
    Writes 5 PNG files (300 DPI). Prints progress.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.clustering import run_kmeans
from src.projection import train_projection_head, project_to_consensus_space

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K = 20
PROJECTION_LAYERS = (1024, 512, 256, 128)
PROJECTION_LR = 0.002
GRID_DIR = ROOT / "figures" / "prefix_grid_search"
PREFIX_DIR = ROOT / "figures" / "prefix_comparison"
OUT_DIR = ROOT / "figures" / "manuscript"

# Tab20 colormap for 20 topics
TOPIC_CMAP = plt.cm.tab20
TOPIC_COLORS = [TOPIC_CMAP(i / 20) for i in range(20)]

SHORT_LABELS = {
    0: "Decentralized Infra",
    1: "AI as Collaborator",
    2: "ML Art Theory",
    3: "Personal Reflections",
    4: "Mental Models",
    5: "Cohen/AARON",
    6: "Arts Journalism",
    7: "Art Practice Talk",
    8: "Copyright/Legal",
    9: "Digital Exhibition",
    10: "Artist Defense Tools",
    11: "Authorship/Agency",
    12: "Media/AI Panic",
    13: "Authenticity/Creativity",
    14: "AI Creative Tools",
    15: "DeepDream/Neural",
    16: "Artist Reflections",
    17: "Artist-Centered AI",
    18: "Art Copyright Debates",
    19: "Generative Art History",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# Density helpers (from regenerate_all_figures.py)
# ---------------------------------------------------------------------------

def density_grid(pts, xr, yr, bins=120, sigma=2.5):
    H, xe, ye = np.histogram2d(
        pts[:, 0], pts[:, 1],
        bins=bins, range=[xr, yr],
    )
    H = gaussian_filter(H.T, sigma=sigma)
    xe = 0.5 * (xe[:-1] + xe[1:])
    ye = 0.5 * (ye[:-1] + ye[1:])
    return H, xe, ye


def density_threshold(H, pct):
    flat = H.flatten()
    total = flat.sum()
    if total == 0:
        return 0.0
    idx = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[idx])
    cutoff = np.searchsorted(cum, pct / 100.0 * total)
    return float(flat[idx[min(cutoff, len(idx) - 1)]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "figures" / "final_pipeline" / "figure_cache"
    if not (cache / "pub2d.npy").exists():
        raise FileNotFoundError(
            f"Figure cache not found at {cache}. "
            "Run `python scripts/build_figure_cache.py` first."
        )

    print(f"Loading shared figure cache from {cache.relative_to(ROOT)}/")
    pub2d = np.load(cache / "pub2d.npy")
    art2d = np.load(cache / "art2d.npy")
    labels_pub = np.load(cache / "labels_pub.npy")
    art_labels = np.load(cache / "art_cluster.npy")
    var_pct = np.load(cache / "var_pct.npy")
    axes = np.load(cache / "axes.npz")
    xr = [float(axes["xr"][0]), float(axes["xr"][1])]
    yr = [float(axes["yr"][0]), float(axes["yr"][1])]
    print(f"  PC1={var_pct[0]:.2f}% variance, PC2={var_pct[1]:.2f}% variance")

    H_pub, xe, ye = density_grid(pub2d, xr, yr)
    H_art, xe_a, ye_a = density_grid(art2d, xr, yr)

    # Centroids per topic (2D)
    centroids_2d = np.array([pub2d[labels_pub == k].mean(axis=0) for k in range(K)])

    def xlabel():
        return f"PC1 ({var_pct[0]:.1f}% variance)"
    def ylabel():
        return f"PC2 ({var_pct[1]:.1f}% variance)"

    # =================================================================
    # VARIANT A: Topic assignment only (20 colors, no density)
    # =================================================================
    print("\nVariant A: Topic assignment only...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for k in range(K):
        mask = labels_pub == k
        ax.scatter(
            pub2d[mask, 0], pub2d[mask, 1],
            c=[TOPIC_COLORS[k]], alpha=0.35, s=6, edgecolors="none",
            label=f"T{k}",
        )
    # Artist probes as black outlined diamonds
    ax.scatter(
        art2d[:, 0], art2d[:, 1],
        c="#e74c3c", alpha=0.5, s=12, edgecolors="black",
        linewidths=0.3, marker="D", zorder=10, label="Artist probes",
    )
    ax.set_xlabel(xlabel(), fontsize=11)
    ax.set_ylabel(ylabel(), fontsize=11)
    ax.set_title("Variant A: Points by Topic Assignment (20 topics)", fontsize=12.5, fontweight="bold", pad=14)
    ax.legend(fontsize=6.5, loc="upper right", ncol=3, framealpha=0.85, markerscale=1.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figure_1_variant_A_topics_only.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved variant A")

    # =================================================================
    # VARIANT B: Topic assignment + artist probe density contours
    # =================================================================
    print("Variant B: Topics + artist density contours...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for k in range(K):
        mask = labels_pub == k
        ax.scatter(
            pub2d[mask, 0], pub2d[mask, 1],
            c=[TOPIC_COLORS[k]], alpha=0.3, s=5, edgecolors="none",
        )
    # Artist density contours
    for p in [30, 50, 70, 90]:
        lv = density_threshold(H_art, p)
        if lv > 0:
            ax.contour(xe_a, ye_a, H_art, levels=[lv],
                       colors="#c0392b", alpha=0.5, linewidths=1.2 if p == 50 else 0.6)
    art50 = density_threshold(H_art, 50)
    if art50 > 0:
        ax.contour(xe_a, ye_a, H_art, levels=[art50],
                   colors="#c0392b", linewidths=2.0, linestyles="-", zorder=6)
    ax.scatter(
        art2d[:, 0], art2d[:, 1],
        c="#e74c3c", alpha=0.4, s=8, edgecolors="none", zorder=5,
    )
    ax.set_xlabel(xlabel(), fontsize=11)
    ax.set_ylabel(ylabel(), fontsize=11)
    ax.set_title("Variant B: Topics + Artist Density Contours", fontsize=12.5, fontweight="bold", pad=14)
    handles = [mpatches.Patch(color=TOPIC_COLORS[k], alpha=0.6, label=f"T{k}")
              for k in range(K)]
    handles.extend([
        mpatches.Patch(color="#e74c3c", alpha=0.6, label="Artist probes"),
        plt.Line2D([0], [0], color="#c0392b", lw=2, label="Artist 50% HDR"),
    ])
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=3,
              framealpha=0.92, markerscale=1.2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figure_1_variant_B_topics_plus_artist_density.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved variant B")

    # =================================================================
    # VARIANT C: Topic convex hulls + public-vs-artist coloring
    # =================================================================
    print("Variant C: Topic hulls + source coloring...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # Draw convex hulls per topic (light fill)
    for k in range(K):
        pts = pub2d[labels_pub == k]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
                poly = plt.Polygon(verts, alpha=0.08, color=TOPIC_COLORS[k],
                                   edgecolor=TOPIC_COLORS[k], linewidth=0.5)
                ax.add_patch(poly)
            except Exception:
                pass
        # Centroid label
        cx, cy = centroids_2d[k]
        ax.text(cx, cy, f"T{k}", fontsize=6, ha="center", va="center",
                fontweight="bold", color=TOPIC_COLORS[k], alpha=0.7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.5))
    # Public discourse (blue)
    ax.scatter(pub2d[:, 0], pub2d[:, 1],
               c="#4C72B0", alpha=0.08, s=3, edgecolors="none", zorder=1)
    # Artist probes (red)
    ax.scatter(art2d[:, 0], art2d[:, 1],
               c="#e74c3c", alpha=0.5, s=10, edgecolors="none", zorder=5)
    handles = [
        mpatches.Patch(color="#4C72B0", alpha=0.3, label="Public discourse (n=1,736)"),
        mpatches.Patch(color="#e74c3c", alpha=0.6, label="Artist probes (n=1,259)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.92)
    ax.set_xlabel(xlabel(), fontsize=11)
    ax.set_ylabel(ylabel(), fontsize=11)
    ax.set_title("Variant C: Topic Convex Hulls + Source Coloring", fontsize=12.5, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figure_1_variant_C_hulls_plus_source.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved variant C")

    # =================================================================
    # VARIANT D: Topic assignment + dual density contours
    # =================================================================
    print("Variant D: Topics + dual density...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for k in range(K):
        mask = labels_pub == k
        ax.scatter(
            pub2d[mask, 0], pub2d[mask, 1],
            c=[TOPIC_COLORS[k]], alpha=0.25, s=4, edgecolors="none",
        )
    # Public density contours (blue)
    for p in [50, 70, 90]:
        lv = density_threshold(H_pub, p)
        if lv > 0:
            ax.contour(xe, ye, H_pub, levels=[lv],
                       colors="#2c5aa0", alpha=0.35, linewidths=0.7)
    pub50 = density_threshold(H_pub, 50)
    if pub50 > 0:
        ax.contour(xe, ye, H_pub, levels=[pub50],
                   colors="#2c5aa0", linewidths=1.8, linestyles="--", zorder=4)
    # Artist density contours (red)
    for p in [50, 70, 90]:
        lv = density_threshold(H_art, p)
        if lv > 0:
            ax.contour(xe_a, ye_a, H_art, levels=[lv],
                       colors="#c0392b", alpha=0.5, linewidths=0.7)
    art50 = density_threshold(H_art, 50)
    if art50 > 0:
        ax.contour(xe_a, ye_a, H_art, levels=[art50],
                   colors="#c0392b", linewidths=2.0, linestyles="-", zorder=6)
    ax.scatter(art2d[:, 0], art2d[:, 1],
               c="#e74c3c", alpha=0.35, s=7, edgecolors="none", zorder=5)
    handles = [mpatches.Patch(color=TOPIC_COLORS[k], alpha=0.6, label=f"T{k}")
              for k in range(K)]
    handles.extend([
        mpatches.Patch(color="#e74c3c", alpha=0.5, label="Artist probes"),
        plt.Line2D([0], [0], color="#2c5aa0", lw=1.8, ls="--", label="Public 50% HDR"),
        plt.Line2D([0], [0], color="#c0392b", lw=2.0, ls="-", label="Artist 50% HDR"),
    ])
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=3,
              framealpha=0.92, markerscale=1.2)
    ax.set_xlabel(xlabel(), fontsize=11)
    ax.set_ylabel(ylabel(), fontsize=11)
    ax.set_title("Variant D: Topics + Dual Density Contours", fontsize=12.5, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figure_1_variant_D_topics_plus_dual_density.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved variant D")

    # =================================================================
    # VARIANT E: Centroids with labels + density + artist overlay (minimalist)
    # =================================================================
    print("Variant E: Centroids + density + artist overlay (minimalist)...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # Public density as filled contours (light blue)
    for p in [30, 50, 70, 85, 95]:
        lv = density_threshold(H_pub, p)
        if lv > 0:
            if p == 30:
                ax.contourf(xe, ye, H_pub, levels=[lv, H_pub.max()],
                            colors=["#4C72B0"], alpha=0.06)
            ax.contour(xe, ye, H_pub, levels=[lv],
                       colors="#4C72B0", alpha=0.25, linewidths=0.5)
    # Artist density as filled contours (light red)
    for p in [30, 50, 70, 85]:
        lv = density_threshold(H_art, p)
        if lv > 0:
            if p == 30:
                ax.contourf(xe_a, ye_a, H_art, levels=[lv, H_art.max()],
                            colors=["#e74c3c"], alpha=0.08)
            ax.contour(xe_a, ye_a, H_art, levels=[lv],
                       colors="#e74c3c", alpha=0.35, linewidths=0.5)
    # Bold 50% HDR
    if pub50 > 0:
        ax.contour(xe, ye, H_pub, levels=[pub50],
                   colors="#2c5aa0", linewidths=1.6, linestyles="--", zorder=4)
    if art50 > 0:
        ax.contour(xe_a, ye_a, H_art, levels=[art50],
                   colors="#c0392b", linewidths=2.2, linestyles="-", zorder=6)
    # Topic centroids with short labels
    for k in range(K):
        cx, cy = centroids_2d[k]
        # Determine if this topic has artist probes
        n_art_k = (art_labels == k).sum()
        marker_color = "#e74c3c" if n_art_k > 5 else "#4C72B0"
        marker_size = 110 + n_art_k * 0.45
        ax.scatter(cx, cy, c=marker_color, s=marker_size, edgecolors="black",
                   linewidths=0.7, zorder=8, alpha=0.85)
        ax.annotate(
            f"T{k}", xy=(cx, cy), xytext=(0, -11),
            textcoords="offset points",
            fontsize=8.5, ha="center", va="top",
            fontweight="bold", color="#111111", zorder=9,
        )
    # Artist probes as small dots
    ax.scatter(art2d[:, 0], art2d[:, 1],
               c="#e74c3c", alpha=0.2, s=4, edgecolors="none", zorder=5)
    handles = [
        plt.Line2D([0], [0], color="#2c5aa0", lw=1.6, ls="--", label="Public 50% HDR"),
        plt.Line2D([0], [0], color="#c0392b", lw=2.2, ls="-", label="Artist 50% HDR"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=8, label="Topic with artist probes"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0",
                   markersize=8, label="Topic without artist probes"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.set_xlabel(xlabel(), fontsize=11)
    ax.set_ylabel(ylabel(), fontsize=11)
    ax.set_title("Variant E: Topic Centroids + Density (Minimalist)", fontsize=12.5, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figure_1_variant_E_centroids_density_minimalist.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved variant E")

    print(f"\nAll 5 variants saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
