"""Regenerate canonical Figure 1 from the shared figure cache.

All figure scripts now load from figures/final_pipeline/figure_cache/ so
every semantic-map figure uses identical PCA axes, point coordinates, and
cluster assignments. If the cache is missing, run scripts/build_figure_cache.py
first.

Meeting-driven design choices (Ariya + Oliver, 2026-04-12):
  - Public points colored by topic assignment (20 tab20 colors); Ariya said
    the topic-colored view was "easier to understand visually."
  - 50% HDR fill and 50% HDR bold contour line live at the same threshold
    for both public and artist distributions (previously the fill was at
    30% while the line was at 50% — Oliver called this out as looking
    like 60-70%).
  - Legend in upper right with T-prefix topic entries.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

K = 20
CACHE_DIR = ROOT / "figures" / "final_pipeline" / "figure_cache"
OUT_DIRS = [
    ROOT / "figures" / "manuscript",
    ROOT / "jan_2026_manuscript" / "NMS2026" / "figures",
    ROOT / "jan_2026_manuscript" / "NatureCollection2026" / "figures",
]

TOPIC_CMAP = plt.cm.tab20
TOPIC_COLORS = [TOPIC_CMAP(i / 20) for i in range(K)]


def density_grid(pts: np.ndarray, xr, yr, bins: int = 120, sigma: float = 2.5):
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=[xr, yr])
    H = gaussian_filter(H.T, sigma=sigma)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H, xc, yc


def density_threshold(H: np.ndarray, pct: float) -> float:
    flat = H.flatten()
    total = flat.sum()
    if total == 0:
        return 0.0
    idx = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[idx])
    cutoff = np.searchsorted(cum, pct / 100.0 * total)
    return float(flat[idx[min(cutoff, len(idx) - 1)]])


def save_figure(fig, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")


def main() -> None:
    if not (CACHE_DIR / "pub2d.npy").exists():
        raise FileNotFoundError(
            f"Figure cache not found at {CACHE_DIR}. "
            "Run `python scripts/build_figure_cache.py` first."
        )

    print(f"Loading figure cache from {CACHE_DIR.relative_to(ROOT)}/")
    pub2d = np.load(CACHE_DIR / "pub2d.npy")
    art2d = np.load(CACHE_DIR / "art2d.npy")
    labels_pub = np.load(CACHE_DIR / "labels_pub.npy")
    var_pct = np.load(CACHE_DIR / "var_pct.npy")
    axes = np.load(CACHE_DIR / "axes.npz")
    xr = (float(axes["xr"][0]), float(axes["xr"][1]))
    yr = (float(axes["yr"][0]), float(axes["yr"][1]))

    print(f"  PC1={var_pct[0]:.2f}% variance, PC2={var_pct[1]:.2f}% variance")

    H_pub, xe_p, ye_p = density_grid(pub2d, xr, yr)
    H_art, xe_a, ye_a = density_grid(art2d, xr, yr)
    pub50 = density_threshold(H_pub, 50)
    art50 = density_threshold(H_art, 50)

    print("Rendering Figure 1...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.8))

    if pub50 > 0:
        ax.contourf(
            xe_p, ye_p, H_pub, levels=[pub50, H_pub.max()],
            colors=["#4C72B0"], alpha=0.09, zorder=1,
        )

    for k in range(K):
        mask = labels_pub == k
        if not mask.any():
            continue
        ax.scatter(
            pub2d[mask, 0], pub2d[mask, 1],
            c=[TOPIC_COLORS[k]], alpha=0.55, s=7,
            edgecolors="none", zorder=2,
        )

    if art50 > 0:
        ax.contourf(
            xe_a, ye_a, H_art, levels=[art50, H_art.max()],
            colors=["#e74c3c"], alpha=0.14, zorder=3,
        )

    ax.scatter(
        art2d[:, 0], art2d[:, 1],
        c="#e74c3c", alpha=0.70, s=10,
        marker="D", edgecolors="#7b1c12", linewidths=0.25, zorder=5,
    )

    if pub50 > 0:
        ax.contour(
            xe_p, ye_p, H_pub, levels=[pub50],
            colors="#2c5aa0", linewidths=1.8, linestyles="--", zorder=6,
        )
    if art50 > 0:
        ax.contour(
            xe_a, ye_a, H_art, levels=[art50],
            colors="#c0392b", linewidths=2.2, linestyles="-", zorder=7,
        )

    if art50 > 0:
        mask_a = H_art >= art50
        if mask_a.any():
            rows, cols = np.where(mask_a)
            w = H_art[mask_a]
            acx = float(np.average(xe_a[cols], weights=w))
            acy = float(np.average(ye_a[rows], weights=w))
            ax.annotate(
                "50% of artist concerns",
                xy=(acx, acy + 0.7),
                xytext=(acx + 3.2, acy + 2.6),
                fontsize=9.5, color="#c0392b", fontweight="bold",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b",
                                linewidth=1.1, alpha=0.85),
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", edgecolor="#e74c3c",
                          alpha=0.94, linewidth=0.8),
                zorder=10,
            )

    handles = [mpatches.Patch(color=TOPIC_COLORS[k], alpha=0.75, label=f"T{k}")
              for k in range(K)]
    handles.extend([
        mpatches.Patch(color="#e74c3c", alpha=0.70, label="Artist probes"),
        plt.Line2D([0], [0], color="#2c5aa0", lw=1.8, linestyle="--",
                   label="Public 50% HDR"),
        plt.Line2D([0], [0], color="#c0392b", lw=2.2, linestyle="-",
                   label="Artist 50% HDR"),
    ])
    ax.legend(
        handles=handles, fontsize=7.2, loc="upper right", ncol=3,
        framealpha=0.94, edgecolor="#cccccc",
        title="Public discourse topics (n=1,736)\nArtist probes (n=1,259)",
        title_fontsize=8,
    )

    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)", fontsize=11)
    ax.set_title(
        "Semantic Density: Public Discourse vs Artist Concerns",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    plt.tight_layout()
    save_figure(fig, "figure_1_semantic_density")
    plt.close()
    print("Saved figure_1_semantic_density")


if __name__ == "__main__":
    main()
