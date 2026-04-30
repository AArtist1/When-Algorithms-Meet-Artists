"""Regenerate Figure S2 (artist concern themes in public discourse space).

Loads 2D coords from the shared figure cache so PCA axes match every
other semantic-map figure. If the cache is missing, run
scripts/build_figure_cache.py first.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "figures" / "final_pipeline" / "figure_cache"
OUT_DIRS = [
    ROOT / "figures" / "manuscript",
    ROOT / "jan_2026_manuscript" / "NMS2026" / "figures",
    ROOT / "jan_2026_manuscript" / "NatureCollection2026" / "figures",
]

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


def percentile_hull(pts: np.ndarray, center: np.ndarray, pct: float):
    dists = np.linalg.norm(pts - center, axis=1)
    k = max(3, int(len(pts) * pct / 100.0))
    k = min(k, len(pts))
    idx = np.argsort(dists)[:k]
    inner = pts[idx]
    if len(inner) < 3:
        return None
    try:
        hull = ConvexHull(inner)
    except Exception:
        return None
    verts = inner[hull.vertices]
    return np.vstack([verts, verts[:1]])


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
    theme_labels = np.load(CACHE_DIR / "theme_labels_art.npy", allow_pickle=True)
    var_pct = np.load(CACHE_DIR / "var_pct.npy")
    axes = np.load(CACHE_DIR / "axes.npz")
    xr = (float(axes["xr"][0]), float(axes["xr"][1]))
    yr = (float(axes["yr"][0]), float(axes["yr"][1]))

    print(f"  PC1={var_pct[0]:.2f}% variance, PC2={var_pct[1]:.2f}% variance")

    print("Rendering Figure S2...")
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.5))

    ax.scatter(
        pub2d[:, 0], pub2d[:, 1],
        c="#b0b0b0", alpha=0.30, s=8, edgecolors="none", zorder=1,
    )

    for theme, color in THEME_COLORS.items():
        mask = theme_labels == theme
        pts = art2d[mask]
        if len(pts) < 4:
            continue

        center = pts.mean(axis=0)
        hp = percentile_hull(pts, center, 95)
        if hp is not None:
            ax.fill(hp[:, 0], hp[:, 1], alpha=0.10, color=color, zorder=2)
            ax.plot(hp[:, 0], hp[:, 1], color=color, alpha=0.55,
                    linewidth=1.0, zorder=3)
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=color, alpha=0.70, s=20,
            edgecolors="white", linewidths=0.3, zorder=4,
        )

    handles = [
        mpatches.Patch(
            color=c, alpha=0.6,
            label=f"{THEME_DISPLAY[t]} (n={int((theme_labels == t).sum())})",
        )
        for t, c in THEME_COLORS.items()
    ]
    handles.append(mpatches.Patch(
        color="#b0b0b0", alpha=0.45,
        label=f"Public discourse (n={len(pub2d):,})",
    ))
    ax.legend(
        handles=handles, fontsize=8.5, loc="upper right",
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
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    plt.tight_layout()
    save_figure(fig, "figure_S2_artist_concern_themes")
    plt.close()
    print("Saved figure_S2_artist_concern_themes")


if __name__ == "__main__":
    main()
