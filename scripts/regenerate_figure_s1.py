"""Regenerate Figure S1 (macro-theme density contours) from the shared cache.

All semantic-map figures now load from figures/final_pipeline/figure_cache/.
If the cache is missing, run scripts/build_figure_cache.py first.
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

CACHE_DIR = ROOT / "figures" / "final_pipeline" / "figure_cache"
OUT_DIRS = [
    ROOT / "figures" / "manuscript",
    ROOT / "jan_2026_manuscript" / "NMS2026" / "figures",
    ROOT / "jan_2026_manuscript" / "NatureCollection2026" / "figures",
]

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


def density_grid(pts: np.ndarray, xr, yr, bins: int = 80, sigma: float = 1.8):
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=[xr, yr])
    H = gaussian_filter(H.T, sigma=sigma)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H, xc, yc


def density_threshold(H: np.ndarray, pct: float) -> float:
    flat = H.flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return 0.0
    idx = np.argsort(flat)[::-1]
    cs = np.cumsum(flat[idx])
    total = cs[-1]
    k = int(np.searchsorted(cs, pct / 100.0 * total))
    k = min(k, len(flat) - 1)
    return float(flat[idx[k]])


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
    labels_pub = np.load(CACHE_DIR / "labels_pub.npy")
    var_pct = np.load(CACHE_DIR / "var_pct.npy")
    axes = np.load(CACHE_DIR / "axes.npz")
    xr = (float(axes["xr"][0]), float(axes["xr"][1]))
    yr = (float(axes["yr"][0]), float(axes["yr"][1]))

    print(f"  PC1={var_pct[0]:.2f}% variance, PC2={var_pct[1]:.2f}% variance")

    print("Rendering Figure S1 (density contours per macro-theme)...")
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.5))

    for macro, cids in MACRO_THEMES.items():
        color = MACRO_COLORS[macro]
        mask = np.isin(labels_pub, cids)
        pts = pub2d[mask]
        if len(pts) < 10:
            continue

        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=color, alpha=0.22, s=7, edgecolors="none", zorder=1,
        )

        H_m, xe_m, ye_m = density_grid(pts, xr, yr)
        lv50 = density_threshold(H_m, 50)
        if lv50 > 0:
            ax.contourf(
                xe_m, ye_m, H_m, levels=[lv50, H_m.max()],
                colors=[color], alpha=0.25, zorder=2,
            )
            ax.contour(
                xe_m, ye_m, H_m, levels=[lv50],
                colors=[color], linewidths=1.4, alpha=0.90, zorder=3,
            )

    handles = [
        mpatches.Patch(color=MACRO_COLORS[m], alpha=0.55, label=m)
        for m in MACRO_THEMES
    ]
    ax.legend(
        handles=handles, fontsize=8.5, loc="upper right",
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
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    plt.tight_layout()
    save_figure(fig, "figure_S1_macro_theme_regions")
    plt.close()
    print("Saved figure_S1_macro_theme_regions")


if __name__ == "__main__":
    main()
