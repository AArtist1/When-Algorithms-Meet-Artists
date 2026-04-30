"""Alternative Figure S2 rendering: 50% HDR density contours per theme.

Matches the visual style of Figure 1 and Figure S1 (which both use 50%
highest-density-region contours) instead of 95% convex hulls. Data source
is identical — the cached art2d coordinates and artist theme labels — so
the two versions sit on exactly the same PCA axes and point positions.

Run alongside the canonical scripts/regenerate_figure_s2.py; outputs a
separate file so the original convex-hull version is preserved for
comparison.

Outputs (in addition to the canonical hull version):
    figures/manuscript/figure_S2_density.{png,pdf}
    jan_2026_manuscript/NMS2026/figures/figure_S2_density.{png,pdf}
    jan_2026_manuscript/NatureCollection2026/figures/figure_S2_density.{png,pdf}
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
    art2d = np.load(CACHE_DIR / "art2d.npy")
    theme_labels = np.load(CACHE_DIR / "theme_labels_art.npy", allow_pickle=True)
    var_pct = np.load(CACHE_DIR / "var_pct.npy")
    axes = np.load(CACHE_DIR / "axes.npz")
    xr = (float(axes["xr"][0]), float(axes["xr"][1]))
    yr = (float(axes["yr"][0]), float(axes["yr"][1]))

    print(f"  PC1={var_pct[0]:.2f}% variance, PC2={var_pct[1]:.2f}% variance")

    print("Rendering Figure S2 (density contours per theme)...")
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.5))

    # Grey backdrop: public discourse so the reader has context
    ax.scatter(
        pub2d[:, 0], pub2d[:, 1],
        c="#b0b0b0", alpha=0.22, s=6, edgecolors="none", zorder=1,
    )

    # 50% HDR density contour per theme
    for theme, color in THEME_COLORS.items():
        mask = theme_labels == theme
        pts = art2d[mask]
        if len(pts) < 10:
            continue

        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=color, alpha=0.35, s=7, edgecolors="none", zorder=2,
        )

        H_t, xe_t, ye_t = density_grid(pts, xr, yr)
        lv50 = density_threshold(H_t, 50)
        if lv50 > 0:
            ax.contourf(
                xe_t, ye_t, H_t, levels=[lv50, H_t.max()],
                colors=[color], alpha=0.22, zorder=3,
            )
            ax.contour(
                xe_t, ye_t, H_t, levels=[lv50],
                colors=[color], linewidths=1.6, alpha=0.90, zorder=4,
            )

    handles = [
        mpatches.Patch(
            color=c, alpha=0.55,
            label=f"{THEME_DISPLAY[t]} (n={int((theme_labels == t).sum())})",
        )
        for t, c in THEME_COLORS.items()
    ]
    handles.append(mpatches.Patch(
        color="#b0b0b0", alpha=0.40,
        label=f"Public discourse (n={len(pub2d):,})",
    ))
    ax.legend(
        handles=handles, fontsize=8.5, loc="upper right",
        framealpha=0.92, edgecolor="#cccccc",
        title="Artist Concern Themes (50% HDR)",
        title_fontsize=9,
    )
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)", fontsize=11)
    ax.set_title(
        "Artist Concern Themes in Public Discourse (density version)",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    plt.tight_layout()
    save_figure(fig, "figure_S2_density")
    plt.close()
    print("Saved figure_S2_density to all output dirs")


if __name__ == "__main__":
    main()
