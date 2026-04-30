"""Regenerate Figure 4 (six-panel compression metrics) with corrected Panel E.

Rebuild of the canonical six-panel figure that Oliver flagged in the
2026-04-12 meeting with Ariya. The previous 6-panel PNG (dated April 9)
was produced by a one-off script that is no longer in the repo, so this
rebuilds it from the canonical data sources.

Meeting-driven fixes:
  Panel E ("Opinion Diversity vs Discursive Spread"): the original
  "proportional representation" diagonal line was misleading — Oliver
  said it implied a near-flat relationship when true proportional
  representation (1 frame = 1 topic) would be a steep y=x line. This
  version plots y=x explicitly as the proportional-representation
  reference and annotates each theme's vertical gap from that line as
  the compression deficit.

Panel D ("Artist Consensus"): values are now computed from
  data/artist_perspectives.csv rather than hard-coded. The previous
  figure showed Utility at 66%; the actual Lovato-subset value is 46%.

Data sources:
  figures/final_pipeline/h3_table.csv   — entropy, topic coverage, FCR
  data/artist_perspectives.csv          — artist consensus per theme

Outputs:
  figures/manuscript/figure_4_compression_metrics.{png,pdf}
  jan_2026_manuscript/NMS2026/figures/figure_4_compression_metrics.{png,pdf}
  jan_2026_manuscript/NatureCollection2026/figures/figure_4_compression_metrics.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

H3_CSV = ROOT / "figures" / "final_pipeline" / "h3_table.csv"
ARTIST_CSV = ROOT / "data" / "artist_perspectives.csv"
OUT_DIRS = [
    ROOT / "figures" / "manuscript",
    ROOT / "jan_2026_manuscript" / "NMS2026" / "figures",
    ROOT / "jan_2026_manuscript" / "NatureCollection2026" / "figures",
]

THEMES_ORDER = ["Ownership", "Utility", "Transparency", "Threat", "Compensation"]
THEME_KEY = {
    "Ownership": "ownership",
    "Utility": "utility",
    "Transparency": "transparency",
    "Threat": "threat",
    "Compensation": "compensation",
}
THEME_COLORS = {
    "Ownership": "#3498db",
    "Utility": "#2ecc71",
    "Transparency": "#9b59b6",
    "Threat": "#e74c3c",
    "Compensation": "#f39c12",
}
K_TOPICS = 20


def compute_artist_consensus(df: pd.DataFrame) -> dict[str, float]:
    """Return the % of artists holding the modal position per theme."""
    out: dict[str, float] = {}
    for display, key in THEME_KEY.items():
        sub = df[df["question_group"] == key]
        top_count = int(sub["perspective_text"].value_counts().iloc[0])
        out[display] = 100.0 * top_count / len(sub)
    return out


def save_figure(fig, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")


def main() -> None:
    h3 = pd.read_csv(H3_CSV).set_index("theme")
    art = pd.read_csv(ARTIST_CSV)

    entropy = {t: max(float(h3.loc[THEME_KEY[t], "entropy_norm"]), 0.0) for t in THEMES_ORDER}
    topics = {t: int(h3.loc[THEME_KEY[t], "topics_any"]) for t in THEMES_ORDER}
    frames = {t: int(h3.loc[THEME_KEY[t], "frames"]) for t in THEMES_ORDER}
    fcr = {t: float(h3.loc[THEME_KEY[t], "fcr"]) for t in THEMES_ORDER}
    consensus = compute_artist_consensus(art)

    print("Theme metrics:")
    for t in THEMES_ORDER:
        print(f"  {t}: frames={frames[t]}, topics={topics[t]}, "
              f"fcr={fcr[t]:.1f}, entropy={entropy[t]:.3f}, "
              f"consensus={consensus[t]:.1f}%")

    bar_labels = [f"{t}\n({frames[t]} frames)" for t in THEMES_ORDER]
    colors = [THEME_COLORS[t] for t in THEMES_ORDER]
    y_pos = np.arange(len(THEMES_ORDER))

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 14))
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axes

    # ---- Panel A: Distributional Entropy ----
    vals_a = [entropy[t] for t in THEMES_ORDER]
    ax_a.barh(y_pos, vals_a, color=colors, alpha=0.85, height=0.62)
    for i, v in enumerate(vals_a):
        ax_a.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(bar_labels, fontsize=9)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Normalized entropy (0 = total compression)", fontsize=10)
    ax_a.set_xlim(0, max(vals_a) * 1.25 + 0.05)
    ax_a.set_title("A. Distributional Entropy", fontsize=11.5, fontweight="bold")
    ax_a.grid(True, axis="x", alpha=0.2)

    # ---- Panel B: Topic Coverage ----
    vals_b = [topics[t] for t in THEMES_ORDER]
    ax_b.barh(y_pos, vals_b, color=colors, alpha=0.85, height=0.62)
    for i, v in enumerate(vals_b):
        ax_b.text(v + 0.1, i, str(v), va="center", fontsize=10, fontweight="bold")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(bar_labels, fontsize=9)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(f"Number of topics (out of {K_TOPICS})", fontsize=10)
    ax_b.set_xlim(0, max(vals_b) + 1.2)
    ax_b.set_title("B. Topic Coverage", fontsize=11.5, fontweight="bold")
    ax_b.grid(True, axis="x", alpha=0.2)

    # ---- Panel C: Compression Ratio ----
    vals_c = [fcr[t] for t in THEMES_ORDER]
    ax_c.barh(y_pos, vals_c, color=colors, alpha=0.85, height=0.62)
    for i, v in enumerate(vals_c):
        ax_c.text(v + 0.3, i, f"{v:g}:1", va="center", fontsize=10, fontweight="bold")
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(bar_labels, fontsize=9)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Frame-to-topic compression ratio (FCR)", fontsize=10)
    ax_c.set_xlim(0, max(vals_c) * 1.12 + 1)
    ax_c.set_title("C. Compression Ratio (frames / topics)",
                   fontsize=11.5, fontweight="bold")
    ax_c.grid(True, axis="x", alpha=0.2)

    # ---- Panel D: Artist Consensus (from Lovato et al. survey) ----
    vals_d = [consensus[t] for t in THEMES_ORDER]
    ax_d.barh(y_pos, vals_d, color=colors, alpha=0.85, height=0.62)
    for i, v in enumerate(vals_d):
        ax_d.text(v + 1.0, i, f"{v:.0f}%", va="center", fontsize=10, fontweight="bold")
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(bar_labels, fontsize=9)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("% of artists holding the most common position", fontsize=10)
    ax_d.set_xlim(0, 100)
    ax_d.set_title("D. Artist Consensus (Lovato et al. survey)",
                   fontsize=11.5, fontweight="bold")
    ax_d.grid(True, axis="x", alpha=0.2)

    # ---- Panel E: Opinion Diversity vs Discursive Spread ----
    x_frames = np.array([frames[t] for t in THEMES_ORDER], dtype=float)
    y_topics = np.array([topics[t] for t in THEMES_ORDER], dtype=float)

    # Slightly jitter overlapping points (Transparency + Threat both at 3, 2)
    jitter = {"Transparency": (-0.25, 0.12), "Threat": (0.25, -0.12)}
    for i, t in enumerate(THEMES_ORDER):
        dx, dy = jitter.get(t, (0.0, 0.0))
        x_frames[i] += dx
        y_topics[i] += dy

    x_max = float(x_frames.max()) + 6
    y_max = max(K_TOPICS * 0.6, float(y_topics.max()) + 8)

    # True proportional-representation line: 1 frame = 1 topic (y = x)
    ref_x = np.linspace(0, min(x_max, K_TOPICS), 50)
    ax_e.plot(ref_x, ref_x, color="#666666", linestyle=(0, (4, 3)),
              linewidth=1.3, alpha=0.80, zorder=1,
              label="Proportional representation (1 frame = 1 topic)")

    # Label-placement offsets (in points) per theme to avoid overlap.
    # Transparency and Threat share nearly the same coordinates so they
    # get explicit leader lines via arrowprops.
    offsets = {
        "Ownership": (10, -4),
        "Utility": (10, -18),
        "Transparency": (32, 28),
        "Threat": (58, 4),
        "Compensation": (-14, 10),
    }
    leader_line = {"Transparency", "Threat"}

    for t, x, y in zip(THEMES_ORDER, x_frames, y_topics):
        ax_e.scatter(x, y, s=260, color=THEME_COLORS[t], alpha=0.90,
                     edgecolors="#222222", linewidths=1.0, zorder=4)
        if t in leader_line:
            ax_e.annotate(
                f"{t}\n({consensus[t]:.0f}% consensus)",
                xy=(x, y), xytext=offsets[t], textcoords="offset points",
                fontsize=8.8, fontweight="bold", zorder=5,
                ha="left", va="center",
                color=THEME_COLORS[t],
                arrowprops=dict(arrowstyle="-", color=THEME_COLORS[t],
                                linewidth=1.1, alpha=0.85,
                                shrinkA=0, shrinkB=4),
            )
        else:
            ax_e.annotate(
                f"{t}\n({consensus[t]:.0f}% consensus)",
                xy=(x, y), xytext=offsets[t], textcoords="offset points",
                fontsize=8.8, fontweight="bold", zorder=5,
                ha="right" if offsets[t][0] < 0 else "left",
            )

    # Callout: most extreme case (Ownership: 24 frames -> 1 topic)
    own_x, own_y = frames["Ownership"], topics["Ownership"]
    ax_e.annotate(
        "24 frames compressed\ninto 1 topic",
        xy=(own_x, own_y + 0.3), xytext=(own_x - 10, own_y + 6.5),
        fontsize=9, fontweight="bold", color="#c0392b", ha="center",
        arrowprops=dict(arrowstyle="->", color="#c0392b",
                        linestyle="--", linewidth=1.4, alpha=0.9),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3f3",
                  edgecolor="#c0392b", alpha=0.92, linewidth=0.9),
        zorder=6,
    )

    ax_e.set_xlabel("Unique artist frames (opinion diversity)", fontsize=10)
    ax_e.set_ylabel("Public discourse topics occupied", fontsize=10)
    ax_e.set_xlim(-1, x_max)
    ax_e.set_ylim(-1, y_max)
    ax_e.set_title("E. Opinion Diversity vs Discursive Spread",
                   fontsize=11.5, fontweight="bold")
    ax_e.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax_e.grid(True, alpha=0.2)

    # ---- Panel F: Consensus vs Compression ----
    x_cons = np.array([consensus[t] for t in THEMES_ORDER])
    y_ent = np.array([entropy[t] for t in THEMES_ORDER])

    f_offsets = {
        "Ownership": (28, 6),
        "Utility": (8, 6),
        "Transparency": (8, 6),
        "Threat": (8, 6),
        "Compensation": (8, 6),
    }
    for t, x, y in zip(THEMES_ORDER, x_cons, y_ent):
        ax_f.scatter(x, y, s=260, color=THEME_COLORS[t], alpha=0.88,
                     edgecolors="#222222", linewidths=1.0, zorder=4)
        ax_f.annotate(f"{t}\n(FCR {fcr[t]:g}:1)",
                      xy=(x, y), xytext=f_offsets[t], textcoords="offset points",
                      fontsize=8.8, fontweight="bold", zorder=5)

    # Callout: low consensus + high diversity = maximum compression (Ownership)
    ax_f.annotate(
        "Low consensus, high diversity\n= maximum compression",
        xy=(consensus["Ownership"], entropy["Ownership"]),
        xytext=(consensus["Ownership"] + 18, entropy["Ownership"] + 0.12),
        fontsize=9, fontweight="bold", color="#c0392b", ha="center",
        arrowprops=dict(arrowstyle="->", color="#c0392b",
                        linestyle="--", linewidth=1.4, alpha=0.9),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3f3",
                  edgecolor="#c0392b", alpha=0.92, linewidth=0.9),
        zorder=6,
    )

    ax_f.set_xlabel("% holding most common position (artist consensus)", fontsize=10)
    ax_f.set_ylabel("Distributional entropy in public discourse", fontsize=10)
    ax_f.set_xlim(0, 100)
    ax_f.set_ylim(-0.03, max(y_ent) * 1.35 + 0.05)
    ax_f.set_title("F. Consensus vs Compression",
                   fontsize=11.5, fontweight="bold")
    ax_f.grid(True, alpha=0.2)

    fig.suptitle(
        "Figure 4: Compression Metrics and Artist Consensus by Theme",
        fontsize=13.5, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure(fig, "figure_4_compression_metrics")
    plt.close()
    print("Saved figure_4_compression_metrics to all output dirs")


if __name__ == "__main__":
    main()
