"""Regenerate Figure 2: side-by-side percentage bars per topic.

For each of 20 topics, shows two bars:
  - Blue: % of public discourse chunks in that topic
  - Red: % of artist probes in that topic

Sorted by artist probe percentage (descending), so the compression
pattern is immediately visible. Uses "T" prefix for topic numbering
(e.g., "T13: AI Art Authenticity and Human Creativity").

Outputs:
    figures/manuscript/figure_2_artist_probe_concentration.{png,pdf}
    jan_2026_manuscript/NMS2026/figures/figure_2_artist_probe_concentration.{png,pdf}
    jan_2026_manuscript/NatureCollection2026/figures/figure_2_artist_probe_concentration.{png,pdf}

Side effects:
    Reads .npy files. Trains MLP. Writes figure files.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.clustering import run_kmeans
from src.projection import train_projection_head, project_to_consensus_space
from sklearn.neighbors import NearestNeighbors

# Config
K = 20
PROJECTION_LAYERS = (1024, 512, 256, 128)
PROJECTION_LR = 0.002
GRID_DIR = ROOT / "figures" / "prefix_grid_search"
PREFIX_DIR = ROOT / "figures" / "prefix_comparison"
OUT_DIRS = [
    ROOT / "figures" / "manuscript",
    ROOT / "jan_2026_manuscript" / "NMS2026" / "figures",
    ROOT / "jan_2026_manuscript" / "NatureCollection2026" / "figures",
]

TOPIC_LABELS = {
    0: "Decentralized Infrastructure for Art Ecosystems",
    1: "AI as Creative Collaborator",
    2: "Machine Learning Art Theory & Practice",
    3: "Personal Reflections on AI and Art",
    4: "Mental Models and Abstract Thought",
    5: "Harold Cohen and AARON Legacy",
    6: "Future of Arts Journalism and Museums",
    7: "Conversational Reflections on Art Practice",
    8: "AI Copyright and Legal Protection",
    9: "Digital Art Exhibition and Display",
    10: "Artist Defense Tools Against AI",
    11: "AI Authorship and Creative Agency",
    12: "Media Coverage and AI Panic",
    13: "AI Art Authenticity and Human Creativity",
    14: "Informal AI Creative Tool Discourse",
    15: "Deep Dream and Neural Network Visualization",
    16: "Artist Reflections on Technology",
    17: "Artist-Centered AI Design & Ethics",
    18: "AI Art Authorship and Copyright Debates",
    19: "Generative Art History and Pioneers",
}


def save_figure(fig: Figure, name: str, out_dirs: list[Path]) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")


def main() -> None:
    print("Loading data...")
    c5d = np.load(GRID_DIR / "prefix_consensus_coords.npy")
    X_pub = np.load(GRID_DIR / "prefix_embeddings_public.npy") \
        if (GRID_DIR / "prefix_embeddings_public.npy").exists() \
        else np.load(PREFIX_DIR / "prefix_embeddings_public.npy")
    X_art = np.load(PREFIX_DIR / "prefix_embeddings_artist.npy")

    labels_pub, _ = run_kmeans(c5d, n_clusters=K, metric="euclidean")
    proj = train_projection_head(
        X_pub, c5d,
        hidden_layer_sizes=PROJECTION_LAYERS,
        learning_rate_init=PROJECTION_LR,
        random_state=42,
    )
    art5d = project_to_consensus_space(
        X_art, proj["model"], proj["scaler_X"], proj["scaler_Y"]
    )
    nn = NearestNeighbors(n_neighbors=1).fit(c5d)
    _, idx = nn.kneighbors(art5d)
    art_labels = labels_pub[idx.flatten()]

    n_pub = len(labels_pub)
    n_art = len(art_labels)

    # Per-topic percentages
    pub_pct = np.array([100 * (labels_pub == k).sum() / n_pub for k in range(K)])
    art_pct = np.array([100 * (art_labels == k).sum() / n_art for k in range(K)])

    # Sort by artist probe percentage descending
    sort_order = np.argsort(art_pct)[::-1]

    # Build labels with T prefix
    ylabels = []
    for k in sort_order:
        name = TOPIC_LABELS.get(k, f"Topic {k}")
        # Truncate long names
        if len(name) > 38:
            name = name[:35] + "..."
        ylabels.append(f"T{k}: {name}")

    y_pos = np.arange(K)
    bar_height = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(11, 9))

    # Public discourse bars (blue, offset up)
    bars_pub = ax.barh(
        y_pos - bar_height / 2,
        pub_pct[sort_order],
        height=bar_height,
        color="#4C72B0",
        alpha=0.75,
        label=f"Public discourse (n={n_pub:,})",
    )

    # Artist probe bars (red, offset down)
    bars_art = ax.barh(
        y_pos + bar_height / 2,
        art_pct[sort_order],
        height=bar_height,
        color="#e74c3c",
        alpha=0.80,
        label=f"Artist probes (n={n_art:,})",
    )

    # Value labels on bars
    for bar, pct_val in zip(bars_pub, pub_pct[sort_order]):
        w = bar.get_width()
        if w > 0.8:
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct_val:.1f}%", va="center", fontsize=7.5, color="#2c5aa0")

    for bar, pct_val in zip(bars_art, art_pct[sort_order]):
        w = bar.get_width()
        if w > 0.05:
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct_val:.1f}%", va="center", fontsize=7.5, color="#c0392b",
                    fontweight="bold")
        elif w == 0:
            ax.text(0.15, bar.get_y() + bar.get_height() / 2,
                    "0%", va="center", fontsize=6.5, color="#999999", fontstyle="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()  # highest artist % at top
    ax.set_xlabel("Percentage of respective corpus (%)", fontsize=11)
    ax.set_xlim(0, max(pub_pct.max(), art_pct.max()) + 5)
    ax.grid(True, axis="x", alpha=0.2)

    # Annotation boxes
    top3_pct = art_pct[sort_order[:3]].sum()
    top4_pct = art_pct[sort_order[:4]].sum()
    zero_topics_n = int((art_pct == 0).sum())
    ax.text(
        0.97, 0.03,
        f"Top 3 topics contain {top3_pct:.1f}% of all artist probes\n"
        f"Top 4 topics contain {top4_pct:.1f}% of all artist probes\n"
        f"{zero_topics_n} topics have 0% artist representation",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3f3",
                  edgecolor="#e74c3c", alpha=0.92, linewidth=0.8),
    )

    ax.legend(fontsize=9.5, loc="center right", bbox_to_anchor=(1.0, 0.78),
              framealpha=0.92)
    ax.set_title(
        "Distribution of Public Discourse and Artist Probes Across 20 Topics",
        fontsize=12, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, "figure_2_artist_probe_concentration", OUT_DIRS)
    plt.close()
    print("Saved figure_2_artist_probe_concentration to all output dirs")

    # Print verification
    print(f"\nVerification:")
    for i, k in enumerate(sort_order[:6]):
        print(f"  {ylabels[i]}: public={pub_pct[k]:.1f}%, artist={art_pct[k]:.1f}%")
    print(f"  Top-4 artist total: {top4_pct:.1f}%")
    print(f"  Zero-artist topics: {(art_pct == 0).sum()}")


if __name__ == "__main__":
    main()
