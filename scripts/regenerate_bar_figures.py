"""Regenerate Figures 2, 3, and 4 with final pipeline data.

Figure 2: Artist probe concentration bar chart (H1)
Figure 3: Macro-thematic distribution comparison (H1/H2)
Figure 4: Compression metrics by theme (H3) - NEW entropy-based figure

Side effects: Writes PNG/PDF files. Prints progress.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus_umap import distance_matrix_consensus, run_umap_multi_seed, umap_from_precomputed_distances
from src.clustering import run_kmeans
from src.projection import train_projection_head, project_to_consensus_space
from src.data_loading import load_artist_perspectives, load_public_discourse, load_public_probes
from src.compression_metrics import get_entropy, get_theme_entropy, get_topic_counts

SEEDS = [137,85,127,59,195,243,170,77,186,79,69,42,240,105,199,91,151,82,177,234,46,101,34,175,108,81,176,241,20,53]
ROOT = Path(__file__).parent.parent
EDIT_DIR = ROOT / "When-Algorithms-Meet-Artists-EDIT"
FIG_DIR = ROOT / "jan_2026_manuscript" / "NMS2026" / "figures"

MACRO_THEMES = {
    "Institutions & Markets": [5,6,8,11,12,13,19,20,25,26],
    "Governance & Rights": [0,2,14,21,23],
    "Technical Genealogy": [4,17,18,22,24],
    "Practice & Pedagogy": [7,9,15,16],
    "Philosophy of Creativity": [1,3,10,27],
}
MACRO_COLORS = {
    "Institutions & Markets": "#4C72B0",
    "Governance & Rights": "#DD8452",
    "Technical Genealogy": "#55A868",
    "Practice & Pedagogy": "#C44E52",
    "Philosophy of Creativity": "#8172B3",
}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.linewidth": 0.8})


def main():
    # Load + pipeline
    print("Loading + pipeline...")
    X_pub = np.load(EDIT_DIR / "embeddings" / "Chunks_of_Public_250words_with_25word_overlap_e5_embeddings.npy")
    X_art = np.load(EDIT_DIR / "embeddings" / "Artists_Perspectives_e5_embeddings.npy")
    X_pp = np.load(EDIT_DIR / "embeddings" / "public_probes_e5_embeddings.npy")
    df_art = load_artist_perspectives(ROOT / "data")
    df_pp = load_public_probes(ROOT / "data")
    theme_labels = df_art["question_group"].str.strip().str.lower().values

    ue = run_umap_multi_seed(X_pub, seeds=SEEDS, n_components=8, n_neighbors=27, min_dist=0.1, metric="cosine")
    D = distance_matrix_consensus(ue, metric="euclidean")
    c8d, _ = umap_from_precomputed_distances(D, n_components=8, n_neighbors=27, min_dist=0.1)
    labels_pub, _ = run_kmeans(c8d, n_clusters=28, metric="euclidean")
    proj = train_projection_head(X_pub, c8d, random_state=42)
    art8d = project_to_consensus_space(X_art, proj['model'], proj['scaler_X'], proj['scaler_Y'])
    pp8d = project_to_consensus_space(X_pp, proj['model'], proj['scaler_X'], proj['scaler_Y'])

    nn = NearestNeighbors(n_neighbors=1).fit(c8d)
    _, idx_a = nn.kneighbors(art8d)
    _, idx_p = nn.kneighbors(pp8d)
    art_labels = labels_pub[idx_a.flatten()]
    pp_labels = labels_pub[idx_p.flatten()]

    # ===== FIGURE 2: Artist probe concentration bar chart =====
    print("Figure 2: Bar chart...")
    pub_counts = get_topic_counts(labels_pub, 28)
    art_counts = get_topic_counts(art_labels, 28)
    pp_counts = get_topic_counts(pp_labels, 28)

    # Sort clusters by artist count descending
    order = np.argsort(art_counts)[::-1]
    # Only show clusters with any artist or significant public presence
    show = [c for c in order if art_counts[c] > 0 or pub_counts[c] > 20][:10]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(show))
    w = 0.27

    bars_art = ax.bar(x - w, [art_counts[c] / len(art_labels) * 100 for c in show], w,
                       label=f"Artist Probes (n={len(art_labels)})", color="#55A868", alpha=0.85)
    bars_pp = ax.bar(x, [pp_counts[c] / len(pp_labels) * 100 for c in show], w,
                      label=f"Public Probes (n={len(pp_labels)})", color="#DD8452", alpha=0.85)
    bars_pub = ax.bar(x + w, [pub_counts[c] / len(labels_pub) * 100 for c in show], w,
                       label=f"Public Discourse (n={len(labels_pub)})", color="#4C72B0", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Topic {c}" for c in show], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Percentage of Corpus", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Percentage of Corpus", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    # Annotation box
    top4 = order[:4]
    top4_pct = sum(art_counts[c] for c in top4) / len(art_labels) * 100
    K = 28
    zero_count = sum(1 for c in range(K) if art_counts[c] == 0)
    zero_mass = sum(pub_counts[c] for c in range(K) if art_counts[c] == 0) / len(labels_pub) * 100

    stats_text = f"{top4_pct:.1f}%\nArtist probes in 4 topics"
    ax.text(0.5, -0.22, stats_text, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="center", color="#55A868",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#55A868", alpha=0.9))

    stats2 = f"{zero_count}\nTopics with zero artist mass"
    ax.text(0.15, -0.22, stats2, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="center", color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fce4ec", edgecolor="#e74c3c", alpha=0.9))

    stats3 = f"{zero_mass:.1f}%\nPublic mass in artist-absent topics"
    ax.text(0.85, -0.22, stats3, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="center", color="#4C72B0",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="#4C72B0", alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    fig.savefig(FIG_DIR / "figure_2_artist_probe_concentration.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_2_artist_probe_concentration.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved figure_2")

    # ===== FIGURE 3: Macro-thematic distribution comparison =====
    print("Figure 3: Macro comparison...")
    c2m = {}
    for m, cs in MACRO_THEMES.items():
        for c in cs:
            c2m[c] = m

    pub_macro = {}
    art_macro = {}
    for m in MACRO_THEMES:
        pub_macro[m] = sum(pub_counts[c] for c in MACRO_THEMES[m]) / len(labels_pub) * 100
        art_macro[m] = sum(art_counts[c] for c in MACRO_THEMES[m]) / len(art_labels) * 100

    macros = list(MACRO_THEMES.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Public discourse
    y = np.arange(len(macros))
    for i, m in enumerate(macros):
        ax1.barh(i, pub_macro[m], color=MACRO_COLORS[m], alpha=0.85)
        ax1.text(pub_macro[m] + 0.5, i, f"{pub_macro[m]:.1f}%", va="center", fontsize=9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(macros, fontsize=9)
    ax1.set_xlabel("% of corpus")
    ax1.set_title(f"Public Discourse (n={len(labels_pub)})", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()

    # Artist probes
    for i, m in enumerate(macros):
        ax2.barh(i, art_macro[m], color=MACRO_COLORS[m], alpha=0.85)
        ax2.text(art_macro[m] + 0.5, i, f"{art_macro[m]:.1f}%", va="center", fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels(macros, fontsize=9)
    ax2.set_xlabel("% of corpus")
    ax2.set_title(f"Artist Probes (n={len(art_labels)})", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()

    fig.suptitle("Macro-Thematic Distribution: Public Discourse vs. Artist Concerns",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure_3_macrotheme_distributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_3_macrotheme_distributions.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved figure_3")

    # ===== FIGURE 4: Compression metrics (entropy-based) =====
    print("Figure 4: Compression metrics...")
    themes = ["transparency", "ownership", "threat", "compensation", "utility"]
    theme_display = {"transparency": "Transparency", "ownership": "Ownership", "threat": "Threat",
                     "compensation": "Compensation", "utility": "Utility"}
    theme_colors_h3 = {"transparency": "#9b59b6", "ownership": "#3498db", "threat": "#e74c3c",
                        "compensation": "#f39c12", "utility": "#2ecc71"}

    entropies = []
    n_topics = []
    frame_counts = []
    for t in themes:
        ent = get_theme_entropy(art_labels, theme_labels, t)
        entropies.append(ent["entropy_normalized"])
        n_topics.append(ent["n_occupied_clusters"])
        mask = theme_labels == t
        texts = df_art["perspective_text"].values[mask]
        frame_counts.append(len(set(texts)))

    pub_ent = get_entropy(labels_pub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Entropy
    y = np.arange(len(themes))
    colors = [theme_colors_h3[t] for t in themes]
    bars = ax1.barh(y, entropies, color=colors, alpha=0.85, height=0.6)
    ax1.axvline(x=pub_ent["entropy_normalized"], color="#333333", linewidth=1.2, linestyle="--",
                alpha=0.6, label=f"Public discourse ({pub_ent['entropy_normalized']:.3f})")
    for i, (e, fc) in enumerate(zip(entropies, frame_counts)):
        ax1.text(e + 0.01, i, f"{e:.3f}  ({fc} frames)", va="center", fontsize=8.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels([theme_display[t] for t in themes], fontsize=10)
    ax1.set_xlabel("Normalized Entropy (0 = single topic, 1 = uniform)", fontsize=10)
    ax1.set_title("A. Distributional Entropy", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_xlim(0, 1.05)
    ax1.invert_yaxis()

    # Panel B: Topic count
    bars2 = ax2.barh(y, n_topics, color=colors, alpha=0.85, height=0.6)
    ax2.axvline(x=28, color="#333333", linewidth=1.2, linestyle="--", alpha=0.6,
                label="Total topics (28)")
    for i, nt in enumerate(n_topics):
        ax2.text(nt + 0.3, i, f"{nt}/28", va="center", fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([theme_display[t] for t in themes], fontsize=10)
    ax2.set_xlabel("Number of Topics with Theme Presence", fontsize=10)
    ax2.set_title("B. Topic Coverage", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_xlim(0, 30)
    ax2.invert_yaxis()

    fig.suptitle("Compression Metrics by Artist Concern Theme", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure_4_compression_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure_4_compression_metrics.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved figure_4")

    # Remove old figure 4 name if different
    old_f4 = FIG_DIR / "figure_4_theme_salience_ratios.png"
    if old_f4.exists():
        old_f4.unlink()
        print("  Removed old figure_4_theme_salience_ratios.png")

    print("\nAll bar figures regenerated.")


if __name__ == "__main__":
    main()
