"""Regenerate Figures S3 and S4 using the canonical pipeline outputs.

Figure S3 (silhouette_curve) — redraws the existing silhouette_scores.csv
with k=20 highlighted as the selected value (previously k=28 in the stale file).

Figure S4 (h1_sensitivity) — re-computes top-4 concentration and zero-artist
topics across k in {10,15,18,20,22,25,30} using the canonical 5-D consensus
coordinates, the canonical projection head architecture, and canonical seeds.
k=20 is highlighted as the selected value.

Inputs:
    figures/prefix_grid_search/prefix_consensus_coords.npy  (1736 x 5)
    figures/prefix_comparison/prefix_embeddings_public.npy  (1736 x 1024) or
    figures/prefix_grid_search/prefix_embeddings_public.npy
    figures/prefix_comparison/prefix_embeddings_artist.npy  (1259 x 1024)
    figures/k_justification/silhouette_scores.csv           (k vs silhouette)

Outputs (written to both NMS and NatCol figure directories, plus k_justification):
    figure_S3_silhouette_curve.{png,pdf}
    figure_S4_h1_sensitivity.{png,pdf}

Side effects:
    Writes PNG+PDF files. Prints progress to stdout. Trains projection head once.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.clustering import run_kmeans
from src.projection import train_projection_head, project_to_consensus_space

GRID_SEARCH_DIR = ROOT / "figures" / "prefix_grid_search"
PREFIX_COMP_DIR = ROOT / "figures" / "prefix_comparison"
KJUST_DIR = ROOT / "figures" / "k_justification"

OUT_DIRS = [
    KJUST_DIR,
    ROOT / "figures" / "manuscript",
    ROOT / "manuscripts_and_presentations" / "manuscript" / "NMS2026" / "figures",
    ROOT / "manuscripts_and_presentations" / "manuscript" / "NatureCollection2026" / "figures",
]

K_SELECTED = 20
K_SENSITIVITY = [10, 15, 18, 20, 22, 25, 30]
PROJECTION_LAYERS = (1024, 512, 256, 128)
PROJECTION_LR = 0.002


def save_figure(fig: Figure, name: str, out_dirs: list[Path]) -> None:
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")


def make_s3_silhouette_curve() -> None:
    """Redraw silhouette curve with k=20 highlighted."""
    scores = pd.read_csv(KJUST_DIR / "silhouette_scores.csv")
    ks = scores["k"].to_numpy()
    ys = scores["silhouette"].to_numpy()

    # Peak silhouette (independent of selection)
    best_i = int(np.argmax(ys))
    best_k = int(ks[best_i])
    best_score = float(ys[best_i])

    selected_score = float(ys[ks == K_SELECTED][0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.plot(ks, ys, marker="o", color="#4C72B0", label="Silhouette score", linewidth=1.6)
    ax.axvline(
        K_SELECTED, color="#e74c3c", linestyle="--", linewidth=1.5,
        label=f"k={K_SELECTED} (selected, score={selected_score:.3f})",
    )
    ax.axvline(
        best_k, color="#2ca02c", linestyle=":", linewidth=1.2,
        label=f"k={best_k} (peak silhouette, score={best_score:.3f})",
    )
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette score", fontsize=11)
    ax.set_title(
        "Silhouette Analysis for KMeans Cluster Selection",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "figure_S3_silhouette_curve", OUT_DIRS)
    plt.close()
    print(
        f"Figure S3: selected k={K_SELECTED} (score={selected_score:.3f}), "
        f"peak k={best_k} (score={best_score:.3f})"
    )


def compute_h1_at_k(
    c5d: np.ndarray,
    art5d: np.ndarray,
    k: int,
) -> tuple[float, int]:
    """Compute top-4 artist concentration and zero-artist topic count at k.

    Args:
        c5d: Canonical public consensus coordinates, shape (n_pub, 5).
        art5d: Artist probe projections into the same 5-D space, shape (n_art, 5).
        k: Number of KMeans clusters.

    Returns:
        (top4_pct, zero_artist_topics)

    Side effects:
        None.
    """
    labels_pub, _ = run_kmeans(c5d, n_clusters=k, metric="euclidean")
    nn = NearestNeighbors(n_neighbors=1).fit(c5d)
    _, idx = nn.kneighbors(art5d)
    art_labels = labels_pub[idx.flatten()]

    _, counts = np.unique(art_labels, return_counts=True)
    top4 = float(np.sort(counts)[::-1][:4].sum() / art_labels.shape[0])
    zero_topics = k - int(len(np.unique(art_labels)))
    return top4, zero_topics


def make_s4_h1_sensitivity() -> None:
    """Compute and plot H1 sensitivity across k values with canonical data."""
    # Load canonical 5-D consensus coords
    c5d = np.load(GRID_SEARCH_DIR / "prefix_consensus_coords.npy")
    assert c5d.shape == (1736, 5), f"Expected (1736, 5), got {c5d.shape}"

    # Load public and artist embeddings
    pub_path = GRID_SEARCH_DIR / "prefix_embeddings_public.npy"
    if not pub_path.exists():
        pub_path = PREFIX_COMP_DIR / "prefix_embeddings_public.npy"
    X_pub = np.load(pub_path)
    X_art = np.load(PREFIX_COMP_DIR / "prefix_embeddings_artist.npy")
    assert X_pub.shape == (1736, 1024)
    assert X_art.shape == (1259, 1024)

    # Train projection head once (deterministic with random_state=42)
    print("Training projection head (for artist projection into 5-D space)...")
    proj = train_projection_head(
        X_pub, c5d,
        hidden_layer_sizes=PROJECTION_LAYERS,
        learning_rate_init=PROJECTION_LR,
        random_state=42,
    )
    art5d = project_to_consensus_space(
        X_art, proj["model"], proj["scaler_X"], proj["scaler_Y"]
    )
    print(f"  R2={proj['r2_val']:.4f}")

    # Compute sensitivity
    rows = []
    for k in K_SENSITIVITY:
        top4, zero_t = compute_h1_at_k(c5d, art5d, k)
        rows.append({"k": k, "top4_pct": top4, "zero_topics": zero_t, "n_clusters": k})
        print(f"  k={k:<3} top-4={top4*100:5.1f}%  zero-artist={zero_t}")
    df = pd.DataFrame(rows)
    df.to_csv(KJUST_DIR / "h1_sensitivity.csv", index=False)

    # Sanity check: canonical k=20 should match all_metrics.csv (99.9%, 15)
    canonical = df[df["k"] == K_SELECTED].iloc[0]
    expected_top4 = 0.999
    expected_zero = 15
    assert abs(canonical["top4_pct"] - expected_top4) < 0.01, (
        f"k=20 top-4 {canonical['top4_pct']} differs from canonical {expected_top4}"
    )
    assert canonical["zero_topics"] == expected_zero, (
        f"k=20 zero-topics {canonical['zero_topics']} differs from canonical {expected_zero}"
    )
    print(f"  Canonical sanity check PASSED at k={K_SELECTED}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors_a = [
        "#e74c3c" if int(k) == K_SELECTED else "#4C72B0"
        for k in df["k"]
    ]
    bars_a = axes[0].bar(
        np.arange(len(df)), df["top4_pct"] * 100,
        color=colors_a, alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    for bar, val in zip(bars_a, df["top4_pct"] * 100):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, val + 0.6,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    axes[0].set_xticks(np.arange(len(df)))
    axes[0].set_xticklabels([str(int(k)) for k in df["k"]])
    axes[0].set_xlabel("Number of clusters (k)", fontsize=11)
    axes[0].set_ylabel("Top-4 concentration (%)", fontsize=11)
    axes[0].set_ylim(0, 110)
    axes[0].set_title("A. Artist Probe Concentration", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    colors_b = [
        "#e74c3c" if int(k) == K_SELECTED else "#55A868"
        for k in df["k"]
    ]
    bars_b = axes[1].bar(
        np.arange(len(df)), df["zero_topics"],
        color=colors_b, alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    for bar, val in zip(bars_b, df["zero_topics"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, val + 0.3,
            f"{int(val)}", ha="center", va="bottom", fontsize=9,
        )
    axes[1].set_xticks(np.arange(len(df)))
    axes[1].set_xticklabels([str(int(k)) for k in df["k"]])
    axes[1].set_xlabel("Number of clusters (k)", fontsize=11)
    axes[1].set_ylabel("Topics with zero artist probes", fontsize=11)
    axes[1].set_ylim(0, max(df["zero_topics"]) + 4)
    axes[1].set_title("B. Artist-Absent Topics", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Sensitivity of H1 Findings to Cluster Count",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, "figure_S4_h1_sensitivity", OUT_DIRS)
    plt.close()
    print("Figure S4: saved")


def main() -> None:
    print("=" * 60)
    print("Regenerating Figures S3 and S4 from canonical data")
    print("=" * 60)
    make_s3_silhouette_curve()
    print()
    make_s4_h1_sensitivity()
    print()
    print("Done. Written to:")
    for d in OUT_DIRS:
        print(f"  {d}")


if __name__ == "__main__":
    main()
