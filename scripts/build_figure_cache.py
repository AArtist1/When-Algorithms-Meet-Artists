"""Build the canonical figure cache consumed by every figure script.

This runs the expensive non-deterministic pipeline steps (MLP projection
head training + PCA fit) exactly once and saves the outputs to
figures/final_pipeline/figure_cache/. Every figure script loads from this
cache so all figures share identical PCA axes, point coordinates, and
cluster assignments.

Root cause this fixes: previously each figure script trained its own MLP
projection head. Even with random_state=42, MLPRegressor drifts slightly
across runs (different BLAS thread counts, sklearn patch versions). That
drift changed the artist 5D coordinates, which changed the joint PCA fit,
which changed the PC1/PC2 percentages and point positions — Figure 1 was
landing at 66.6/14.2 while Figure S2 was at 66.3/13.2.

Run this once (or whenever the upstream consensus coords change). All
figure scripts downstream consume the saved arrays.

Outputs (figures/final_pipeline/figure_cache/):
    labels_pub.npy            — (1736,) KMeans topic assignment
    pub2d.npy                 — (1736, 2) public discourse in PCA 2D
    art2d.npy                 — (1259, 2) artist probes in PCA 2D
    pp2d.npy                  — (750, 2) embedding-based public probes in PCA 2D
    art5d.npy                 — (1259, 5) artist probes in consensus 5D
    pp5d.npy                  — (750, 5) public probes in consensus 5D
    art_cluster.npy           — (1259,) artist probe → nearest public topic
    pp_cluster.npy            — (750,) public probe → nearest public topic
    theme_labels_art.npy      — (1259,) artist theme labels (question_group)
    theme_labels_pp.npy       — (750,) public-probe theme labels
    var_pct.npy               — (2,) explained variance percentages
    axes.npz                  — xr (2,) and yr (2,) joint axis ranges
    pca_components.npy        — (2, 5) PCA basis for reproducibility
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.clustering import run_kmeans  # noqa: E402
from src.projection import train_projection_head, project_to_consensus_space  # noqa: E402

K = 20
PROJECTION_LAYERS = (1024, 512, 256, 128)
PROJECTION_LR = 0.002
GRID_DIR = ROOT / "figures" / "prefix_grid_search"
PREFIX_DIR = ROOT / "figures" / "prefix_comparison"
FINAL_DIR = ROOT / "figures" / "final_pipeline"
CACHE_DIR = FINAL_DIR / "figure_cache"

ARTIST_CSV = ROOT / "data" / "artist_perspectives.csv"
PUB_PROBES_CSV = ROOT / "data" / "public_probes.csv"

PAD = 0.8


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading consensus coords + embeddings...")
    c5d = np.load(GRID_DIR / "prefix_consensus_coords.npy")
    X_pub = np.load(GRID_DIR / "prefix_embeddings_public.npy")
    X_art = np.load(PREFIX_DIR / "prefix_embeddings_artist.npy")

    X_pp = np.load(GRID_DIR / "prefix_embeddings_probes.npy") \
        if (GRID_DIR / "prefix_embeddings_probes.npy").exists() \
        else np.load(FINAL_DIR / "embedding_probes_embeddings.npy")

    print(f"  c5d: {c5d.shape}  X_pub: {X_pub.shape}  "
          f"X_art: {X_art.shape}  X_pp: {X_pp.shape}")

    print("Clustering (KMeans k=20)...")
    labels_pub, _ = run_kmeans(c5d, n_clusters=K, metric="euclidean")

    print(f"Training MLP projection head {PROJECTION_LAYERS}...")
    proj = train_projection_head(
        X_pub, c5d,
        hidden_layer_sizes=PROJECTION_LAYERS,
        learning_rate_init=PROJECTION_LR,
        random_state=42,
    )
    print(f"  R2 val = {proj['r2_val']:.4f}")

    print("Projecting artist and public probes into 5D consensus space...")
    art5d = project_to_consensus_space(
        X_art, proj["model"], proj["scaler_X"], proj["scaler_Y"]
    )
    pp5d = project_to_consensus_space(
        X_pp, proj["model"], proj["scaler_X"], proj["scaler_Y"]
    )

    print("Fitting 2D PCA on (public + artist) 5D — the H1 comparison space...")
    print("  Public probes get projected THROUGH the fixed basis afterward,")
    print("  so they are plotted in the same axes but do not bend the fit.")
    X_fit = np.vstack([c5d, art5d])
    pca = PCA(n_components=2)
    pca.fit(X_fit)

    pub2d = pca.transform(c5d)
    art2d = pca.transform(art5d)
    pp2d = pca.transform(pp5d)

    var_pct = pca.explained_variance_ratio_ * 100
    print(f"  Variance: PC1={var_pct[0]:.2f}%, PC2={var_pct[1]:.2f}%")

    print("Assigning artist + public probes to nearest public topic (in 5D)...")
    nn = NearestNeighbors(n_neighbors=1).fit(c5d)
    _, idx_art = nn.kneighbors(art5d)
    art_cluster = labels_pub[idx_art.flatten()]
    _, idx_pp = nn.kneighbors(pp5d)
    pp_cluster = labels_pub[idx_pp.flatten()]

    print("Loading theme labels...")
    df_art = pd.read_csv(ARTIST_CSV)
    theme_labels_art = df_art["question_group"].str.strip().str.lower().values
    assert len(theme_labels_art) == len(art5d), \
        f"Artist CSV rows ({len(theme_labels_art)}) != art5d rows ({len(art5d)})"

    df_pp = pd.read_csv(PUB_PROBES_CSV)
    theme_labels_pp = df_pp["theme"].str.strip().str.lower().values
    assert len(theme_labels_pp) == len(pp5d), \
        f"Public probes CSV rows ({len(theme_labels_pp)}) != pp5d rows ({len(pp5d)})"

    all2d = np.vstack([pub2d, art2d, pp2d])
    xr = np.array([float(all2d[:, 0].min() - PAD), float(all2d[:, 0].max() + PAD)])
    yr = np.array([float(all2d[:, 1].min() - PAD), float(all2d[:, 1].max() + PAD)])

    print(f"Saving cache to {CACHE_DIR.relative_to(ROOT)}/")
    np.save(CACHE_DIR / "labels_pub.npy", labels_pub)
    np.save(CACHE_DIR / "pub2d.npy", pub2d)
    np.save(CACHE_DIR / "art2d.npy", art2d)
    np.save(CACHE_DIR / "pp2d.npy", pp2d)
    np.save(CACHE_DIR / "art5d.npy", art5d)
    np.save(CACHE_DIR / "pp5d.npy", pp5d)
    np.save(CACHE_DIR / "art_cluster.npy", art_cluster)
    np.save(CACHE_DIR / "pp_cluster.npy", pp_cluster)
    np.save(CACHE_DIR / "theme_labels_art.npy", theme_labels_art)
    np.save(CACHE_DIR / "theme_labels_pp.npy", theme_labels_pp)
    np.save(CACHE_DIR / "var_pct.npy", var_pct)
    np.save(CACHE_DIR / "pca_components.npy", pca.components_)
    np.savez(CACHE_DIR / "axes.npz", xr=xr, yr=yr)

    print("\nCache build complete.")
    print(f"  PC1={var_pct[0]:.2f}% variance, PC2={var_pct[1]:.2f}% variance")
    print(f"  Combined 2D capture: {var_pct.sum():.2f}%")
    print(f"  xr={xr.tolist()}  yr={yr.tolist()}")


if __name__ == "__main__":
    main()
