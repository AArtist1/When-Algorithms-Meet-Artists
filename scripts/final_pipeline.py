"""Final pipeline: compute all manuscript metrics with prefix embeddings.

Configuration (from 4-stage grid search, Config C selected 2026-04-04):
    - Embeddings: e5-large-v2 with "query: " prefix
    - 30 unique UMAP seeds
    - UMAP: n_neighbors=45, min_dist=0.03, n_components=8
    - Consensus: distance-matrix averaging
    - KMeans k=20 (min cluster size >= 10, 85% valid topics)
    - Selection rationale: highest ARI (0.707) = most reproducible consensus
    - Projection head: (1024, 256), alpha=0.0001
    - Public probes: re-extracted with prefix embeddings (557 probes)
    - Data: 1,742 chunks from 125 articles (public_discourse_clean_chunks.csv)

Outputs all numbers needed for the manuscript, organized by hypothesis.

Output:
    figures/final_pipeline/all_metrics.csv
    figures/final_pipeline/h3_table.csv
    figures/final_pipeline/cluster_sizes.csv
    Console output with every manuscript number.

Side effects:
    Prints to stdout. Writes CSV. Trains MLP.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import run_kmeans
from src.projection import train_projection_head, project_to_consensus_space
from src.data_loading import load_artist_perspectives, load_public_probes
from src.compression_metrics import (
    get_entropy,
    get_theme_entropy,
    get_theme_topic_coverage,
    get_theme_cluster_set,
    get_article_coverage,
    get_frame_counts,
    get_frame_compression_ratio,
    get_cramers_v,
    get_topic_distribution,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [
    137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
    69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
    46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
]

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures" / "final_pipeline"
GRID_SEARCH_DIR = ROOT / "figures" / "prefix_grid_search"
THEMES = ["ownership", "transparency", "compensation", "threat", "utility"]

# Final optimized parameters (Config A, selected 2026-04-05)
# Rationale: after data cleaning (1736 chunks), Config A has highest consensus
# silhouette (0.657), fewest single-article clusters (3), highest valid topic
# rate (85%), and cleanest H3 gradient. See figures/config_comparison/.
K = 20
N_NEIGHBORS = 53
MIN_DIST = 0.01
N_COMPONENTS = 5
PROJECTION_LAYERS = (1024, 512, 256, 128)
PROJECTION_LR = 0.002


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ===== LOAD =====
    print("=" * 70)
    print("FINAL PIPELINE: ALL MANUSCRIPT METRICS")
    print(f"  Config: k={K}, nn={N_NEIGHBORS}, md={MIN_DIST}, nc={N_COMPONENTS}")
    print(f"  Projection: {PROJECTION_LAYERS}")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Embeddings: e5-large-v2 + 'query: ' prefix")
    print("=" * 70)

    # Load prefix embeddings
    X_pub = np.load(GRID_SEARCH_DIR / "prefix_embeddings_public.npy") \
        if (GRID_SEARCH_DIR / "prefix_embeddings_public.npy").exists() \
        else np.load(ROOT / "figures" / "prefix_comparison" / "prefix_embeddings_public.npy")
    X_art = np.load(ROOT / "figures" / "prefix_comparison" / "prefix_embeddings_artist.npy")

    # Public probes: use prefix-extracted probes
    probe_emb_path = GRID_SEARCH_DIR / "prefix_embeddings_probes.npy"
    if not probe_emb_path.exists():
        raise FileNotFoundError(
            f"Prefix probe embeddings not found at {probe_emb_path}. "
            f"Run scripts/extract_public_probes.py first."
        )
    X_pp = np.load(probe_emb_path)

    df_art = load_artist_perspectives(DATA_DIR)
    df_pub = pd.read_csv(DATA_DIR / "public_discourse_clean_chunks.csv")
    df_pp = load_public_probes(DATA_DIR)

    assert len(df_pub) == X_pub.shape[0], (
        f"Chunk count mismatch: {len(df_pub)} rows vs {X_pub.shape[0]} embeddings"
    )

    theme_labels = df_art["question_group"].str.strip().str.lower().values
    article_names = df_pub["article_name"].values

    print(f"Public: {X_pub.shape[0]}, Artist: {X_art.shape[0]}, Public probes: {X_pp.shape[0]}")

    # ===== CONSENSUS UMAP (reuse from grid search) =====
    consensus_path = GRID_SEARCH_DIR / "prefix_consensus_coords.npy"
    if consensus_path.exists():
        print(f"\nLoading consensus coordinates from grid search...")
        c8d = np.load(consensus_path)
        print(f"  Shape: {c8d.shape}")
    else:
        print(f"\nConsensus UMAP ({len(SEEDS)} seeds)...")
        from src.consensus_umap import (
            distance_matrix_consensus, run_umap_multi_seed,
            umap_from_precomputed_distances,
        )
        ue = run_umap_multi_seed(X_pub, seeds=SEEDS, n_components=N_COMPONENTS,
                                  n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric="cosine")
        D = distance_matrix_consensus(ue, metric="euclidean")
        c8d, _ = umap_from_precomputed_distances(D, n_components=N_COMPONENTS,
                                                   n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST)
        np.save(GRID_SEARCH_DIR / "prefix_consensus_coords.npy", c8d)

    # ===== TRUSTWORTHINESS =====
    tw = float(trustworthiness(X_pub, c8d, n_neighbors=15))
    print(f"  Trustworthiness: {tw:.4f}")

    # ===== CLUSTERING =====
    print(f"Clustering (KMeans k={K})...")
    labels_pub, _ = run_kmeans(c8d, n_clusters=K, metric="euclidean")

    # Verify min cluster size
    sizes = Counter(int(l) for l in labels_pub)
    min_cs = min(sizes.values())
    avg_cs = len(labels_pub) / K
    print(f"  Clusters: {len(sizes)}, min size: {min_cs}, avg size: {avg_cs:.1f}")
    if min_cs < 10:
        print(f"  WARNING: Min cluster size {min_cs} < 10 (cluster quality threshold)")
    assert min_cs >= 5, f"FAIL: Min cluster size {min_cs} < 5 (hard floor)"

    # Save cluster sizes
    df_sizes = pd.DataFrame([{"cluster": c, "size": s} for c, s in sorted(sizes.items())])
    df_sizes.to_csv(OUTPUT_DIR / "cluster_sizes.csv", index=False)

    # ===== PROJECTION (OPTIMIZED FOR PREFIX) =====
    print(f"Training projection head {PROJECTION_LAYERS}...")
    proj = train_projection_head(X_pub, c8d,
                                  hidden_layer_sizes=PROJECTION_LAYERS,
                                  learning_rate_init=PROJECTION_LR,
                                  random_state=42)
    r2_val = proj["r2_val"]
    print(f"  R2 (single split): {r2_val:.4f}")

    art8d = project_to_consensus_space(X_art, proj['model'], proj['scaler_X'], proj['scaler_Y'])
    pp8d = project_to_consensus_space(X_pp, proj['model'], proj['scaler_X'], proj['scaler_Y'])

    # ===== CLUSTER ASSIGNMENTS =====
    nn_model = NearestNeighbors(n_neighbors=1).fit(c8d)
    _, idx_a = nn_model.kneighbors(art8d)
    _, idx_p = nn_model.kneighbors(pp8d)
    art_labels = labels_pub[idx_a.flatten()]
    pp_labels = labels_pub[idx_p.flatten()]

    # ===== PCA =====
    X_comb = np.vstack([c8d, art8d])
    pca = PCA(n_components=2)
    pca.fit(X_comb)
    pc1 = pca.explained_variance_ratio_[0]
    pc2 = pca.explained_variance_ratio_[1]

    # ============================================================
    # H1: COMPRESSION
    # ============================================================
    print(f"\n{'='*70}")
    print("H1: COMPRESSION")
    print(f"{'='*70}")

    unique_art, counts_art = np.unique(art_labels, return_counts=True)
    top4_idx = np.argsort(counts_art)[::-1][:4]
    top4_clusters = unique_art[top4_idx]
    top4_pcts = counts_art[top4_idx] / len(art_labels) * 100
    top4_total = counts_art[top4_idx].sum() / len(art_labels)

    zero_topics = K - len(unique_art)
    zero_clusters = set(range(K)) - set(int(c) for c in unique_art)
    zero_mass = sum((labels_pub == c).sum() for c in zero_clusters) / len(labels_pub)

    print(f"  Top-4 concentration: {top4_total*100:.1f}%")
    print(f"  Top-4 clusters: {top4_clusters.tolist()}")
    print(f"  Top-4 percentages: {[f'{p:.1f}%' for p in top4_pcts]}")
    print(f"  Zero-artist topics: {zero_topics}/{K}")
    print(f"  Public mass in zero-artist topics: {zero_mass*100:.1f}%")

    # ============================================================
    # H2: SEMANTIC vs STYLISTIC
    # ============================================================
    print(f"\n{'='*70}")
    print("H2: SEMANTIC VS STYLISTIC")
    print(f"{'='*70}")

    cv_raw = get_cramers_v(labels_pub, art_labels)
    cv_style = get_cramers_v(pp_labels, art_labels)

    dist_pub = get_topic_distribution(labels_pub)
    dist_art = get_topic_distribution(art_labels)
    dist_pp = get_topic_distribution(pp_labels)
    eps = 1e-10
    jsd_raw = float(jensenshannon(dist_pub + eps, dist_art + eps) ** 2)
    jsd_style = float(jensenshannon(dist_pp + eps, dist_art + eps) ** 2)

    cd_raw = float(np.linalg.norm(c8d.mean(axis=0) - art8d.mean(axis=0)))
    cd_style = float(np.linalg.norm(pp8d.mean(axis=0) - art8d.mean(axis=0)))

    def knn_ssr(coords_q, coords_r, k=15):
        combined = np.vstack([coords_q, coords_r])
        source = np.array([0]*len(coords_q) + [1]*len(coords_r))
        _nn = NearestNeighbors(n_neighbors=k+1).fit(combined)
        _, indices = _nn.kneighbors(coords_q)
        return float((source[indices[:, 1:]] == 0).mean())

    knn_art = knn_ssr(art8d, c8d)
    knn_pp = knn_ssr(pp8d, art8d)

    print(f"  RAW (public vs artist):")
    print(f"    Cramer's V: {cv_raw:.3f}")
    print(f"    JSD: {jsd_raw:.3f}")
    print(f"    Centroid distance: {cd_raw:.3f}")
    print(f"    Artist kNN same-source: {knn_art:.3f}")
    print(f"  STYLE-CONTROLLED (public probes vs artist):")
    print(f"    Cramer's V: {cv_style:.3f}")
    print(f"    JSD: {jsd_style:.3f}")
    print(f"    Centroid distance: {cd_style:.3f}")
    print(f"    Public probe kNN same-source: {knn_pp:.3f}")

    # ============================================================
    # H3: DIFFERENTIAL COMPRESSION
    # ============================================================
    print(f"\n{'='*70}")
    print("H3: DIFFERENTIAL COMPRESSION")
    print(f"{'='*70}")

    pub_ent = get_entropy(labels_pub)
    pp_ent = get_entropy(pp_labels)
    art_ent = get_entropy(art_labels)

    print(f"  Corpus entropies (normalized):")
    print(f"    Public discourse: {pub_ent['entropy_normalized']:.3f}")
    print(f"    Public probes: {pp_ent['entropy_normalized']:.3f}")
    print(f"    All artists: {art_ent['entropy_normalized']:.3f}")

    total_articles = len(set(article_names))

    print(f"\n  Per-theme metrics:")
    h3_rows = []
    for theme in THEMES:
        fc = get_frame_counts(theme_labels, np.array(df_art["perspective_text"].values), theme)
        ent = get_theme_entropy(art_labels, theme_labels, theme)
        cov = get_theme_topic_coverage(art_labels, theme_labels, theme, min_count_thresholds=[1, 5])
        cs = get_theme_cluster_set(art_labels, theme_labels, theme)
        ac = get_article_coverage(cs, labels_pub, np.array(article_names))
        fcr = get_frame_compression_ratio(fc["n_frames"], int(ent["n_occupied_clusters"]))

        pub_chunks_in = sum(int((labels_pub == c).sum()) for c in cs)

        row = {
            "theme": theme, "frames": fc["n_frames"], "probes": fc["n_probes"],
            "entropy_norm": ent["entropy_normalized"],
            "topics_any": cov[1], "topics_5plus": cov[5],
            "max_pct": ent["max_cluster_fraction"],
            "articles": ac["n_articles"], "pct_articles": ac["pct_of_total_articles"],
            "chunks": pub_chunks_in, "pct_chunks": pub_chunks_in / len(labels_pub),
            "fcr": fcr,
        }
        h3_rows.append(row)

        print(f"    {theme.upper():14s}  frames={fc['n_frames']:2d}  entropy={ent['entropy_normalized']:.3f}  "
              f"topics={cov[1]}/{K}  max={ent['max_cluster_fraction']*100:.1f}%  "
              f"arts={ac['n_articles']}/{total_articles}  fcr={fcr:.1f}")

    # ============================================================
    # PROJECTION METRICS
    # ============================================================
    print(f"\n{'='*70}")
    print("PROJECTION HEAD METRICS")
    print(f"{'='*70}")
    print(f"  Architecture: {PROJECTION_LAYERS}")
    print(f"  R2 (single split): {r2_val:.4f}")
    print(f"  Trustworthiness: {tw:.4f}")
    print(f"  PCA: PC1={pc1*100:.1f}%, PC2={pc2*100:.1f}%")

    # ============================================================
    # SUMMARY: ALL MANUSCRIPT NUMBERS
    # ============================================================
    print(f"\n{'='*70}")
    print("ALL NUMBERS FOR MANUSCRIPT")
    print(f"{'='*70}")

    numbers = {
        # Config
        "k": str(K),
        "n_neighbors": str(N_NEIGHBORS),
        "min_dist": str(MIN_DIST),
        "n_components": str(N_COMPONENTS),
        "projection_arch": str(PROJECTION_LAYERS),
        "n_seeds": str(len(SEEDS)),
        "embedding_prefix": "query: ",
        # H1
        "h1_top4_pct": f"{top4_total*100:.1f}%",
        "h1_zero_topics": f"{zero_topics}/{K}",
        "h1_zero_mass_pct": f"{zero_mass*100:.1f}%",
        "h1_top4_clusters": str(top4_clusters.tolist()),
        # H2 raw
        "h2_cramers_v_raw": f"{cv_raw:.3f}",
        "h2_jsd_raw": f"{jsd_raw:.3f}",
        "h2_centroid_raw": f"{cd_raw:.3f}",
        "h2_knn_artist": f"{knn_art:.3f}",
        # H2 style
        "h2_cramers_v_style": f"{cv_style:.3f}",
        "h2_jsd_style": f"{jsd_style:.3f}",
        "h2_centroid_style": f"{cd_style:.3f}",
        "h2_knn_probes": f"{knn_pp:.3f}",
        # H3
        "h3_entropy_transparency": f"{h3_rows[1]['entropy_norm']:.3f}",
        "h3_entropy_ownership": f"{h3_rows[0]['entropy_norm']:.3f}",
        "h3_entropy_threat": f"{h3_rows[3]['entropy_norm']:.3f}",
        "h3_entropy_utility": f"{h3_rows[4]['entropy_norm']:.3f}",
        "h3_entropy_compensation": f"{h3_rows[2]['entropy_norm']:.3f}",
        # Projection
        "proj_r2": f"{r2_val:.4f}",
        "trustworthiness": f"{tw:.4f}",
        "pca_pc1": f"{pc1*100:.1f}%",
        "pca_pc2": f"{pc2*100:.1f}%",
        # Counts
        "n_public_chunks": str(len(labels_pub)),
        "n_artist_probes": str(len(art_labels)),
        "n_public_probes": str(len(pp_labels)),
        "n_articles": str(total_articles),
        "min_cluster_size": str(min_cs),
        "avg_cluster_size": f"{avg_cs:.1f}",
    }

    for k_name, v in numbers.items():
        print(f"  {k_name}: {v}")

    # Save
    df_numbers = pd.DataFrame([numbers])
    df_numbers.to_csv(OUTPUT_DIR / "all_metrics.csv", index=False)

    df_h3 = pd.DataFrame(h3_rows)
    df_h3.to_csv(OUTPUT_DIR / "h3_table.csv", index=False)

    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
