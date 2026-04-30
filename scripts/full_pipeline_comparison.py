"""Full pipeline comparison: new (30 seeds + 574 probes) vs manuscript values.

Runs the complete analysis pipeline with the corrected 30 seeds and
574 public probes, then compares every manuscript-reported metric against
the old values. This is the definitive comparison for deciding what
numbers need to be updated in the manuscript.

Metrics compared:
    H1: Top-4 concentration, zero-artist topics, cluster breakdown
    H2: Cramer's V (raw + style-controlled), JSD (raw + style-controlled),
        centroid distances, k-NN same-source rates
    H3: Per-theme salience ratios at 90% coverage
    Figures: PCA variance, projection R-squared

Output:
    Prints detailed comparison table.
    Saves CSV to figures/pipeline_comparison/.

Side effects:
    Prints to stdout. Writes CSV. Trains MLP. Loads embedding model.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus_umap import (
    distance_matrix_consensus,
    run_umap_multi_seed,
    umap_from_precomputed_distances,
)
from src.clustering import run_kmeans
from src.projection import train_projection_head, project_to_consensus_space
from src.data_loading import load_public_discourse, load_artist_perspectives, load_public_probes


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS: list[int] = [
    137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
    69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
    46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
]

ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT / "data"
EDIT_DIR: Path = ROOT / "When-Algorithms-Meet-Artists-EDIT"
OUTPUT_DIR: Path = ROOT / "figures" / "pipeline_comparison"

THEME_ORDER: list[str] = ["ownership", "transparency", "compensation", "threat", "utility"]

# Old manuscript values (from NMS2026 manuscript before this revision)
OLD_VALUES: dict = {
    "top4_concentration": 0.955,
    "zero_artist_topics": 14,
    "cramers_v_raw": 0.750,
    "cramers_v_style": 0.606,
    "jsd_raw": 0.338,
    "jsd_style": 0.200,
    "centroid_dist_raw": 2.034,
    "centroid_dist_style": 1.336,
    "knn_same_source_artist": 0.921,
    "knn_same_source_public_probe": 0.765,
    "r2_val": 0.73,
    "pca_pc1": 0.345,
    "pca_pc2": 0.329,
    "salience_ownership_raw": 6.95,
    "salience_transparency_raw": 6.95,
    "salience_compensation_raw": 3.14,
    "salience_threat_raw": 4.81,
    "salience_utility_raw": 4.81,
    "salience_ownership_style": 3.20,
    "salience_transparency_style": 3.20,
    "salience_compensation_style": 2.62,
    "salience_threat_style": 1.35,
    "salience_utility_style": 1.35,
    "n_public_probes": 379,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_topic_distribution(labels: np.ndarray, n_clusters: int = 22) -> np.ndarray:
    """Get normalized topic distribution over n_clusters.

    Returns array of shape (n_clusters,) summing to 1.

    Side effects: None.
    """
    counts = np.zeros(n_clusters)
    for label in labels:
        if 0 <= int(label) < n_clusters:
            counts[int(label)] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def get_cramers_v(labels_a: np.ndarray, labels_b: np.ndarray, n_clusters: int = 22) -> float:
    """Compute Cramer's V between two sets of cluster labels.

    Side effects: None.
    """
    # Build contingency table: 2 groups x n_clusters
    table = np.zeros((2, n_clusters))
    for l in labels_a:
        if 0 <= int(l) < n_clusters:
            table[0, int(l)] += 1
    for l in labels_b:
        if 0 <= int(l) < n_clusters:
            table[1, int(l)] += 1

    # Remove zero columns
    nonzero = table.sum(axis=0) > 0
    table = table[:, nonzero]

    if table.shape[1] < 2:
        return 0.0

    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    k = min(table.shape) - 1
    if k == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def get_jsd(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    Side effects: None.
    """
    # Add small epsilon to avoid zero divisions
    eps = 1e-10
    a = dist_a + eps
    b = dist_b + eps
    a = a / a.sum()
    b = b / b.sum()
    return float(jensenshannon(a, b) ** 2)  # squared JSD to match manuscript convention


def get_centroid_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Euclidean distance between centroids.

    Side effects: None.
    """
    return float(np.linalg.norm(coords_a.mean(axis=0) - coords_b.mean(axis=0)))


def get_knn_same_source_rate(
    coords_query: np.ndarray,
    coords_ref: np.ndarray,
    k: int = 15,
) -> float:
    """Fraction of k-NN that are same-source for query points.

    Combines query and ref, labels by source, checks how many of
    query's neighbors are also query points.

    Side effects: None.
    """
    combined = np.vstack([coords_query, coords_ref])
    source = np.array([0] * len(coords_query) + [1] * len(coords_ref))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(combined)
    _, indices = nn.kneighbors(coords_query)
    neighbor_sources = source[indices[:, 1:]]  # exclude self
    return float((neighbor_sources == 0).mean())


def get_salience_ratios_at_90(
    artist_labels: np.ndarray,
    comparison_labels: np.ndarray,
    theme_labels: np.ndarray,
    n_artist: int,
    n_comparison: int,
) -> dict[str, float]:
    """Compute salience ratios at 90% coverage threshold per theme.

    Side effects: None.
    """
    ratios = {}
    for theme in THEME_ORDER:
        mask = theme_labels == theme
        theme_artist_labels = artist_labels[mask]

        # Find minimal cluster set capturing 90% of this theme's artist probes
        unique, counts = np.unique(theme_artist_labels, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        cumsum = np.cumsum(counts[sorted_idx])
        threshold = 0.9 * len(theme_artist_labels)
        n_needed = int(np.searchsorted(cumsum, threshold)) + 1
        top_clusters = set(unique[sorted_idx[:n_needed]])

        # Artist fraction in these clusters (over ALL artist probes)
        artist_in_clusters = sum(1 for l in artist_labels if int(l) in top_clusters)
        artist_frac = artist_in_clusters / n_artist

        # Comparison fraction in these clusters
        comp_in_clusters = sum(1 for l in comparison_labels if int(l) in top_clusters)
        comp_frac = comp_in_clusters / n_comparison

        if comp_frac > 0:
            ratios[theme] = artist_frac / comp_frac
        else:
            ratios[theme] = float('inf')

    return ratios


def get_top4_stats(artist_labels: np.ndarray, n_clusters: int = 22) -> dict:
    """Get top-4 cluster concentration and zero-artist-topic stats.

    Side effects: None.
    """
    unique, counts = np.unique(artist_labels, return_counts=True)
    top4_idx = np.argsort(counts)[::-1][:4]
    top4_clusters = unique[top4_idx].tolist()
    top4_pcts = (counts[top4_idx] / len(artist_labels) * 100).tolist()
    top4_total = counts[top4_idx].sum() / len(artist_labels)

    # Zero-artist topics
    all_clusters = set(range(n_clusters))
    present_clusters = set(int(c) for c in unique)
    zero_topics = len(all_clusters - present_clusters)

    return {
        "top4_concentration": float(top4_total),
        "top4_clusters": top4_clusters,
        "top4_pcts": top4_pcts,
        "zero_artist_topics": zero_topics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full pipeline and compare all metrics.

    Side effects: Prints results. Writes CSV. Trains MLP.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ===== LOAD DATA =====
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    X_public = np.load(EDIT_DIR / "embeddings" / "Chunks_of_Public_250words_with_25word_overlap_e5_embeddings.npy")
    X_artist = np.load(EDIT_DIR / "embeddings" / "Artists_Perspectives_e5_embeddings.npy")
    X_probes = np.load(EDIT_DIR / "embeddings" / "public_probes_e5_embeddings.npy")

    df_artist = load_artist_perspectives(DATA_DIR)
    df_probes = load_public_probes(DATA_DIR)

    theme_labels_artist = df_artist["question_group"].str.strip().str.lower().values
    theme_labels_probes = df_probes["theme"].str.strip().str.lower().values

    print(f"Public chunks: {X_public.shape}")
    print(f"Artist probes: {X_artist.shape}")
    print(f"Public probes: {X_probes.shape} (574 expected)")
    assert X_probes.shape[0] == len(df_probes), "Embedding/CSV row mismatch for public probes"

    # ===== CONSENSUS UMAP =====
    print(f"\n{'='*70}")
    print(f"CONSENSUS UMAP (30 seeds)")
    print(f"{'='*70}")

    umap_embeddings = run_umap_multi_seed(
        X_public, seeds=SEEDS, n_components=8,
        n_neighbors=27, min_dist=0.1, metric="cosine",
    )
    D_consensus = distance_matrix_consensus(umap_embeddings, metric="euclidean")
    consensus_8d, _ = umap_from_precomputed_distances(
        D_consensus, n_components=8, n_neighbors=27, min_dist=0.1,
    )

    # ===== CLUSTERING =====
    print(f"\n{'='*70}")
    print("CLUSTERING (KMeans, k=22)")
    print(f"{'='*70}")

    labels_public, _ = run_kmeans(consensus_8d, n_clusters=22, metric="euclidean")
    print(f"Public cluster labels: {len(labels_public)}")

    # ===== PROJECTION =====
    print(f"\n{'='*70}")
    print("PROJECTION HEAD")
    print(f"{'='*70}")

    proj = train_projection_head(X_public, consensus_8d, random_state=42)
    r2_val = proj['r2_val']
    print(f"R-squared (validation): {r2_val:.4f}")

    artist_8d = project_to_consensus_space(X_artist, proj['model'], proj['scaler_X'], proj['scaler_Y'])
    probes_8d = project_to_consensus_space(X_probes, proj['model'], proj['scaler_X'], proj['scaler_Y'])
    print(f"Artist 8D: {artist_8d.shape}")
    print(f"Probes 8D: {probes_8d.shape}")

    # ===== CLUSTER ASSIGNMENTS =====
    print(f"\n{'='*70}")
    print("CLUSTER ASSIGNMENTS")
    print(f"{'='*70}")

    nn = NearestNeighbors(n_neighbors=1).fit(consensus_8d)

    _, idx_artist = nn.kneighbors(artist_8d)
    artist_cluster_labels = labels_public[idx_artist.flatten()]

    _, idx_probes = nn.kneighbors(probes_8d)
    probe_cluster_labels = labels_public[idx_probes.flatten()]

    print(f"Artist cluster labels: {len(artist_cluster_labels)}")
    print(f"Probe cluster labels: {len(probe_cluster_labels)}")

    # ===== PCA =====
    print(f"\n{'='*70}")
    print("PCA VARIANCE")
    print(f"{'='*70}")

    X_combined = np.vstack([consensus_8d, artist_8d])
    pca = PCA(n_components=2)
    pca.fit(X_combined)
    pca_pc1 = float(pca.explained_variance_ratio_[0])
    pca_pc2 = float(pca.explained_variance_ratio_[1])
    print(f"PC1: {pca_pc1*100:.1f}%, PC2: {pca_pc2*100:.1f}%")

    # ===== COMPUTE ALL METRICS =====
    print(f"\n{'='*70}")
    print("COMPUTING ALL METRICS")
    print(f"{'='*70}")

    # --- H1 ---
    h1 = get_top4_stats(artist_cluster_labels)
    print(f"\nH1: Top-4 concentration: {h1['top4_concentration']:.3f}")
    print(f"H1: Top-4 clusters: {h1['top4_clusters']} at {[f'{p:.1f}%' for p in h1['top4_pcts']]}")
    print(f"H1: Zero-artist topics: {h1['zero_artist_topics']}")

    # Zero-topic public mass
    zero_clusters = set(range(22)) - set(int(c) for c in np.unique(artist_cluster_labels))
    zero_mass = sum((labels_public == c).sum() for c in zero_clusters) / len(labels_public)
    print(f"H1: Public mass in zero-artist topics: {zero_mass:.3f}")

    # --- H2: Raw (public vs artist) ---
    dist_public = get_topic_distribution(labels_public)
    dist_artist = get_topic_distribution(artist_cluster_labels)
    dist_probes = get_topic_distribution(probe_cluster_labels)

    cramers_v_raw = get_cramers_v(labels_public, artist_cluster_labels)
    jsd_raw = get_jsd(dist_public, dist_artist)
    centroid_raw = get_centroid_distance(consensus_8d, artist_8d)
    knn_artist = get_knn_same_source_rate(artist_8d, consensus_8d, k=15)

    print(f"\nH2 (raw): Cramer's V = {cramers_v_raw:.3f}")
    print(f"H2 (raw): JSD = {jsd_raw:.3f}")
    print(f"H2 (raw): Centroid distance = {centroid_raw:.3f}")
    print(f"H2 (raw): Artist k-NN same-source = {knn_artist:.3f}")

    # --- H2: Style-controlled (public probes vs artist) ---
    cramers_v_style = get_cramers_v(probe_cluster_labels, artist_cluster_labels)
    jsd_style = get_jsd(dist_probes, dist_artist)
    centroid_style = get_centroid_distance(probes_8d, artist_8d)
    knn_probes = get_knn_same_source_rate(probes_8d, artist_8d, k=15)

    print(f"\nH2 (style): Cramer's V = {cramers_v_style:.3f}")
    print(f"H2 (style): JSD = {jsd_style:.3f}")
    print(f"H2 (style): Centroid distance = {centroid_style:.3f}")
    print(f"H2 (style): Public probe k-NN same-source = {knn_probes:.3f}")

    # --- H3: Salience ratios ---
    sr_raw = get_salience_ratios_at_90(
        artist_cluster_labels, labels_public, theme_labels_artist,
        len(artist_cluster_labels), len(labels_public),
    )
    sr_style = get_salience_ratios_at_90(
        artist_cluster_labels, probe_cluster_labels, theme_labels_artist,
        len(artist_cluster_labels), len(probe_cluster_labels),
    )

    print(f"\nH3 salience ratios (vs public / vs style-controlled):")
    for theme in THEME_ORDER:
        print(f"  {theme:<16} {sr_raw[theme]:>6.2f}x / {sr_style[theme]:>6.2f}x")

    # ===== COMPARISON TABLE =====
    print(f"\n{'='*70}")
    print("COMPARISON: NEW vs OLD MANUSCRIPT VALUES")
    print(f"{'='*70}\n")

    new_values = {
        "n_public_probes": len(df_probes),
        "top4_concentration": h1["top4_concentration"],
        "zero_artist_topics": h1["zero_artist_topics"],
        "cramers_v_raw": cramers_v_raw,
        "cramers_v_style": cramers_v_style,
        "jsd_raw": jsd_raw,
        "jsd_style": jsd_style,
        "centroid_dist_raw": centroid_raw,
        "centroid_dist_style": centroid_style,
        "knn_same_source_artist": knn_artist,
        "knn_same_source_public_probe": knn_probes,
        "r2_val": r2_val,
        "pca_pc1": pca_pc1,
        "pca_pc2": pca_pc2,
        "salience_ownership_raw": sr_raw["ownership"],
        "salience_transparency_raw": sr_raw["transparency"],
        "salience_compensation_raw": sr_raw["compensation"],
        "salience_threat_raw": sr_raw["threat"],
        "salience_utility_raw": sr_raw["utility"],
        "salience_ownership_style": sr_style["ownership"],
        "salience_transparency_style": sr_style["transparency"],
        "salience_compensation_style": sr_style["compensation"],
        "salience_threat_style": sr_style["threat"],
        "salience_utility_style": sr_style["utility"],
    }

    print(f"{'Metric':<35} {'Old':>10} {'New':>10} {'Delta':>10} {'%Change':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    rows = []
    for key in new_values:
        old = OLD_VALUES.get(key, None)
        new = new_values[key]
        if old is not None:
            delta = new - old
            pct = (delta / old * 100) if old != 0 else float('inf')
            print(f"{key:<35} {old:>10.3f} {new:>10.3f} {delta:>+10.3f} {pct:>+9.1f}%")
            rows.append({"metric": key, "old": old, "new": new, "delta": delta, "pct_change": pct})
        else:
            print(f"{key:<35} {'N/A':>10} {new:>10.3f}")
            rows.append({"metric": key, "old": None, "new": new, "delta": None, "pct_change": None})

    # Save comparison CSV
    df_comp = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "full_pipeline_comparison.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ===== VERIFICATION CHECKS =====
    print(f"\n{'='*70}")
    print("VERIFICATION CHECKS")
    print(f"{'='*70}")

    # Verify H1 claim: "95.5% of artist concerns cluster in 4 of 22 topics"
    print(f"\n1. Top-4 concentration: {h1['top4_concentration']*100:.1f}%")
    print(f"   Old claim: 95.5%")
    print(f"   Status: {'CLOSE ENOUGH' if abs(h1['top4_concentration'] - 0.955) < 0.05 else 'NEEDS UPDATE'}")

    # Verify H1 claim: "14 topics contain zero artist perspective"
    print(f"\n2. Zero-artist topics: {h1['zero_artist_topics']}")
    print(f"   Old claim: 14")
    print(f"   Status: {'MATCHES' if h1['zero_artist_topics'] == 14 else 'NEEDS UPDATE'}")

    # Verify H2 claim: "Cramer's V = 0.606 after style control"
    print(f"\n3. Cramer's V (style-controlled): {cramers_v_style:.3f}")
    print(f"   Old claim: 0.606")
    print(f"   Status: {'CLOSE ENOUGH' if abs(cramers_v_style - 0.606) < 0.05 else 'NEEDS UPDATE'}")

    # Verify H2 claim: "centroid distance = 1.336 after style control"
    print(f"\n4. Centroid distance (style): {centroid_style:.3f}")
    print(f"   Old claim: 1.336")
    print(f"   Status: {'CLOSE ENOUGH' if abs(centroid_style - 1.336) < 0.3 else 'NEEDS UPDATE'}")

    # Verify H3 claim: "Governance 6.95x, affective 1.35x after style"
    print(f"\n5. Salience (ownership, style): {sr_style['ownership']:.2f}x")
    print(f"   Old claim: 3.20x")
    own_match = abs(sr_style['ownership'] - 3.20) < 1.0
    print(f"   Status: {'CLOSE ENOUGH' if own_match else 'NEEDS UPDATE'}")

    print(f"\n6. Salience (threat, style): {sr_style['threat']:.2f}x")
    print(f"   Old claim: 1.35x")
    threat_match = abs(sr_style['threat'] - 1.35) < 0.5
    print(f"   Status: {'CLOSE ENOUGH' if threat_match else 'NEEDS UPDATE'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
