"""Validate probe construction using free-text artist responses from Lovato survey.

Updated April 2026 to use Config A (final pipeline):
  - Prefix embeddings from figures/prefix_grid_search/
  - UMAP: n_components=5, n_neighbors=53, min_dist=0.01
  - Clustering: k=20
  - Free-text embedded with e5-large-v2 + "query: " prefix

Embeds the 39 free-text responses from the Fair_compensation_tax_text field
and compares them to the templated compensation probes. If both land in
the same semantic region, this validates that the probe templates capture
the same territory as organic artist language.

Also identifies the most articulate free-text responses for potential
inclusion as illustrative quotes in the manuscript.

Output:
    figures/free_text_validation/validation_results.csv
    figures/free_text_validation/quote_candidates.csv
    figures/free_text_validation/validation_report.html
    Console output with metrics and candidate quotes.

Side effects:
    Writes CSV and HTML files. Prints to stdout. Loads embedding model.
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus_umap import (
    distance_matrix_consensus,
    run_umap_multi_seed,
    umap_from_precomputed_distances,
)
from src.clustering import run_kmeans
from src.projection import train_projection_head, project_to_consensus_space
from src.embeddings import embed_chunks


# ---------------------------------------------------------------------------
# Config A (final pipeline, April 2026)
# ---------------------------------------------------------------------------

SEEDS: list[int] = [
    137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
    69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
    46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
]

UMAP_N_COMPONENTS: int = 5
UMAP_N_NEIGHBORS: int = 53
UMAP_MIN_DIST: float = 0.01
N_CLUSTERS: int = 20

ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT / "data"
EMBEDDING_DIR: Path = ROOT / "figures" / "prefix_grid_search"
OUTPUT_DIR: Path = ROOT / "figures" / "free_text_validation"

PUBLIC_EMBEDDINGS_PATH: Path = EMBEDDING_DIR / "prefix_embeddings_public.npy"
ARTIST_EMBEDDINGS_PATH: Path = EMBEDDING_DIR / "prefix_embeddings_artist.npy"
LOVATO_PATH: Path = DATA_DIR / "lovato_survey" / "ai_art_surveydata_cleaned.csv"
MIN_TEXT_LENGTH: int = 8


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def get_free_text_responses(lovato_path: Path, min_length: int = 8) -> pd.DataFrame:
    """Load non-trivial free-text responses from the Lovato survey.

    Args:
        lovato_path: Path to the Lovato survey CSV.
        min_length: Minimum character length to keep.

    Returns:
        DataFrame with columns: text, respondent_idx.

    Side effects:
        Reads from disk.
    """
    df = pd.read_csv(lovato_path)
    raw: pd.Series = pd.Series(df["Fair_compensation_tax_text"].dropna())
    mask = raw.str.strip().str.len() >= min_length
    filtered: pd.Series = pd.Series(raw[mask])
    texts = filtered.str.strip().tolist()
    indices = filtered.index.tolist()

    return pd.DataFrame({
        "text": texts,
        "respondent_idx": indices,
    })


def get_compensation_probes(data_dir: Path) -> pd.DataFrame:
    """Load compensation-themed artist probes.

    Returns:
        DataFrame filtered to compensation theme.

    Side effects:
        Reads from disk.
    """
    from src.data_loading import load_artist_perspectives
    df = load_artist_perspectives(data_dir)
    comp: pd.DataFrame = df[df["question_group"].str.strip().str.lower() == "compensation"].copy()  # type: ignore[assignment]
    return comp


def get_cluster_overlap(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Compute the fraction of clusters that overlap between two label sets.

    Returns fraction of unique clusters in A that also appear in B.

    Side effects: None.
    """
    clusters_a = set(int(l) for l in labels_a)
    clusters_b = set(int(l) for l in labels_b)
    if len(clusters_a) == 0:
        return 0.0
    return len(clusters_a & clusters_b) / len(clusters_a)


def get_validation_report_html(
    df_results: pd.DataFrame,
    df_candidates: pd.DataFrame,
    metrics: dict[str, float | str],
) -> str:
    """Generate an HTML validation report with summary metrics and tables.

    Args:
        df_results: Per-response validation results.
        df_candidates: Quote candidates ranked by quality.
        metrics: Dictionary of validation metric name -> value.

    Returns:
        HTML string for the full report page.

    Side effects: None.
    """
    def _esc(text: str) -> str:
        """Escape text for safe HTML embedding."""
        return html.escape(str(text))

    # Build metric rows
    metric_rows = ""
    for name, value in metrics.items():
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = _esc(str(value))
        metric_rows += f"        <tr><td>{_esc(name)}</td><td>{formatted}</td></tr>\n"

    # Build results rows
    result_rows = ""
    for _, row in df_results.iterrows():
        in_comp = row.get("in_compensation_region", False)
        badge = '<span class="badge yes">Yes</span>' if in_comp else '<span class="badge no">No</span>'
        text_preview = _esc(str(row["text"])[:120])
        if len(str(row["text"])) > 120:
            text_preview += "..."
        result_rows += (
            f'        <tr><td>{int(row["cluster"])}</td>'
            f"<td>{badge}</td>"
            f"<td>{text_preview}</td></tr>\n"
        )

    # Build candidate rows
    candidate_rows = ""
    rank = 0
    for _, row in df_candidates.iterrows():
        if not row.get("in_comp_region", False):
            continue
        rank += 1
        text_full = _esc(str(row["text"]))
        candidate_rows += (
            f'        <tr><td>{rank}</td><td>{int(row["cluster"])}</td>'
            f"<td>{text_full}</td>"
            f'<td>{int(row["text_len"])}</td></tr>\n'
        )
        if rank >= 15:
            break

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Free-Text Validation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
            padding: 2rem;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{
            font-size: 1.6rem;
            color: #1a1a2e;
            border-bottom: 2px solid #1a1a2e;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        h2 {{
            font-size: 1.2rem;
            color: #16213e;
            margin: 2rem 0 0.75rem 0;
        }}
        .config-box {{
            background: #e8eaf6;
            border-left: 4px solid #3f51b5;
            padding: 1rem 1.25rem;
            border-radius: 4px;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }}
        .config-box code {{ font-family: "SF Mono", Consolas, monospace; font-size: 0.85rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #fff;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }}
        th {{
            background: #1a1a2e;
            color: #fff;
            padding: 0.6rem 0.75rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        td {{
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid #eee;
            font-size: 0.85rem;
            vertical-align: top;
        }}
        tr:last-child td {{ border-bottom: none; }}
        tr:nth-child(even) {{ background: #fafbfc; }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge.yes {{ background: #c8e6c9; color: #2e7d32; }}
        .badge.no {{ background: #ffcdd2; color: #c62828; }}
        .footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
            font-size: 0.8rem;
            color: #888;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Free-Text Validation Report</h1>

    <div class="config-box">
        <strong>Pipeline:</strong> Config A (April 2026)<br>
        <strong>UMAP:</strong> <code>n_components={UMAP_N_COMPONENTS}, n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}</code><br>
        <strong>Clustering:</strong> <code>k={N_CLUSTERS}</code><br>
        <strong>Seeds:</strong> {len(SEEDS)} consensus seeds<br>
        <strong>Embeddings:</strong> <code>e5-large-v2</code> with <code>"query: "</code> prefix
    </div>

    <h2>Validation Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
{metric_rows}    </table>

    <h2>Cluster Assignments</h2>
    <table>
        <tr><th>Cluster</th><th>In Compensation Region</th><th>Text</th></tr>
{result_rows}    </table>

    <h2>Quote Candidates (in compensation region, ranked by length)</h2>
    <table>
        <tr><th>Rank</th><th>Cluster</th><th>Text</th><th>Length</th></tr>
{candidate_rows}    </table>

    <div class="footer">
        Generated by <code>scripts/validate_free_text.py</code> | Config A pipeline
    </div>
</div>
</body>
</html>"""
    return report_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run free-text validation against Config A consensus pipeline.

    Side effects:
        Creates output directory. Writes CSV and HTML files. Prints to stdout.
        Loads embedding model via sentence-transformers.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load free-text responses
    print("Loading free-text responses...")
    df_ft = get_free_text_responses(LOVATO_PATH, min_length=MIN_TEXT_LENGTH)
    print(f"  Free-text responses: {len(df_ft)}")

    # Load compensation probes
    print("Loading compensation probes...")
    df_comp = get_compensation_probes(DATA_DIR)
    print(f"  Compensation probes: {len(df_comp)}")
    print(f"  Unique compensation texts: {df_comp['perspective_text'].nunique()}")

    # Embed free-text with "query: " prefix (matching pipeline Config A)
    print("\nEmbedding free-text responses with prefix='query: '...")
    X_ft: np.ndarray = embed_chunks(
        df_ft, text_col="text", model_name="intfloat/e5-large-v2",
        batch_size=32, prefix="query: ",
    )
    print(f"  Shape: {X_ft.shape}")

    # Load precomputed prefix embeddings (Config A)
    print("Loading precomputed prefix embeddings...")
    X_public: np.ndarray = np.load(PUBLIC_EMBEDDINGS_PATH)
    X_artist: np.ndarray = np.load(ARTIST_EMBEDDINGS_PATH)
    print(f"  Public embeddings: {X_public.shape}")
    print(f"  Artist embeddings: {X_artist.shape}")

    # Run consensus UMAP pipeline (Config A: nn=53, md=0.01, nc=5)
    print(
        f"\nRunning consensus UMAP ({len(SEEDS)} seeds, "
        f"nn={UMAP_N_NEIGHBORS}, md={UMAP_MIN_DIST}, nc={UMAP_N_COMPONENTS})..."
    )
    ue: list[np.ndarray] = run_umap_multi_seed(
        X_public, seeds=SEEDS,
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
    )
    D: np.ndarray = distance_matrix_consensus(ue, metric="euclidean")
    umap_result = umap_from_precomputed_distances(
        D,
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
    )
    consensus_nd: np.ndarray = umap_result[0]

    print(f"Clustering (k={N_CLUSTERS}) and projection...")
    kmeans_result = run_kmeans(consensus_nd, n_clusters=N_CLUSTERS, metric="euclidean")
    labels_pub: np.ndarray = kmeans_result[0]
    proj: dict[str, object] = train_projection_head(X_public, consensus_nd, random_state=42)

    # Project all three sets into consensus space
    artist_nd: np.ndarray = project_to_consensus_space(
        X_artist, proj["model"], proj["scaler_X"], proj["scaler_Y"],  # type: ignore[arg-type]
    )
    ft_nd: np.ndarray = project_to_consensus_space(
        X_ft, proj["model"], proj["scaler_X"], proj["scaler_Y"],  # type: ignore[arg-type]
    )

    # Assign clusters via nearest-neighbor lookup
    nn = NearestNeighbors(n_neighbors=1).fit(consensus_nd)
    knn_art = nn.kneighbors(artist_nd)
    knn_ft = nn.kneighbors(ft_nd)
    idx_art: np.ndarray = knn_art[1]
    idx_ft: np.ndarray = knn_ft[1]
    art_labels: np.ndarray = labels_pub[idx_art.flatten()]
    ft_labels: np.ndarray = labels_pub[idx_ft.flatten()]

    # Filter to compensation probes only
    theme_labels: np.ndarray = (
        pd.read_csv(DATA_DIR / "artist_perspectives.csv")["question_group"]
        .str.strip().str.lower().values
    )
    comp_mask: np.ndarray = theme_labels == "compensation"
    comp_labels: np.ndarray = art_labels[comp_mask]
    comp_nd: np.ndarray = artist_nd[comp_mask]

    # ===== VALIDATION METRICS =====
    print(f"\n{'='*60}")
    print("FREE-TEXT VALIDATION RESULTS (Config A: k=20, nn=53, md=0.01, nc=5)")
    print(f"{'='*60}")

    # 1. Cluster overlap
    ft_clusters: set[int] = set(int(l) for l in ft_labels)
    comp_clusters: set[int] = set(int(l) for l in comp_labels)
    overlap: float = get_cluster_overlap(ft_labels, comp_labels)
    print(f"\n1. Cluster overlap:")
    print(f"   Free-text clusters: {sorted(ft_clusters)}")
    print(f"   Compensation probe clusters: {sorted(comp_clusters)}")
    print(f"   Overlap: {overlap*100:.0f}% of free-text clusters also contain compensation probes")

    # 2. Centroid distance
    ft_centroid: np.ndarray = ft_nd.mean(axis=0)
    comp_centroid: np.ndarray = comp_nd.mean(axis=0)
    pub_centroid: np.ndarray = consensus_nd.mean(axis=0)
    dist_ft_comp: float = float(np.linalg.norm(ft_centroid - comp_centroid))
    dist_ft_pub: float = float(np.linalg.norm(ft_centroid - pub_centroid))
    dist_comp_pub: float = float(np.linalg.norm(comp_centroid - pub_centroid))
    print(f"\n2. Centroid distances ({UMAP_N_COMPONENTS}D):")
    print(f"   Free-text to compensation probes: {dist_ft_comp:.3f}")
    print(f"   Free-text to public discourse: {dist_ft_pub:.3f}")
    print(f"   Compensation probes to public discourse: {dist_comp_pub:.3f}")
    print(f"   Ratio (ft-comp / ft-pub): {dist_ft_comp/dist_ft_pub:.3f} (lower = closer to comp than pub)")

    # 3. Mean cosine similarity in embedding space
    cos_ft_comp: float = float(cosine_similarity(X_ft, X_artist[comp_mask]).mean())
    cos_ft_pub: float = float(cosine_similarity(X_ft, X_public).mean())
    print(f"\n3. Mean cosine similarity (embedding space):")
    print(f"   Free-text to compensation probes: {cos_ft_comp:.4f}")
    print(f"   Free-text to public discourse: {cos_ft_pub:.4f}")

    # 4. Per-response cluster assignments
    print(f"\n4. Per-response details:")
    results: list[dict[str, object]] = []
    for i, (_, row) in enumerate(df_ft.iterrows()):
        cluster: int = int(ft_labels[i])
        in_comp: bool = cluster in comp_clusters
        results.append({
            "text": row["text"],
            "cluster": cluster,
            "in_compensation_region": in_comp,
        })
        text_preview: str = str(row["text"])[:80]
        ellipsis: str = "..." if len(str(row["text"])) > 80 else ""
        print(f'   [{cluster:2d}] {"*" if in_comp else " "} "{text_preview}{ellipsis}"')

    df_results: pd.DataFrame = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "validation_results.csv", index=False)
    print(f"\nSaved: validation_results.csv")

    in_region_pct: float = float(df_results["in_compensation_region"].mean() * 100)
    print(f"\n   {in_region_pct:.0f}% of free-text responses land in the same cluster region as compensation probes")

    # ===== QUOTE CANDIDATES =====
    print(f"\n{'='*60}")
    print("QUOTE CANDIDATES FOR MANUSCRIPT")
    print(f"{'='*60}")

    # Rank by text length and whether they're in the compensation region
    df_ft_ranked: pd.DataFrame = df_ft.copy()
    df_ft_ranked["cluster"] = ft_labels
    df_ft_ranked["in_comp_region"] = [int(l) in comp_clusters for l in ft_labels]
    df_ft_ranked["text_len"] = df_ft_ranked["text"].str.len()
    df_ft_ranked = df_ft_ranked.sort_values(["in_comp_region", "text_len"], ascending=[False, False])

    print("\nTop candidates (in compensation region, by length):")
    for i, (_, row) in enumerate(df_ft_ranked[df_ft_ranked["in_comp_region"]].head(10).iterrows()):
        print(f'  {i+1}. "{row["text"]}"')

    df_ft_ranked.to_csv(OUTPUT_DIR / "quote_candidates.csv", index=False)
    print(f"\nSaved: quote_candidates.csv")

    # ===== HTML REPORT =====
    print("\nGenerating HTML validation report...")
    metrics: dict[str, float | str] = {
        "Free-text responses": str(len(df_ft)),
        "Compensation probes": str(len(df_comp)),
        "Cluster overlap (free-text in comp region)": overlap,
        "In-region percentage": in_region_pct / 100.0,
        "Centroid dist: free-text to comp probes": dist_ft_comp,
        "Centroid dist: free-text to public discourse": dist_ft_pub,
        "Centroid dist: comp probes to public discourse": dist_comp_pub,
        "Centroid ratio (ft-comp / ft-pub)": dist_ft_comp / dist_ft_pub,
        "Cosine sim: free-text to comp probes": cos_ft_comp,
        "Cosine sim: free-text to public discourse": cos_ft_pub,
    }
    report_html: str = get_validation_report_html(df_results, df_ft_ranked, metrics)
    report_path: Path = OUTPUT_DIR / "validation_report.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(f"Saved: {report_path.relative_to(ROOT)}")

    elapsed: float = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
