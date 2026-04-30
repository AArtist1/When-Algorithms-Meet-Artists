"""Post-clustering quality assessment and topic validation.

Implements a multi-criteria filter to distinguish genuine discourse topics
from document-specific clusters (single-article artifacts).

Criteria for a valid discourse topic:
    1. Minimum chunk count (default: 10) — enough text for c-TF-IDF
    2. Minimum source documents (default: 2) — cross-document pattern
    3. No single source dominance (optional) — no article > X% of cluster

Clusters that fail are tagged as "document-specific" and reported
separately, NOT silently dropped.

Usage:
    from src.cluster_quality import assess_cluster_quality, TopicQualityReport

    report = assess_cluster_quality(
        labels=labels,
        article_names=article_names,
        texts=texts,
    )
    print(report.summary())
    valid_labels = report.valid_cluster_ids
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClusterInfo:
    """Quality assessment for a single cluster."""

    cluster_id: int
    n_chunks: int
    n_articles: int
    article_counts: dict[str, int]
    top_article_name: str
    top_article_pct: float
    is_valid_topic: bool
    rejection_reasons: list[str]
    avg_word_count: float = 0.0


@dataclass
class TopicQualityReport:
    """Full quality report for a clustering solution."""

    k: int
    n_chunks: int
    n_articles_total: int
    clusters: list[ClusterInfo]
    min_chunk_threshold: int
    min_article_threshold: int
    max_single_source_pct: float

    @property
    def valid_clusters(self) -> list[ClusterInfo]:
        return [c for c in self.clusters if c.is_valid_topic]

    @property
    def invalid_clusters(self) -> list[ClusterInfo]:
        return [c for c in self.clusters if not c.is_valid_topic]

    @property
    def valid_cluster_ids(self) -> list[int]:
        return sorted(c.cluster_id for c in self.valid_clusters)

    @property
    def invalid_cluster_ids(self) -> list[int]:
        return sorted(c.cluster_id for c in self.invalid_clusters)

    @property
    def n_valid(self) -> int:
        return len(self.valid_clusters)

    @property
    def n_invalid(self) -> int:
        return len(self.invalid_clusters)

    @property
    def valid_chunk_count(self) -> int:
        return sum(c.n_chunks for c in self.valid_clusters)

    @property
    def valid_chunk_pct(self) -> float:
        return self.valid_chunk_count / self.n_chunks if self.n_chunks > 0 else 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Topic Quality Report (k={self.k})",
            f"  Criteria: min_chunks={self.min_chunk_threshold}, "
            f"min_articles={self.min_article_threshold}, "
            f"max_single_source={self.max_single_source_pct:.0%}",
            f"  Valid topics: {self.n_valid}/{self.k} "
            f"({self.n_valid/self.k*100:.0f}%)",
            f"  Valid chunks: {self.valid_chunk_count}/{self.n_chunks} "
            f"({self.valid_chunk_pct:.1%})",
            f"  Document-specific clusters: {self.n_invalid}",
        ]
        if self.invalid_clusters:
            lines.append("  Rejected clusters:")
            for c in self.invalid_clusters:
                reasons = ", ".join(c.rejection_reasons)
                lines.append(
                    f"    Cluster {c.cluster_id}: {c.n_chunks} chunks, "
                    f"{c.n_articles} articles — {reasons}"
                )
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for saving/analysis."""
        rows = []
        for c in self.clusters:
            rows.append({
                "cluster_id": c.cluster_id,
                "n_chunks": c.n_chunks,
                "n_articles": c.n_articles,
                "top_article": c.top_article_name,
                "top_article_pct": round(c.top_article_pct, 3),
                "avg_word_count": round(c.avg_word_count, 1),
                "is_valid_topic": c.is_valid_topic,
                "rejection_reasons": "; ".join(c.rejection_reasons) if c.rejection_reasons else "",
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------

def assess_cluster_quality(
    labels: np.ndarray,
    article_names: np.ndarray,
    texts: np.ndarray | list[str] | None = None,
    min_chunks: int = 10,
    min_articles: int = 2,
    max_single_source_pct: float = 1.0,
) -> TopicQualityReport:
    """Assess quality of each cluster in a clustering solution.

    Each cluster is evaluated against three criteria:
        1. Contains at least `min_chunks` text segments
        2. Contains text from at least `min_articles` distinct source documents
        3. No single source document accounts for more than `max_single_source_pct`
           of the cluster (set to 1.0 to disable)

    Clusters failing any criterion are tagged as "document-specific" rather
    than valid discourse topics.

    Args:
        labels: Cluster assignments, shape (n,).
        article_names: Source document identifier for each chunk, shape (n,).
        texts: Optional text content for word count statistics.
        min_chunks: Minimum chunks per cluster for a valid topic.
        min_articles: Minimum distinct source documents per cluster.
        max_single_source_pct: Maximum fraction of cluster from one source
            (1.0 = no limit, 0.5 = no source can be >50%).

    Returns:
        TopicQualityReport with per-cluster assessments.
    """
    if len(labels) != len(article_names):
        raise ValueError(
            f"labels ({len(labels)}) and article_names ({len(article_names)}) "
            f"must have the same length"
        )

    k = len(set(int(l) for l in labels))
    n_chunks = len(labels)
    n_articles_total = len(set(article_names))

    cluster_infos = []
    for cluster_id in sorted(set(int(l) for l in labels)):
        mask = labels == cluster_id
        chunk_count = int(mask.sum())
        cluster_articles = article_names[mask]
        article_counter = Counter(cluster_articles)

        n_articles = len(article_counter)
        top_article, top_count = article_counter.most_common(1)[0]
        top_pct = top_count / chunk_count

        # Word count stats
        avg_wc = 0.0
        if texts is not None:
            cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
            word_counts = [len(str(t).split()) for t in cluster_texts]
            avg_wc = float(np.mean(word_counts)) if word_counts else 0.0

        # Evaluate criteria
        reasons = []
        if chunk_count < min_chunks:
            reasons.append(f"too_few_chunks ({chunk_count}<{min_chunks})")
        if n_articles < min_articles:
            reasons.append(f"single_source ({n_articles} article(s))")
        if top_pct > max_single_source_pct:
            reasons.append(
                f"source_dominated ({top_pct:.0%} from '{top_article[:40]}')"
            )

        is_valid = len(reasons) == 0

        cluster_infos.append(ClusterInfo(
            cluster_id=cluster_id,
            n_chunks=chunk_count,
            n_articles=n_articles,
            article_counts=dict(article_counter),
            top_article_name=str(top_article),
            top_article_pct=top_pct,
            is_valid_topic=is_valid,
            rejection_reasons=reasons,
            avg_word_count=avg_wc,
        ))

    return TopicQualityReport(
        k=k,
        n_chunks=n_chunks,
        n_articles_total=n_articles_total,
        clusters=cluster_infos,
        min_chunk_threshold=min_chunks,
        min_article_threshold=min_articles,
        max_single_source_pct=max_single_source_pct,
    )


# ---------------------------------------------------------------------------
# K selection with quality filter
# ---------------------------------------------------------------------------

def get_best_k_with_quality(
    per_k_results: list[dict],
    article_names: np.ndarray,
    consensus_coords: np.ndarray,
    min_chunks: int = 10,
    min_articles: int = 2,
    max_single_source_pct: float = 1.0,
    min_valid_topic_pct: float = 0.60,
    sil_weight: float = 0.5,
    ari_weight: float = 0.5,
    random_state_kmeans: int = 42,
) -> dict:
    """Select the best k considering both metrics AND cluster quality.

    For each k, clusters the data, evaluates quality (multi-source criterion),
    and only considers k values where at least `min_valid_topic_pct` of clusters
    are valid discourse topics.

    The score combines silhouette, ARI, and the valid topic fraction.

    Args:
        per_k_results: List of dicts from get_config_metrics, one per k.
        article_names: Source document names, shape (n,).
        consensus_coords: Consensus coordinates, shape (n, d).
        min_chunks: Minimum chunks for a valid topic.
        min_articles: Minimum articles for a valid topic.
        max_single_source_pct: Max single-source fraction.
        min_valid_topic_pct: Minimum fraction of clusters that must be valid.
        sil_weight: Weight for silhouette in score.
        ari_weight: Weight for ARI in score.
        random_state_kmeans: Random state for KMeans.

    Returns:
        Dict with best_k, quality report, and ranked table.
    """
    from sklearn.cluster import KMeans

    rows = []
    reports = {}

    for pk in per_k_results:
        k = pk["k"]

        # Cluster at this k
        km = KMeans(n_clusters=k, random_state=random_state_kmeans, n_init="auto")
        labels = km.fit_predict(consensus_coords)

        # Assess quality
        report = assess_cluster_quality(
            labels=labels,
            article_names=article_names,
            min_chunks=min_chunks,
            min_articles=min_articles,
            max_single_source_pct=max_single_source_pct,
        )
        reports[k] = report

        valid_pct = report.n_valid / report.k if report.k > 0 else 0
        valid_chunk_pct = report.valid_chunk_pct

        rows.append({
            "k": k,
            "silhouette_consensus": pk["silhouette_consensus"],
            "mean_ari_seed_consensus": pk["mean_ari_seed_consensus"],
            "min_cluster_size": pk.get("min_cluster_size", 0),
            "n_valid_topics": report.n_valid,
            "n_invalid_topics": report.n_invalid,
            "valid_topic_pct": valid_pct,
            "valid_chunk_pct": valid_chunk_pct,
        })

    df = pd.DataFrame(rows)

    # Filter to k values with enough valid topics
    viable = df[df["valid_topic_pct"] >= min_valid_topic_pct].copy()
    if viable.empty:
        # Relax: find the k with the highest valid_topic_pct
        viable = df.copy()
        print(f"  WARNING: No k has {min_valid_topic_pct:.0%} valid topics. "
              f"Using best available.")

    # Normalize and score
    sil = viable["silhouette_consensus"].values
    ari = viable["mean_ari_seed_consensus"].values
    sil_range = sil.max() - sil.min()
    ari_range = ari.max() - ari.min()

    viable["sil_norm"] = (sil - sil.min()) / (sil_range + 1e-8)
    viable["ari_norm"] = (ari - ari.min()) / (ari_range + 1e-8)
    viable["combined_score"] = (
        sil_weight * viable["sil_norm"] + ari_weight * viable["ari_norm"]
    )

    viable = viable.sort_values("combined_score", ascending=False)
    best_k = int(viable.iloc[0]["k"])
    best_report = reports[best_k]

    return {
        "best_k": best_k,
        "quality_report": best_report,
        "ranked_table": viable.reset_index(drop=True),
        "all_k_table": df,
        "all_reports": reports,
    }
