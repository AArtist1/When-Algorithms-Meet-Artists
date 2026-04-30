"""Compression metrics for measuring differential marginalization of stakeholder themes.

This module provides principled alternatives to simple salience ratios for
quantifying how artist concern themes are compressed in public discourse.

Metrics:
    1. Topic coverage: how many topics contain each theme
    2. Article coverage: how many source documents discuss each theme
    3. Shannon entropy: distributional spread across topics
    4. Style-control delta: how metrics change with style-matched comparison
    5. Frame context: frames-per-topic compression ratio

All functions are pure (no side effects) unless documented otherwise.
All functions follow get_* naming convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# ---------------------------------------------------------------------------
# Metric 1: Topic Coverage
# ---------------------------------------------------------------------------

def get_topic_distribution(
    cluster_labels: np.ndarray,
    n_clusters: int = 35,
) -> np.ndarray:
    """Compute normalized distribution of labels across clusters.

    Args:
        cluster_labels: Cluster assignments, shape (n,).
        n_clusters: Total number of clusters.

    Returns:
        Normalized distribution, shape (n_clusters,), summing to 1.
        Zero-count clusters get probability 0.

    Side effects:
        None.
    """
    counts = np.zeros(n_clusters, dtype=float)
    for label in cluster_labels:
        idx = int(label)
        if 0 <= idx < n_clusters:
            counts[idx] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def get_topic_counts(
    cluster_labels: np.ndarray,
    n_clusters: int = 35,
) -> np.ndarray:
    """Compute raw counts per cluster.

    Args:
        cluster_labels: Cluster assignments, shape (n,).
        n_clusters: Total number of clusters.

    Returns:
        Integer counts, shape (n_clusters,).

    Side effects:
        None.
    """
    counts = np.zeros(n_clusters, dtype=int)
    for label in cluster_labels:
        idx = int(label)
        if 0 <= idx < n_clusters:
            counts[idx] += 1
    return counts


def get_topic_coverage(
    cluster_labels: np.ndarray,
    n_clusters: int = 35,
    min_count_thresholds: list[int] | None = None,
) -> dict[int, int]:
    """Count how many topics contain at least N items at various thresholds.

    Args:
        cluster_labels: Cluster assignments, shape (n,).
        n_clusters: Total number of clusters.
        min_count_thresholds: List of minimum count thresholds.
            Defaults to [1, 5, 10].

    Returns:
        Dict mapping threshold to number of topics meeting it.

    Side effects:
        None.
    """
    if min_count_thresholds is None:
        min_count_thresholds = [1, 5, 10]

    counts = get_topic_counts(cluster_labels, n_clusters)
    return {
        thresh: int((counts >= thresh).sum())
        for thresh in min_count_thresholds
    }


def get_theme_topic_coverage(
    cluster_labels: np.ndarray,
    theme_labels: np.ndarray,
    theme: str,
    n_clusters: int = 35,
    min_count_thresholds: list[int] | None = None,
) -> dict[int, int]:
    """Count how many topics contain at least N probes for a specific theme.

    Args:
        cluster_labels: Cluster assignments for all probes, shape (n,).
        theme_labels: Theme label for each probe, shape (n,).
        theme: The theme to compute coverage for.
        n_clusters: Total number of clusters.
        min_count_thresholds: List of thresholds.

    Returns:
        Dict mapping threshold to number of topics meeting it.

    Side effects:
        None.
    """
    mask = theme_labels == theme
    theme_cluster_labels = cluster_labels[mask]
    return get_topic_coverage(theme_cluster_labels, n_clusters, min_count_thresholds)


# ---------------------------------------------------------------------------
# Metric 2: Article Coverage
# ---------------------------------------------------------------------------

def get_article_coverage(
    theme_cluster_set: set[int],
    public_labels: np.ndarray,
    article_names: np.ndarray,
) -> dict[str, int | float]:
    """Count unique articles in the clusters where a theme is present.

    Args:
        theme_cluster_set: Set of cluster IDs capturing the theme's probes.
        public_labels: Cluster labels for public chunks, shape (n_public,).
        article_names: Article name for each public chunk, shape (n_public,).

    Returns:
        Dict with n_articles, n_chunks, and pct_of_total_articles.

    Side effects:
        None.
    """
    mask = np.isin(public_labels, list(theme_cluster_set))
    articles_in_clusters = set(article_names[mask])
    total_articles = len(set(article_names))

    return {
        "n_articles": len(articles_in_clusters),
        "n_chunks": int(mask.sum()),
        "pct_of_total_articles": len(articles_in_clusters) / total_articles if total_articles > 0 else 0.0,
    }


def get_theme_cluster_set(
    cluster_labels: np.ndarray,
    theme_labels: np.ndarray,
    theme: str,
    coverage: float = 0.90,
) -> set[int]:
    """Find the minimal set of clusters capturing a given coverage of a theme.

    Args:
        cluster_labels: Cluster assignments, shape (n,).
        theme_labels: Theme labels, shape (n,).
        theme: Theme to compute for.
        coverage: Fraction of theme probes to capture (default 0.90).

    Returns:
        Set of cluster IDs.

    Side effects:
        None.
    """
    mask = theme_labels == theme
    theme_clusters = cluster_labels[mask]

    if len(theme_clusters) == 0:
        return set()

    unique, counts = np.unique(theme_clusters, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    cumsum = np.cumsum(counts[sorted_idx])
    threshold = coverage * len(theme_clusters)
    n_needed = int(np.searchsorted(cumsum, threshold)) + 1
    return set(int(c) for c in unique[sorted_idx[:n_needed]])


# ---------------------------------------------------------------------------
# Metric 3: Shannon Entropy
# ---------------------------------------------------------------------------

def get_entropy(
    cluster_labels: np.ndarray,
    n_clusters: int = 35,
) -> dict[str, float]:
    """Compute Shannon entropy of a distribution across clusters.

    Args:
        cluster_labels: Cluster assignments, shape (n,).
        n_clusters: Total number of clusters.

    Returns:
        Dict with entropy_bits, entropy_normalized (0-1 scale),
        n_occupied_clusters, and max_cluster_fraction.

    Side effects:
        None.
    """
    dist = get_topic_distribution(cluster_labels, n_clusters)
    nonzero = dist[dist > 0]

    if len(nonzero) == 0:
        return {
            "entropy_bits": 0.0,
            "entropy_normalized": 0.0,
            "n_occupied_clusters": 0,
            "max_cluster_fraction": 0.0,
        }

    entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))
    max_entropy = float(np.log2(n_clusters))
    entropy_normalized = entropy_bits / max_entropy if max_entropy > 0 else 0.0

    return {
        "entropy_bits": entropy_bits,
        "entropy_normalized": entropy_normalized,
        "n_occupied_clusters": int((dist > 0).sum()),
        "max_cluster_fraction": float(dist.max()),
    }


def get_theme_entropy(
    cluster_labels: np.ndarray,
    theme_labels: np.ndarray,
    theme: str,
    n_clusters: int = 35,
) -> dict[str, float]:
    """Compute Shannon entropy for a specific theme's distribution across clusters.

    Args:
        cluster_labels: Cluster assignments for all probes, shape (n,).
        theme_labels: Theme label for each probe, shape (n,).
        theme: The theme to compute entropy for.
        n_clusters: Total number of clusters.

    Returns:
        Dict with entropy metrics plus n_probes for that theme.

    Side effects:
        None.
    """
    mask = theme_labels == theme
    theme_cluster_labels = cluster_labels[mask]
    result = get_entropy(theme_cluster_labels, n_clusters)
    result["n_probes"] = int(mask.sum())
    return result


# ---------------------------------------------------------------------------
# Metric 4: Salience Ratios (kept for comparison / backward compatibility)
# ---------------------------------------------------------------------------

def get_salience_ratio(
    artist_labels: np.ndarray,
    comparison_labels: np.ndarray,
    theme_labels: np.ndarray,
    theme: str,
    coverage: float = 0.90,
) -> dict[str, float]:
    """Compute salience ratio for a theme at a given coverage threshold.

    The salience ratio is (artist fraction in theme clusters) /
    (comparison fraction in same clusters).

    Args:
        artist_labels: Cluster labels for all artist probes.
        comparison_labels: Cluster labels for comparison corpus.
        theme_labels: Theme label for each artist probe.
        theme: Theme to compute for.
        coverage: Coverage threshold for selecting clusters.

    Returns:
        Dict with ratio, artist_fraction, comparison_fraction,
        and the cluster set used.

    Side effects:
        None.
    """
    cluster_set = get_theme_cluster_set(artist_labels, theme_labels, theme, coverage)

    n_artist = len(artist_labels)
    n_comparison = len(comparison_labels)

    artist_in = sum(1 for l in artist_labels if int(l) in cluster_set)
    artist_frac = artist_in / n_artist if n_artist > 0 else 0.0

    comp_in = sum(1 for l in comparison_labels if int(l) in cluster_set)
    comp_frac = comp_in / n_comparison if n_comparison > 0 else 0.0

    ratio = artist_frac / comp_frac if comp_frac > 0 else float('inf')

    return {
        "ratio": ratio,
        "artist_fraction": artist_frac,
        "comparison_fraction": comp_frac,
        "cluster_set": cluster_set,
    }


# ---------------------------------------------------------------------------
# Metric 5: Cramer's V
# ---------------------------------------------------------------------------

def get_cramers_v(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_clusters: int = 35,
) -> float:
    """Compute Cramer's V between two sets of cluster labels.

    Args:
        labels_a: First set of cluster labels.
        labels_b: Second set of cluster labels.
        n_clusters: Total number of clusters.

    Returns:
        Cramer's V statistic (0 to 1).

    Side effects:
        None.
    """
    table = np.zeros((2, n_clusters))
    for l in labels_a:
        idx = int(l)
        if 0 <= idx < n_clusters:
            table[0, idx] += 1
    for l in labels_b:
        idx = int(l)
        if 0 <= idx < n_clusters:
            table[1, idx] += 1

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


# ---------------------------------------------------------------------------
# Metric 6: Frame Context
# ---------------------------------------------------------------------------

def get_frame_counts(
    theme_labels: np.ndarray,
    probe_texts: np.ndarray,
    theme: str,
) -> dict[str, int]:
    """Count unique frames (probe texts) for a theme.

    Args:
        theme_labels: Theme label for each probe, shape (n,).
        probe_texts: Text of each probe, shape (n,).
        theme: Theme to count frames for.

    Returns:
        Dict with n_frames (unique texts) and n_probes (total probes).

    Side effects:
        None.
    """
    mask = theme_labels == theme
    texts = probe_texts[mask]
    return {
        "n_frames": len(set(texts)),
        "n_probes": int(mask.sum()),
    }


def get_frame_compression_ratio(
    n_frames: int,
    n_occupied_topics: int,
) -> float:
    """Compute frames-per-topic compression ratio.

    A ratio > 1 means multiple frames collapse into single topics.
    Higher = more compressed.

    Args:
        n_frames: Number of unique frames for the theme.
        n_occupied_topics: Number of topics containing any probes for the theme.

    Returns:
        Compression ratio (frames / topics).

    Side effects:
        None.
    """
    if n_occupied_topics == 0:
        return float('inf')
    return n_frames / n_occupied_topics


# ---------------------------------------------------------------------------
# Style-Control Delta
# ---------------------------------------------------------------------------

def get_style_control_comparison(
    artist_labels: np.ndarray,
    public_labels: np.ndarray,
    probe_labels: np.ndarray,
    theme_labels: np.ndarray,
    theme: str,
    n_clusters: int = 35,
) -> dict[str, dict[str, float]]:
    """Compute all metrics for a theme against both raw and style-controlled comparisons.

    Args:
        artist_labels: Cluster labels for all artist probes.
        public_labels: Cluster labels for raw public discourse.
        probe_labels: Cluster labels for style-matched public probes.
        theme_labels: Theme label for each artist probe.
        theme: Theme to compute for.
        n_clusters: Total number of clusters.

    Returns:
        Nested dict with raw, style, and delta values for each metric.

    Side effects:
        None.
    """
    # Entropy (independent of comparison, but included for completeness)
    theme_ent = get_theme_entropy(artist_labels, theme_labels, theme, n_clusters)

    # Salience ratios
    sr_raw = get_salience_ratio(artist_labels, public_labels, theme_labels, theme)
    sr_style = get_salience_ratio(artist_labels, probe_labels, theme_labels, theme)

    # Topic coverage
    cov = get_theme_topic_coverage(artist_labels, theme_labels, theme, n_clusters)

    return {
        "entropy_normalized": {
            "value": theme_ent["entropy_normalized"],
        },
        "n_occupied_clusters": {
            "value": theme_ent["n_occupied_clusters"],
        },
        "max_cluster_fraction": {
            "value": theme_ent["max_cluster_fraction"],
        },
        "topic_coverage_1": {
            "value": cov.get(1, 0),
        },
        "topic_coverage_5": {
            "value": cov.get(5, 0),
        },
        "salience_raw": {
            "value": sr_raw["ratio"],
        },
        "salience_style": {
            "value": sr_style["ratio"],
        },
        "salience_delta": {
            "value": sr_style["ratio"] - sr_raw["ratio"],
        },
    }
