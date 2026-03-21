"""Clustering (HDBSCAN + KMeans) for the analysis pipeline."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def find_best_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 30,
    metric: str = "cosine",
    random_state: int = 42,
) -> tuple[int, float, dict[int, float]]:
    """Find best K via silhouette score over a range.

    Returns:
        (best_k, best_score, {k: score} dict)
    """
    X_work = l2_normalize(X) if metric == "cosine" else X.copy()

    scores: dict[int, float] = {}
    best_score = -1.0
    best_k = k_min

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_work)
        score = float(silhouette_score(X_work, labels, metric=metric))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, scores


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    metric: str = "cosine",
) -> tuple[np.ndarray, KMeans]:
    """Run KMeans clustering.

    Returns:
        (labels, fitted_kmeans_model)
    """
    X_work = l2_normalize(X) if metric == "cosine" else X.copy()
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_work)
    return labels, km


def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> tuple[np.ndarray, object]:
    """Run HDBSCAN clustering.

    Returns:
        (labels, fitted_hdbscan_model)
        Labels of -1 indicate noise points.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


def cluster_all_seeds(
    aligned_embeddings: np.ndarray,
    n_clusters: int,
    random_state_base: int = 42,
) -> np.ndarray:
    """Cluster each seed-specific embedding with KMeans.

    Args:
        aligned_embeddings: Array of shape (n_seeds, n_samples, d).
        n_clusters: Number of clusters.
        random_state_base: Base random state (incremented per seed).

    Returns:
        Labels array of shape (n_seeds, n_samples).
    """
    n_seeds = aligned_embeddings.shape[0]
    all_labels = []
    for i in range(n_seeds):
        km = KMeans(n_clusters=n_clusters, random_state=random_state_base + i, n_init="auto")
        labels = km.fit_predict(aligned_embeddings[i])
        all_labels.append(labels)
    return np.stack(all_labels, axis=0)


def get_cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Get the number of points in each cluster."""
    return dict(Counter(int(l) for l in labels))


def top_ngrams(
    texts: list[str],
    n: int = 2,
    top_k: int = 15,
    max_features: int = 5000,
) -> list[tuple[str, float]]:
    """Extract top TF-IDF n-grams from a list of texts.

    Returns:
        List of (ngram, score) tuples sorted by score descending.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(n, n),
        max_features=max_features,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf.mean(axis=0)).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(feature_names[i], float(scores[i])) for i in top_indices]
