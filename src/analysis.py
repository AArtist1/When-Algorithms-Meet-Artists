"""Statistical analysis: salience ratios, divergence metrics, permutation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def safe_probabilities(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert counts to a probability vector with smoothing to avoid zeros."""
    counts = np.asarray(counts, dtype=np.float64) + eps
    return counts / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) with smoothing."""
    p = safe_probabilities(p, eps=eps)
    q = safe_probabilities(q, eps=eps)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (symmetric, bounded in [0, ln(2)])."""
    p = safe_probabilities(p, eps=eps)
    q = safe_probabilities(q, eps=eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def topic_counts(
    df: pd.DataFrame,
    label_col: str,
    all_labels: list,
) -> np.ndarray:
    """Count occurrences of each topic label aligned to a fixed label set."""
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
    counts = np.zeros(len(all_labels), dtype=np.int64)
    for lab in df[label_col].values:
        if lab in label_to_idx:
            counts[label_to_idx[lab]] += 1
    return counts


def centroid_distance(
    df: pd.DataFrame,
    group_col: str,
    coord_cols: list[str],
    group_a: str,
    group_b: str,
) -> float:
    """Compute Euclidean distance between centroids of two groups."""
    mask_a = df[group_col] == group_a
    mask_b = df[group_col] == group_b
    centroid_a = df.loc[mask_a, coord_cols].mean().values
    centroid_b = df.loc[mask_b, coord_cols].mean().values
    return float(np.linalg.norm(centroid_a - centroid_b))


def permutation_test_centroids(
    df: pd.DataFrame,
    group_col: str,
    coord_cols: list[str],
    group_a: str,
    group_b: str,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict:
    """Permutation test for centroid distance between two groups.

    Returns:
        Dictionary with observed_distance, permutation_mean, p_value, n_permutations.
    """
    observed = centroid_distance(df, group_col, coord_cols, group_a, group_b)

    rng = np.random.default_rng(random_state)
    labels = df[group_col].values.copy()
    coords = df[coord_cols].values

    null_distances = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        mask_a = shuffled == group_a
        mask_b = shuffled == group_b
        ca = coords[mask_a].mean(axis=0)
        cb = coords[mask_b].mean(axis=0)
        null_distances[i] = np.linalg.norm(ca - cb)

    p_value = float(np.mean(null_distances >= observed))

    return {
        "observed_distance": observed,
        "permutation_mean": float(null_distances.mean()),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def knn_same_source_rate(
    df: pd.DataFrame,
    group_col: str,
    coord_cols: list[str],
    k: int = 15,
) -> dict:
    """Compute k-NN same-source rate: fraction of each point's k neighbors from the same group.

    Returns:
        Dictionary with observed_rate, expected_rate (by chance), k.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = df[coord_cols].values
    labels = df[group_col].values

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nn.kneighbors(coords)

    same_count = 0
    total = 0
    for i in range(len(labels)):
        neighbors = indices[i, 1:]  # exclude self
        same_count += np.sum(labels[neighbors] == labels[i])
        total += k

    observed_rate = same_count / total

    # Expected rate by chance: proportion of each group squared, summed
    unique, group_counts = np.unique(labels, return_counts=True)
    group_fracs = group_counts / group_counts.sum()
    expected_rate = float(np.sum(group_fracs ** 2))

    return {
        "observed_rate": float(observed_rate),
        "expected_rate": expected_rate,
        "k": k,
    }


def cramers_v(contingency_table: np.ndarray) -> float:
    """Compute Cramér's V from a contingency table."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def compute_salience_ratios(
    artist_counts: np.ndarray,
    public_counts: np.ndarray,
    topic_labels: list,
    theme_topic_map: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Compute salience ratios per topic.

    Salience ratio = (artist_fraction / public_fraction) for each topic.
    Values > 1 indicate artist overconcentration (public under-emphasis).

    Args:
        artist_counts: Array of artist probe counts per topic.
        public_counts: Array of public chunk counts per topic.
        topic_labels: List of topic identifiers.
        theme_topic_map: Optional mapping from theme name to list of topic indices.

    Returns:
        List of dicts with topic_label, artist_frac, public_frac, salience_ratio.
    """
    artist_frac = safe_probabilities(artist_counts)
    public_frac = safe_probabilities(public_counts)

    results = []
    for i, label in enumerate(topic_labels):
        ratio = float(artist_frac[i] / public_frac[i]) if public_frac[i] > 0 else float("inf")
        results.append({
            "topic_label": label,
            "artist_frac": float(artist_frac[i]),
            "public_frac": float(public_frac[i]),
            "salience_ratio": ratio,
        })

    return results


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)
