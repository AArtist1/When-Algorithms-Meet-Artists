"""Consensus UMAP pipeline: multi-seed UMAP with distance-matrix averaging."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def run_umap(
    X: np.ndarray,
    n_components: int,
    metric: str = "cosine",
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> tuple[np.ndarray, object]:
    """Run a single UMAP projection.

    Returns:
        (embedding, fitted_umap_model)
    """
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    X_umap = reducer.fit_transform(X)
    return X_umap, reducer


def run_umap_multi_seed(
    X: np.ndarray,
    seeds: list[int],
    n_components: int = 8,
    n_neighbors: int = 27,
    min_dist: float = 0.1,
    metric: str = "cosine",
    verbose: bool = True,
) -> list[np.ndarray]:
    """Run UMAP multiple times with different random seeds.

    Returns:
        List of arrays, each of shape (n_samples, n_components).
    """
    from umap import UMAP

    embeddings = []
    for seed in seeds:
        if verbose:
            print(f"[UMAP] seed={seed} (n_neighbors={n_neighbors}, min_dist={min_dist})")
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        emb = reducer.fit_transform(X)
        embeddings.append(emb.astype(np.float32))
    return embeddings


def procrustes_align(embeddings: list[np.ndarray]) -> np.ndarray:
    """Align embeddings to the first one using orthogonal Procrustes.

    Args:
        embeddings: List of (n_samples, d) arrays.

    Returns:
        Array of shape (n_seeds, n_samples, d).
    """
    if len(embeddings) == 0:
        raise ValueError("No embeddings provided for alignment.")
    ref = embeddings[0]
    aligned = [ref]

    for i, E in enumerate(embeddings[1:], start=1):
        if E.shape != ref.shape:
            raise ValueError(
                f"FAILED: Embedding {i} shape {E.shape} != reference shape {ref.shape}"
            )
        R, _ = orthogonal_procrustes(E, ref)
        aligned.append(E @ R)

    return np.stack(aligned, axis=0)


def compute_consensus_average(aligned_embeddings: np.ndarray) -> np.ndarray:
    """Compute consensus by averaging aligned embeddings across seeds.

    Args:
        aligned_embeddings: Array of shape (n_seeds, n_samples, d).

    Returns:
        Consensus embedding of shape (n_samples, d).
    """
    return aligned_embeddings.mean(axis=0)


def distance_matrix_consensus(
    umap_embeddings: list[np.ndarray],
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute consensus distance matrix by averaging pairwise distances across seeds.

    This is the recommended consensus method (ARI: 0.71 vs 0.56 for coordinate averaging).

    Args:
        umap_embeddings: List of (n_samples, d) arrays from seeded UMAP runs.
        metric: Distance metric in the low-d UMAP space.

    Returns:
        Dense symmetric distance matrix of shape (n_samples, n_samples).
    """
    if len(umap_embeddings) == 0:
        raise ValueError("FAILED: umap_embeddings list is empty.")

    n = umap_embeddings[0].shape[0]
    D_sum = np.zeros((n, n), dtype=np.float64)

    for E in umap_embeddings:
        if E.shape[0] != n:
            raise ValueError(
                f"FAILED: Embedding has {E.shape[0]} rows, expected {n}"
            )
        D = squareform(pdist(E, metric=metric)).astype(np.float64)
        D_sum += D

    D_avg = D_sum / float(len(umap_embeddings))
    np.fill_diagonal(D_avg, 0.0)
    return D_avg


def umap_from_precomputed_distances(
    D: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int = 42,
) -> tuple[np.ndarray, object]:
    """Run UMAP on a precomputed distance matrix.

    Returns:
        (embedding, fitted_umap_model)
    """
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    emb = reducer.fit_transform(D)
    return emb, reducer


def compute_pairwise_ari(labels_arr: np.ndarray) -> np.ndarray:
    """Compute pairwise ARI between multiple labelings.

    Args:
        labels_arr: Array of shape (n_labelings, n_samples).

    Returns:
        ARI matrix of shape (n_labelings, n_labelings).
    """
    n = labels_arr.shape[0]
    ari_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(labels_arr[i], labels_arr[j])
            ari_mat[i, j] = ari_mat[j, i] = ari
    return ari_mat
