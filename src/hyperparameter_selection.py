"""Hyperparameter selection for the consensus UMAP pipeline.

Implements a 3-stage cascading filter for selecting optimal parameters:
    Stage 1: Filter by trustworthiness (neighborhood preservation from
             high-dimensional to low-dimensional space)
    Stage 2: Normalize silhouette and ARI within the trustworthy set
    Stage 3: Select by weighted combination of silhouette (cluster quality)
             and ARI (seed stability)

The rationale for this ordering:
    - Trustworthiness first because it measures whether the low-dimensional
      space faithfully represents the high-dimensional structure. No
      downstream analysis is valid if the projection distorts neighborhoods.
    - Silhouette second because it measures cluster separability. Higher
      silhouette means cleaner topic boundaries.
    - ARI third because it measures whether the same clusters emerge
      across different UMAP seeds. Higher ARI means the clustering is
      not an artifact of a particular random initialization.

Also provides UMAP hyperparameter grid search (n_neighbors, min_dist,
n_components) using distance-matrix consensus and a 2-stage search strategy:
    1. Coarse pass: fewer seeds to narrow the search space
    2. Fine pass: full seed count on top candidates to confirm

All functions follow get_* naming. No hidden side effects unless documented.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score


# ---------------------------------------------------------------------------
# Stage 0: Evaluate a single configuration
# ---------------------------------------------------------------------------

def get_config_metrics(
    X_highdim: np.ndarray,
    consensus_coords: np.ndarray,
    seed_embeddings: list[np.ndarray],
    k_range: list[int],
    n_neighbors_tw: int = 15,
    random_state_kmeans: int = 42,
) -> dict[str, Any]:
    """Evaluate clustering metrics for a set of consensus coordinates across k values.

    Computes trustworthiness, silhouette, and ARI for each k.

    Args:
        X_highdim: Original high-dimensional embeddings, shape (n, d_high).
            Must be L2-normalized if the original metric was cosine.
        consensus_coords: Consensus low-dimensional coordinates, shape (n, d_low).
        seed_embeddings: List of per-seed UMAP embeddings, each shape (n, d_low).
        k_range: List of k values to evaluate.
        n_neighbors_tw: Number of neighbors for trustworthiness computation.
        random_state_kmeans: Random state for KMeans.

    Returns:
        Dict with trustworthiness metrics and per-k results.

    Side effects:
        None.
    """
    # Trustworthiness: does the low-dim space preserve high-dim neighborhoods?
    tw_consensus = float(trustworthiness(X_highdim, consensus_coords,
                                          n_neighbors=n_neighbors_tw))

    tw_seeds = []
    for seed_emb in seed_embeddings:
        tw = float(trustworthiness(X_highdim, seed_emb, n_neighbors=n_neighbors_tw))
        tw_seeds.append(tw)

    # Per-k metrics
    per_k = []
    for k in k_range:
        # Cluster consensus
        km = KMeans(n_clusters=k, random_state=random_state_kmeans, n_init="auto")
        labels_consensus = km.fit_predict(consensus_coords)
        sil_consensus = float(silhouette_score(consensus_coords, labels_consensus,
                                                metric="euclidean"))

        # Cluster each seed embedding
        seed_labels = []
        sil_seeds = []
        for seed_emb in seed_embeddings:
            km_s = KMeans(n_clusters=k, random_state=random_state_kmeans, n_init="auto")
            lab = km_s.fit_predict(seed_emb)
            seed_labels.append(lab)
            sil_seeds.append(float(silhouette_score(seed_emb, lab, metric="euclidean")))

        # ARI: seed vs consensus
        ari_seed_cons = []
        for lab in seed_labels:
            ari_seed_cons.append(float(adjusted_rand_score(lab, labels_consensus)))

        # ARI: pairwise between seeds
        ari_pairs = []
        for (a, b) in combinations(seed_labels, 2):
            ari_pairs.append(float(adjusted_rand_score(a, b)))

        # Min cluster size
        from collections import Counter
        sizes = Counter(int(l) for l in labels_consensus)
        min_cs = min(sizes.values())

        per_k.append({
            "k": k,
            "silhouette_consensus": sil_consensus,
            "silhouette_mean_seeds": float(np.mean(sil_seeds)),
            "mean_ari_seed_consensus": float(np.mean(ari_seed_cons)),
            "std_ari_seed_consensus": float(np.std(ari_seed_cons)),
            "mean_ari_pairs": float(np.mean(ari_pairs)) if ari_pairs else 0.0,
            "min_cluster_size": min_cs,
        })

    return {
        "trustworthiness_consensus": tw_consensus,
        "trustworthiness_mean_seeds": float(np.mean(tw_seeds)),
        "trustworthiness_std_seeds": float(np.std(tw_seeds)),
        "per_k": per_k,
    }


# ---------------------------------------------------------------------------
# Stage 1-3: Select best k using cascading filter
# ---------------------------------------------------------------------------

def get_best_k(
    per_k_results: list[dict],
    tw_consensus: float,
    tw_mean_seeds: float,
    tw_weight_consensus: float = 0.9,
    tw_weight_seeds: float = 0.1,
    sil_weight: float = 0.5,
    ari_weight: float = 0.5,
    min_cluster_size_floor: int = 35,
) -> dict[str, Any]:
    """Select the best k from per-k metrics using weighted combination.

    Since we're evaluating a single (n_neighbors, min_dist) configuration
    (already determined by prior grid search), this function selects
    the best k by combining silhouette and ARI stability.

    K values where any cluster has fewer than min_cluster_size_floor
    observations are excluded. For our corpus of 1742 public discourse
    chunks (250-word natural text, 25-word overlap from 125 articles),
    a floor of 15 ensures each cluster contains text from multiple
    articles for reliable c-TF-IDF labeling (~0.9% of corpus).

    The trustworthiness values are reported but not used for filtering
    (there's only one UMAP configuration to evaluate).

    Args:
        per_k_results: List of dicts from get_config_metrics, one per k.
        tw_consensus: Trustworthiness of the consensus embedding.
        tw_mean_seeds: Mean trustworthiness across seeds.
        tw_weight_consensus: Weight for consensus trustworthiness in reporting.
        tw_weight_seeds: Weight for seed trustworthiness in reporting.
        sil_weight: Weight for silhouette in the combined score.
        ari_weight: Weight for ARI in the combined score.
        min_cluster_size_floor: Exclude k values where any cluster has fewer
            than this many observations. Set to 0 to disable.

    Returns:
        Dict with best_k, its metrics, and the full ranked table.

    Side effects:
        None.
    """
    df = pd.DataFrame(per_k_results)

    # Filter by minimum cluster size if available and requested
    if min_cluster_size_floor > 0 and "min_cluster_size" in df.columns:
        n_before = len(df)
        df = df[df["min_cluster_size"] >= min_cluster_size_floor].reset_index(drop=True)
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} k values with min_cluster_size < "
                  f"{min_cluster_size_floor} ({len(df)} remaining)")
        if df.empty:
            raise ValueError(
                f"No k values have min_cluster_size >= {min_cluster_size_floor}. "
                f"Try lowering min_cluster_size_floor."
            )

    # Normalize silhouette and ARI to [0, 1]
    sil = df["silhouette_consensus"].values
    ari = df["mean_ari_seed_consensus"].values

    sil_range = sil.max() - sil.min()
    ari_range = ari.max() - ari.min()

    df["sil_norm"] = (sil - sil.min()) / (sil_range + 1e-8)
    df["ari_norm"] = (ari - ari.min()) / (ari_range + 1e-8)

    # Combined score
    df["combined_score"] = sil_weight * df["sil_norm"] + ari_weight * df["ari_norm"]

    # Sort and select
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    best = df.iloc[0]

    tw_score = tw_weight_consensus * tw_consensus + tw_weight_seeds * tw_mean_seeds

    return {
        "best_k": int(best["k"]),
        "best_silhouette": float(best["silhouette_consensus"]),
        "best_ari_seed_consensus": float(best["mean_ari_seed_consensus"]),
        "best_combined_score": float(best["combined_score"]),
        "trustworthiness_score": tw_score,
        "trustworthiness_consensus": tw_consensus,
        "trustworthiness_mean_seeds": tw_mean_seeds,
        "ranked_table": df,
    }


# ---------------------------------------------------------------------------
# UMAP Hyperparameter Grid Search
# ---------------------------------------------------------------------------

def get_umap_config_score(
    X_highdim: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    seeds: list[int],
    k_range: list[int],
    metric: str = "cosine",
    n_neighbors_tw: int = 15,
    random_state_kmeans: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate a single UMAP (n_neighbors, min_dist, n_components) configuration.

    Runs multi-seed UMAP, computes distance-matrix consensus, then evaluates
    trustworthiness, silhouette, and ARI across k values.

    Args:
        X_highdim: L2-normalized high-dimensional embeddings, shape (n, d).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        n_components: UMAP n_components (dimensionality of UMAP output).
        seeds: Random seeds for multi-seed UMAP.
        k_range: List of k values to evaluate clustering at.
        metric: Distance metric for UMAP.
        n_neighbors_tw: Number of neighbors for trustworthiness.
        random_state_kmeans: Random state for KMeans.
        verbose: Print per-seed progress.

    Returns:
        Dict with config params, trustworthiness, and per-k metrics.
    """
    from .consensus_umap import (
        distance_matrix_consensus,
        run_umap_multi_seed,
        umap_from_precomputed_distances,
    )

    # Run multi-seed UMAP
    seed_embeddings = run_umap_multi_seed(
        X_highdim,
        seeds=seeds,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        verbose=verbose,
    )

    # Distance-matrix consensus
    D_avg = distance_matrix_consensus(seed_embeddings, metric="euclidean")

    # Embed consensus distances back into coordinates for clustering/evaluation
    consensus_coords, _ = umap_from_precomputed_distances(
        D_avg,
        n_components=n_components,
        n_neighbors=min(n_neighbors, D_avg.shape[0] - 1),
        min_dist=min_dist,
        random_state=42,
    )

    # Trustworthiness
    tw_consensus = float(trustworthiness(X_highdim, consensus_coords,
                                          n_neighbors=n_neighbors_tw))
    tw_seeds = [float(trustworthiness(X_highdim, emb, n_neighbors=n_neighbors_tw))
                for emb in seed_embeddings]

    # Per-k metrics
    per_k = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state_kmeans, n_init="auto")
        labels_consensus = km.fit_predict(consensus_coords)
        sil_consensus = float(silhouette_score(consensus_coords, labels_consensus,
                                                metric="euclidean"))

        # Cluster each seed, compute ARI vs consensus
        ari_seed_cons = []
        for emb in seed_embeddings:
            km_s = KMeans(n_clusters=k, random_state=random_state_kmeans, n_init="auto")
            lab = km_s.fit_predict(emb)
            ari_seed_cons.append(float(adjusted_rand_score(lab, labels_consensus)))

        # Min cluster size
        from collections import Counter
        sizes = Counter(labels_consensus)
        min_cluster_size = min(sizes.values())

        per_k.append({
            "k": k,
            "silhouette_consensus": sil_consensus,
            "mean_ari_seed_consensus": float(np.mean(ari_seed_cons)),
            "std_ari_seed_consensus": float(np.std(ari_seed_cons)),
            "min_cluster_size": min_cluster_size,
        })

    return {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_components": n_components,
        "n_seeds": len(seeds),
        "trustworthiness_consensus": tw_consensus,
        "trustworthiness_mean_seeds": float(np.mean(tw_seeds)),
        "trustworthiness_std_seeds": float(np.std(tw_seeds)),
        "per_k": per_k,
    }


def get_grid_search_results(
    X_highdim: np.ndarray,
    n_neighbors_list: list[int],
    min_dist_list: list[float],
    n_components_list: list[int],
    seeds: list[int],
    k_range: list[int],
    metric: str = "cosine",
    n_neighbors_tw: int = 15,
    random_state_kmeans: int = 42,
    verbose: bool = True,
    output_jsonl: str | None = None,
) -> list[dict[str, Any]]:
    """Run UMAP grid search over (n_neighbors, min_dist, n_components).

    Args:
        X_highdim: L2-normalized high-dimensional embeddings.
        n_neighbors_list: Grid values for n_neighbors.
        min_dist_list: Grid values for min_dist.
        n_components_list: Grid values for n_components.
        seeds: Random seeds for multi-seed UMAP.
        k_range: List of k values to evaluate.
        metric: UMAP distance metric.
        n_neighbors_tw: Neighbors for trustworthiness.
        random_state_kmeans: Random state for KMeans.
        verbose: Print progress.
        output_jsonl: If provided, append each result as a JSON line to this
            file immediately after evaluation. Enables incremental monitoring.

    Returns:
        List of result dicts, one per configuration.
    """
    import json

    total = len(n_neighbors_list) * len(min_dist_list) * len(n_components_list)
    results = []
    idx = 0

    for nn in n_neighbors_list:
        for md in min_dist_list:
            for nc in n_components_list:
                idx += 1
                if verbose:
                    print(f"[{idx}/{total}] n_neighbors={nn}, min_dist={md}, "
                          f"n_components={nc}, seeds={len(seeds)}",
                          flush=True)

                result = get_umap_config_score(
                    X_highdim=X_highdim,
                    n_neighbors=nn,
                    min_dist=md,
                    n_components=nc,
                    seeds=seeds,
                    k_range=k_range,
                    metric=metric,
                    n_neighbors_tw=n_neighbors_tw,
                    random_state_kmeans=random_state_kmeans,
                    verbose=False,
                )
                results.append(result)

                if verbose:
                    tw = result["trustworthiness_consensus"]
                    best_k_entry = max(result["per_k"],
                                       key=lambda d: d["silhouette_consensus"])
                    print(f"  -> TW={tw:.4f}, best_k={best_k_entry['k']}, "
                          f"sil={best_k_entry['silhouette_consensus']:.4f}, "
                          f"ARI={best_k_entry['mean_ari_seed_consensus']:.4f}",
                          flush=True)

                if output_jsonl is not None:
                    with open(output_jsonl, "a") as f:
                        f.write(json.dumps(result, default=str) + "\n")

    return results


def get_grid_results_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten grid search results into a DataFrame with one row per config.

    For each config, selects the best k by silhouette and reports its metrics.

    Args:
        results: Output of get_grid_search_results.

    Returns:
        DataFrame sorted by combined score (descending).
    """
    rows = []
    for r in results:
        best_k_entry = max(r["per_k"], key=lambda d: d["silhouette_consensus"])
        rows.append({
            "n_neighbors": r["n_neighbors"],
            "min_dist": r["min_dist"],
            "n_components": r["n_components"],
            "n_seeds": r["n_seeds"],
            "trustworthiness_consensus": r["trustworthiness_consensus"],
            "trustworthiness_mean_seeds": r["trustworthiness_mean_seeds"],
            "best_k": best_k_entry["k"],
            "best_silhouette": best_k_entry["silhouette_consensus"],
            "best_ari_seed_consensus": best_k_entry["mean_ari_seed_consensus"],
            "best_min_cluster_size": best_k_entry["min_cluster_size"],
        })
    return pd.DataFrame(rows)


def get_best_umap_config(
    df_results: pd.DataFrame,
    tw_quantile: float = 0.5,
    sil_weight: float = 0.5,
    ari_weight: float = 0.5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Select the best UMAP config using trustworthiness filtering + combined score.

    Stage 1: Filter to configs in the top tw_quantile by trustworthiness.
    Stage 2: Rank by weighted combination of normalized silhouette and ARI.

    Args:
        df_results: Output of get_grid_results_df.
        tw_quantile: Keep configs at or above this quantile of trustworthiness.
        sil_weight: Weight for silhouette in combined score.
        ari_weight: Weight for ARI in combined score.

    Returns:
        (best_row, candidates_df) — best config and full candidate table.
    """
    df = df_results.copy()

    # Trustworthiness filtering
    cutoff = df["trustworthiness_consensus"].quantile(tw_quantile)
    candidates = df[df["trustworthiness_consensus"] >= cutoff].copy()

    if candidates.empty:
        raise ValueError("No candidates after trustworthiness filtering.")

    # Normalize silhouette and ARI within candidates
    sil = candidates["best_silhouette"].values
    ari = candidates["best_ari_seed_consensus"].values

    sil_range = sil.max() - sil.min()
    ari_range = ari.max() - ari.min()

    candidates["sil_norm"] = (sil - sil.min()) / (sil_range + 1e-8)
    candidates["ari_norm"] = (ari - ari.min()) / (ari_range + 1e-8)
    candidates["combined_score"] = (sil_weight * candidates["sil_norm"]
                                    + ari_weight * candidates["ari_norm"])

    candidates = candidates.sort_values("combined_score", ascending=False)
    return candidates.iloc[0], candidates.reset_index(drop=True)
