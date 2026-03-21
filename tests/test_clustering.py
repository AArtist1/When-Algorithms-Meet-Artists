"""Tests for clustering functionality."""

import numpy as np
import pytest

from src.clustering import find_best_k, get_cluster_sizes, run_kmeans, top_ngrams


@pytest.mark.clustering
class TestFindBestK:
    def test_returns_valid_k(self, random_umap_coords):
        best_k, best_score, scores = find_best_k(
            random_umap_coords, k_min=2, k_max=10, metric="euclidean"
        )
        assert 2 <= best_k <= 10, (
            f"FAILED: Expected best_k in [2, 10], got {best_k}"
        )
        assert best_score > -1.0, (
            f"FAILED: Expected positive silhouette, got {best_score}"
        )

    def test_returns_scores_for_all_k(self, random_umap_coords):
        _, _, scores = find_best_k(
            random_umap_coords, k_min=2, k_max=5, metric="euclidean"
        )
        assert len(scores) == 4, (
            f"FAILED: Expected 4 scores (k=2..5), got {len(scores)}"
        )


@pytest.mark.clustering
class TestKMeans:
    def test_correct_label_count(self, random_umap_coords):
        labels, _ = run_kmeans(random_umap_coords, n_clusters=5, metric="euclidean")
        n_unique = len(set(labels))
        assert n_unique == 5, (
            f"FAILED: Expected 5 unique labels, got {n_unique}"
        )

    def test_all_points_assigned(self, random_umap_coords):
        labels, _ = run_kmeans(random_umap_coords, n_clusters=5, metric="euclidean")
        assert len(labels) == len(random_umap_coords), (
            f"FAILED: Expected {len(random_umap_coords)} labels, got {len(labels)}"
        )

    def test_labels_are_integers(self, random_umap_coords):
        labels, _ = run_kmeans(random_umap_coords, n_clusters=3, metric="euclidean")
        assert labels.dtype in (np.int32, np.int64), (
            f"FAILED: Expected integer labels, got dtype {labels.dtype}"
        )


@pytest.mark.clustering
class TestClusterSizes:
    def test_sizes_sum_to_total(self, random_labels):
        sizes = get_cluster_sizes(random_labels)
        total = sum(sizes.values())
        assert total == len(random_labels), (
            f"FAILED: Cluster sizes sum to {total}, expected {len(random_labels)}"
        )


@pytest.mark.clustering
class TestTopNgrams:
    def test_returns_expected_count(self):
        texts = [
            "artificial intelligence is changing art",
            "artists are concerned about AI and copyright",
            "machine learning models generate images",
            "creative workers face new challenges",
            "generative AI transforms artistic practice",
        ] * 10  # Repeat for more robust TF-IDF
        result = top_ngrams(texts, n=2, top_k=5)
        assert len(result) <= 5, (
            f"FAILED: Expected at most 5 ngrams, got {len(result)}"
        )
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result), (
            "FAILED: Expected list of (ngram, score) tuples"
        )
