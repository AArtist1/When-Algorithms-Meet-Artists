"""Tests for compression metrics module.

Tests cover:
    - Topic distribution and coverage
    - Article coverage
    - Shannon entropy (synthetic + edge cases)
    - Salience ratios
    - Frame counts and compression ratios
    - Style-control comparison
    - Real data validation

All tests use deterministic inputs. Real-data tests load actual project data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression_metrics import (
    get_article_coverage,
    get_cramers_v,
    get_entropy,
    get_frame_compression_ratio,
    get_frame_counts,
    get_salience_ratio,
    get_style_control_comparison,
    get_theme_cluster_set,
    get_theme_entropy,
    get_theme_topic_coverage,
    get_topic_counts,
    get_topic_coverage,
    get_topic_distribution,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def simple_labels():
    """100 items: 50 in cluster 0, 30 in cluster 1, 20 in cluster 2."""
    return np.array([0]*50 + [1]*30 + [2]*20)


@pytest.fixture
def uniform_labels():
    """22 items, one per cluster."""
    return np.arange(22)


@pytest.fixture
def all_same_labels():
    """100 items all in cluster 5."""
    return np.full(100, 5)


@pytest.fixture
def real_data():
    """Load real project data for validation tests."""
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    edit_dir = root / "When-Algorithms-Meet-Artists-EDIT"

    if not (data_dir / "artist_perspectives.csv").exists():
        pytest.skip("Real data not available")

    from src.data_loading import load_artist_perspectives, load_public_discourse

    df_artist = load_artist_perspectives(data_dir)
    df_public = load_public_discourse(data_dir)
    X_public = np.load(edit_dir / "embeddings" / "Chunks_of_Public_250words_with_25word_overlap_e5_embeddings.npy")

    return {
        "df_artist": df_artist,
        "df_public": df_public,
        "X_public": X_public,
        "theme_labels": df_artist["question_group"].str.strip().str.lower().values,
        "article_names": df_public["article_name"].values,
    }


# ===================================================================
# Test: get_topic_distribution
# ===================================================================

class TestTopicDistribution:

    def test_simple_distribution(self, simple_labels):
        dist = get_topic_distribution(simple_labels, n_clusters=5)
        assert dist.shape == (5,)
        assert abs(dist.sum() - 1.0) < 1e-10
        assert abs(dist[0] - 0.5) < 1e-10
        assert abs(dist[1] - 0.3) < 1e-10
        assert abs(dist[2] - 0.2) < 1e-10

    def test_empty_input(self):
        dist = get_topic_distribution(np.array([]), n_clusters=5)
        assert dist.shape == (5,)
        assert dist.sum() == 0.0

    def test_uniform(self, uniform_labels):
        dist = get_topic_distribution(uniform_labels, n_clusters=22)
        assert abs(dist.sum() - 1.0) < 1e-10
        expected = 1.0 / 22
        for i in range(22):
            assert abs(dist[i] - expected) < 1e-10

    def test_all_same(self, all_same_labels):
        dist = get_topic_distribution(all_same_labels, n_clusters=22)
        assert dist[5] == 1.0
        assert sum(dist[i] for i in range(22) if i != 5) == 0.0


# ===================================================================
# Test: get_topic_counts
# ===================================================================

class TestTopicCounts:

    def test_simple_counts(self, simple_labels):
        counts = get_topic_counts(simple_labels, n_clusters=5)
        assert counts[0] == 50
        assert counts[1] == 30
        assert counts[2] == 20
        assert counts[3] == 0
        assert counts[4] == 0

    def test_empty(self):
        counts = get_topic_counts(np.array([]), n_clusters=5)
        assert all(c == 0 for c in counts)


# ===================================================================
# Test: get_topic_coverage
# ===================================================================

class TestTopicCoverage:

    def test_all_in_one_cluster(self, all_same_labels):
        cov = get_topic_coverage(all_same_labels, n_clusters=22, min_count_thresholds=[1, 5, 10])
        assert cov[1] == 1
        assert cov[5] == 1
        assert cov[10] == 1

    def test_uniform_spread(self, uniform_labels):
        cov = get_topic_coverage(uniform_labels, n_clusters=22, min_count_thresholds=[1, 5, 10])
        assert cov[1] == 22  # each cluster has 1
        assert cov[5] == 0   # none have 5
        assert cov[10] == 0

    def test_sparse_spread(self):
        labels = np.array([0, 1, 2, 3, 4])  # 5 clusters with 1 each
        cov = get_topic_coverage(labels, n_clusters=22, min_count_thresholds=[1, 5])
        assert cov[1] == 5
        assert cov[5] == 0

    def test_empty(self):
        cov = get_topic_coverage(np.array([]), n_clusters=22, min_count_thresholds=[1])
        assert cov[1] == 0


# ===================================================================
# Test: get_theme_topic_coverage
# ===================================================================

class TestThemeTopicCoverage:

    def test_single_theme(self):
        labels = np.array([0, 0, 1, 1, 2])
        themes = np.array(["a", "a", "a", "b", "b"])
        cov = get_theme_topic_coverage(labels, themes, "a", n_clusters=5, min_count_thresholds=[1, 2])
        assert cov[1] == 2  # theme "a" in clusters 0 and 1
        assert cov[2] == 1  # only cluster 0 has 2+ of theme "a"

    def test_missing_theme(self):
        labels = np.array([0, 1, 2])
        themes = np.array(["a", "a", "a"])
        cov = get_theme_topic_coverage(labels, themes, "b", n_clusters=5, min_count_thresholds=[1])
        assert cov[1] == 0


# ===================================================================
# Test: get_entropy
# ===================================================================

class TestEntropy:

    def test_all_in_one_cluster(self, all_same_labels):
        ent = get_entropy(all_same_labels, n_clusters=22)
        assert ent["entropy_bits"] == 0.0
        assert ent["entropy_normalized"] == 0.0
        assert ent["n_occupied_clusters"] == 1
        assert ent["max_cluster_fraction"] == 1.0

    def test_uniform_across_all(self, uniform_labels):
        ent = get_entropy(uniform_labels, n_clusters=22)
        expected_bits = np.log2(22)
        assert abs(ent["entropy_bits"] - expected_bits) < 1e-10
        assert abs(ent["entropy_normalized"] - 1.0) < 1e-10
        assert ent["n_occupied_clusters"] == 22

    def test_uniform_across_two(self):
        labels = np.array([0]*50 + [1]*50)
        ent = get_entropy(labels, n_clusters=22)
        assert abs(ent["entropy_bits"] - 1.0) < 1e-10  # log2(2) = 1
        assert ent["n_occupied_clusters"] == 2

    def test_known_distribution(self):
        # P = [0.5, 0.25, 0.25] → H = 1.5 bits
        labels = np.array([0]*100 + [1]*50 + [2]*50)
        ent = get_entropy(labels, n_clusters=5)
        assert abs(ent["entropy_bits"] - 1.5) < 1e-10
        assert ent["n_occupied_clusters"] == 3

    def test_single_item(self):
        ent = get_entropy(np.array([3]), n_clusters=22)
        assert ent["entropy_bits"] == 0.0
        assert ent["n_occupied_clusters"] == 1

    def test_empty(self):
        ent = get_entropy(np.array([]), n_clusters=22)
        assert ent["entropy_bits"] == 0.0
        assert ent["n_occupied_clusters"] == 0


# ===================================================================
# Test: get_theme_entropy
# ===================================================================

class TestThemeEntropy:

    def test_basic(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        themes = np.array(["a", "a", "a", "b", "b", "b"])
        ent = get_theme_entropy(labels, themes, "a", n_clusters=5)
        assert ent["n_probes"] == 3
        assert ent["n_occupied_clusters"] == 2  # theme "a" in clusters 0 and 1

    def test_missing_theme(self):
        labels = np.array([0, 1])
        themes = np.array(["a", "a"])
        ent = get_theme_entropy(labels, themes, "b", n_clusters=5)
        assert ent["n_probes"] == 0
        assert ent["entropy_bits"] == 0.0


# ===================================================================
# Test: get_article_coverage
# ===================================================================

class TestArticleCoverage:

    def test_single_cluster(self):
        public_labels = np.array([0, 0, 0, 1, 1])
        articles = np.array(["art1", "art2", "art3", "art4", "art5"])
        cov = get_article_coverage({0}, public_labels, articles)
        assert cov["n_articles"] == 3
        assert cov["n_chunks"] == 3

    def test_multiple_clusters(self):
        public_labels = np.array([0, 0, 1, 1, 2])
        articles = np.array(["art1", "art2", "art1", "art3", "art4"])
        cov = get_article_coverage({0, 1}, public_labels, articles)
        assert cov["n_articles"] == 3  # art1, art2, art3 (art1 deduped)
        assert cov["n_chunks"] == 4

    def test_empty_cluster_set(self):
        public_labels = np.array([0, 1, 2])
        articles = np.array(["a", "b", "c"])
        cov = get_article_coverage(set(), public_labels, articles)
        assert cov["n_articles"] == 0

    def test_pct_of_total(self):
        public_labels = np.array([0, 0, 1, 1])
        articles = np.array(["a", "b", "c", "d"])
        cov = get_article_coverage({0}, public_labels, articles)
        assert abs(cov["pct_of_total_articles"] - 0.5) < 1e-10  # 2/4


# ===================================================================
# Test: get_theme_cluster_set
# ===================================================================

class TestThemeClusterSet:

    def test_single_cluster(self):
        labels = np.array([5]*100)
        themes = np.array(["a"]*100)
        cs = get_theme_cluster_set(labels, themes, "a")
        assert cs == {5}

    def test_two_clusters_90_pct(self):
        # 80 in cluster 0, 20 in cluster 1 → need both for 90%
        labels = np.array([0]*80 + [1]*20)
        themes = np.array(["a"]*100)
        cs = get_theme_cluster_set(labels, themes, "a", coverage=0.90)
        assert 0 in cs  # cluster 0 alone covers 80%, need cluster 1 for 90%
        assert len(cs) == 2

    def test_empty_theme(self):
        labels = np.array([0, 1])
        themes = np.array(["a", "a"])
        cs = get_theme_cluster_set(labels, themes, "b")
        assert cs == set()


# ===================================================================
# Test: get_salience_ratio
# ===================================================================

class TestSalienceRatio:

    def test_equal_distributions(self):
        artist = np.array([0]*50 + [1]*50)
        comparison = np.array([0]*50 + [1]*50)
        themes = np.array(["a"]*100)
        sr = get_salience_ratio(artist, comparison, themes, "a")
        assert abs(sr["ratio"] - 1.0) < 0.1  # approximately equal

    def test_artist_concentrated(self):
        artist = np.array([0]*100)
        comparison = np.array([0]*10 + [1]*90)
        themes = np.array(["a"]*100)
        sr = get_salience_ratio(artist, comparison, themes, "a")
        assert sr["ratio"] > 1.0  # artists overrepresented in cluster 0

    def test_empty_theme(self):
        artist = np.array([0, 1])
        comparison = np.array([0, 1])
        themes = np.array(["a", "a"])
        sr = get_salience_ratio(artist, comparison, themes, "b")
        assert sr["cluster_set"] == set()


# ===================================================================
# Test: get_cramers_v
# ===================================================================

class TestCramersV:

    def test_identical_distributions(self):
        a = np.array([0]*50 + [1]*50)
        b = np.array([0]*50 + [1]*50)
        v = get_cramers_v(a, b, n_clusters=5)
        assert v < 0.05  # should be near 0

    def test_completely_different(self):
        a = np.array([0]*100)
        b = np.array([1]*100)
        v = get_cramers_v(a, b, n_clusters=5)
        assert v > 0.9  # should be near 1


# ===================================================================
# Test: get_frame_counts
# ===================================================================

class TestFrameCounts:

    def test_basic(self):
        themes = np.array(["a", "a", "a", "b", "b"])
        texts = np.array(["t1", "t1", "t2", "t3", "t3"])
        fc = get_frame_counts(themes, texts, "a")
        assert fc["n_frames"] == 2  # t1, t2
        assert fc["n_probes"] == 3

    def test_missing_theme(self):
        themes = np.array(["a", "a"])
        texts = np.array(["t1", "t2"])
        fc = get_frame_counts(themes, texts, "b")
        assert fc["n_frames"] == 0
        assert fc["n_probes"] == 0


# ===================================================================
# Test: get_frame_compression_ratio
# ===================================================================

class TestFrameCompressionRatio:

    def test_nine_frames_one_topic(self):
        assert get_frame_compression_ratio(9, 1) == 9.0

    def test_three_frames_two_topics(self):
        assert get_frame_compression_ratio(3, 2) == 1.5

    def test_equal(self):
        assert get_frame_compression_ratio(5, 5) == 1.0

    def test_zero_topics(self):
        assert get_frame_compression_ratio(5, 0) == float('inf')


# ===================================================================
# Test: get_style_control_comparison
# ===================================================================

class TestStyleControlComparison:

    def test_basic_structure(self):
        artist = np.array([0]*50 + [1]*50)
        public = np.array([0]*30 + [1]*30 + [2]*30)
        probes = np.array([0]*20 + [1]*20 + [2]*10)
        themes = np.array(["a"]*100)

        result = get_style_control_comparison(artist, public, probes, themes, "a", n_clusters=5)

        assert "entropy_normalized" in result
        assert "salience_raw" in result
        assert "salience_style" in result
        assert "salience_delta" in result
        assert "value" in result["entropy_normalized"]

    def test_identical_public_and_probes(self):
        artist = np.array([0]*100)
        public = np.array([0]*10 + [1]*90)
        themes = np.array(["a"]*100)

        result = get_style_control_comparison(artist, public, public, themes, "a", n_clusters=5)
        assert abs(result["salience_delta"]["value"]) < 0.01


# ===================================================================
# Test: Real Data Validation
# ===================================================================

class TestRealDataValidation:

    def test_public_discourse_high_entropy(self, real_data):
        """Public discourse should have high entropy (spread across many topics)."""
        # We need cluster labels, so let's compute a simple KMeans
        from src.clustering import run_kmeans
        from src.consensus_umap import distance_matrix_consensus, run_umap_multi_seed, umap_from_precomputed_distances

        seeds = [42, 7, 101]  # minimal seeds for testing
        ue = run_umap_multi_seed(real_data["X_public"], seeds=seeds, n_components=8,
                                  n_neighbors=27, min_dist=0.1, metric="cosine")
        D = distance_matrix_consensus(ue, metric="euclidean")
        c8d, _ = umap_from_precomputed_distances(D, n_components=8, n_neighbors=27, min_dist=0.1)
        labels, _ = run_kmeans(c8d, n_clusters=28, metric="euclidean")

        ent = get_entropy(labels, n_clusters=28)
        # Public discourse should use most topics → high entropy
        assert ent["entropy_normalized"] > 0.7, f"Expected high entropy, got {ent['entropy_normalized']}"
        assert ent["n_occupied_clusters"] >= 18, f"Expected many occupied clusters, got {ent['n_occupied_clusters']}"

    def test_all_themes_present(self, real_data):
        """All 5 themes should be present in the data."""
        themes = set(real_data["theme_labels"])
        for expected in ["threat", "utility", "ownership", "transparency", "compensation"]:
            assert expected in themes, f"Missing theme: {expected}"

    def test_theme_probe_counts(self, real_data):
        """Each theme should have roughly 250 probes (252 respondents, 5 questions)."""
        for theme in ["threat", "utility", "ownership", "transparency", "compensation"]:
            count = (real_data["theme_labels"] == theme).sum()
            assert 200 <= count <= 300, f"Theme {theme} has {count} probes, expected ~252"

    def test_article_names_not_empty(self, real_data):
        """Article names should be non-empty strings."""
        articles = real_data["article_names"]
        assert len(articles) > 0
        assert all(isinstance(a, str) and len(a) > 0 for a in articles[:10])

    def test_frame_counts_sum_to_total(self, real_data):
        """Frame counts across all themes should sum to total probes."""
        total_frames = 0
        total_probes = 0
        texts = real_data["df_artist"]["perspective_text"].values
        for theme in ["threat", "utility", "ownership", "transparency", "compensation"]:
            fc = get_frame_counts(real_data["theme_labels"], texts, theme)
            total_frames += fc["n_frames"]
            total_probes += fc["n_probes"]
        assert total_probes == len(real_data["theme_labels"])
        assert total_frames >= 5  # at least 1 frame per theme
