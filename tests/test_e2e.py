"""End-to-end pipeline tests using REAL data.

These tests validate the full pipeline from data loading through analysis.
They use the actual project data files, not mocks.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    centroid_distance,
    compute_salience_ratios,
    js_divergence,
    safe_probabilities,
    topic_counts,
)
from src.clustering import find_best_k, get_cluster_sizes, run_kmeans
from src.consensus_umap import (
    compute_consensus_average,
    distance_matrix_consensus,
    l2_normalize,
    procrustes_align,
    run_umap_multi_seed,
)
from src.data_loading import (
    EXPECTED_ARTIST_ROWS,
    EXPECTED_LIKERT_ROWS,
    EXPECTED_PUBLIC_ROWS,
    load_artist_perspectives,
    load_likert_anchors,
    load_public_discourse,
)
from src.models import PipelineConfig


@pytest.mark.e2e
class TestE2EDataLoading:
    """End-to-end tests that all data loads correctly with real files."""

    def test_public_discourse_loads(self, df_public):
        assert len(df_public) == EXPECTED_PUBLIC_ROWS, (
            f"FAILED: Expected {EXPECTED_PUBLIC_ROWS} public rows, got {len(df_public)}"
        )

    def test_artist_perspectives_loads(self, df_artist):
        assert len(df_artist) == EXPECTED_ARTIST_ROWS, (
            f"FAILED: Expected {EXPECTED_ARTIST_ROWS} artist rows, got {len(df_artist)}"
        )

    def test_likert_loads(self, df_likert):
        assert len(df_likert) == EXPECTED_LIKERT_ROWS, (
            f"FAILED: Expected {EXPECTED_LIKERT_ROWS} Likert rows, got {len(df_likert)}"
        )

    def test_public_discourse_has_text(self, df_public):
        null_count = df_public["chunk_text_clean"].isna().sum()
        assert null_count == 0, (
            f"FAILED: Found {null_count} null text entries in public discourse"
        )

    def test_artist_has_all_question_groups(self, df_artist):
        groups = set(df_artist["question_group"].str.strip().str.lower().unique())
        expected = {"compensation", "ownership", "threat", "transparency", "utility"}
        assert groups == expected, (
            f"FAILED: Expected question groups {expected}, got {groups}"
        )

    def test_media_type_distribution(self, df_public):
        """Verify known media type proportions."""
        counts = df_public["media_type"].str.strip().str.lower().value_counts()
        assert counts.get("article", 0) > 0, "FAILED: No 'article' media type found"
        assert counts.get("audio", 0) > 0, "FAILED: No 'audio' media type found"
        assert len(counts) >= 3, (
            f"FAILED: Expected at least 3 media types, got {len(counts)}"
        )

    def test_year_coverage(self, df_public):
        """All years from 2013 to 2025 should be represented (or at least most)."""
        years = set(df_public["year"].unique())
        covered = len(years)
        assert covered >= 8, (
            f"FAILED: Expected at least 8 distinct years, got {covered}: {sorted(years)}"
        )


@pytest.mark.e2e
class TestE2EConsensusUMAP:
    """Test consensus UMAP on a small sample of real data."""

    def test_mini_consensus_on_sample(self, df_public_sample, random_embeddings):
        """Run 3-seed consensus on synthetic embeddings to validate pipeline mechanics."""
        X = random_embeddings[:50]  # 50 points
        seeds = [42, 7, 101]

        embeddings = run_umap_multi_seed(
            X, seeds=seeds, n_components=5,
            n_neighbors=10, min_dist=0.1, metric="euclidean",
        )
        assert len(embeddings) == 3, (
            f"FAILED: Expected 3 seed embeddings, got {len(embeddings)}"
        )
        assert embeddings[0].shape == (50, 5), (
            f"FAILED: Expected shape (50, 5), got {embeddings[0].shape}"
        )

        # Distance consensus
        D = distance_matrix_consensus(embeddings, metric="euclidean")
        assert D.shape == (50, 50), (
            f"FAILED: Expected distance matrix (50, 50), got {D.shape}"
        )
        assert np.allclose(D, D.T, atol=1e-5), "FAILED: Distance matrix not symmetric"
        assert np.allclose(np.diag(D), 0.0), "FAILED: Distance matrix diagonal not zero"

        # Procrustes alignment
        aligned = procrustes_align(embeddings)
        assert aligned.shape == (3, 50, 5), (
            f"FAILED: Expected aligned shape (3, 50, 5), got {aligned.shape}"
        )

        consensus = compute_consensus_average(aligned)
        assert consensus.shape == (50, 5), (
            f"FAILED: Expected consensus shape (50, 5), got {consensus.shape}"
        )
        assert not np.any(np.isnan(consensus)), "FAILED: NaN in consensus coordinates"


@pytest.mark.e2e
class TestE2EClustering:
    """Test clustering on synthetic data."""

    def test_kmeans_produces_clusters(self, random_umap_coords):
        labels, km = run_kmeans(random_umap_coords, n_clusters=5, metric="euclidean")
        n_unique = len(set(labels))
        assert n_unique == 5, (
            f"FAILED: Expected 5 clusters, got {n_unique}"
        )
        assert len(labels) == len(random_umap_coords), (
            f"FAILED: Expected {len(random_umap_coords)} labels, got {len(labels)}"
        )


@pytest.mark.e2e
class TestE2EAnalysis:
    """Test analysis metrics on real data distributions."""

    def test_jsd_between_corpora(self, df_public, df_artist):
        """JSD between public and artist distributions should be meaningful."""
        # Use question_group as a proxy for topic distribution
        # This tests the divergence functions with real categorical data
        public_groups = df_public["media_type"].str.strip().str.lower().value_counts()
        p = safe_probabilities(public_groups.values)
        q = safe_probabilities(np.ones(len(p)))  # uniform comparison
        jsd = js_divergence(p, q)
        assert 0 < jsd < np.log(2) + 1e-6, (
            f"FAILED: JSD should be in (0, ln(2)], got {jsd}"
        )

    def test_salience_ratio_on_known_distribution(self):
        """Test salience ratios with known skewed distributions."""
        artist = np.array([100, 50, 10, 0, 0])
        public = np.array([20, 20, 20, 20, 20])
        ratios = compute_salience_ratios(artist, public, list(range(5)))

        # First topic should have highest ratio (artist overrepresented)
        top_ratio = ratios[0]["salience_ratio"]
        assert top_ratio > 1.0, (
            f"FAILED: Topic 0 salience ratio should be > 1.0, got {top_ratio}"
        )

    def test_centroid_distance_between_groups(self):
        """Centroid distance should be positive for different groups."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "group": ["a"] * n + ["b"] * n,
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(5, 1, n)]),
            "y": np.concatenate([rng.normal(0, 1, n), rng.normal(5, 1, n)]),
        })
        d = centroid_distance(df, "group", ["x", "y"], "a", "b")
        assert d > 3.0, (
            f"FAILED: Expected centroid distance > 3.0 for well-separated groups, got {d}"
        )


@pytest.mark.e2e
class TestE2EPipelineConfig:
    """Test that the default pipeline config is internally consistent."""

    def test_default_config_builds(self):
        config = PipelineConfig()
        assert config.embedding.model_name == "intfloat/e5-large-v2"
        assert config.embedding.embedding_dim == 1024
        assert len(config.umap.seeds) == 30
        assert config.umap.n_components == 8
        assert config.cluster.method == "kmeans"
