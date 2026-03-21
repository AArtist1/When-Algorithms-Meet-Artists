"""Tests for embedding generation and quality."""

import numpy as np
import pytest

from src.embeddings import clean_for_embedding, normalize_phrases


@pytest.mark.embedding
class TestNormalization:
    def test_normalize_ai(self):
        result = normalize_phrases("ai is changing art")
        assert "artificial_intelligence" in result, (
            f"FAILED: Expected 'artificial_intelligence' in '{result}'"
        )

    def test_normalize_ml(self):
        result = normalize_phrases("machine learning models")
        assert "machine_learning" in result, (
            f"FAILED: Expected 'machine_learning' in '{result}'"
        )

    def test_normalize_preserves_other_text(self):
        result = normalize_phrases("artists are concerned about copyright")
        assert "artists" in result
        assert "copyright" in result

    def test_clean_lowercases(self):
        result = clean_for_embedding("AI Art Is GREAT")
        assert result == result.lower(), (
            f"FAILED: Expected lowercase output, got '{result}'"
        )


@pytest.mark.embedding
class TestEmbeddingShape:
    def test_random_embeddings_shape(self, random_embeddings):
        assert random_embeddings.shape == (100, 1024), (
            f"FAILED: Expected shape (100, 1024), got {random_embeddings.shape}"
        )

    def test_random_embeddings_no_nan(self, random_embeddings):
        n_nan = np.isnan(random_embeddings).sum()
        assert n_nan == 0, (
            f"FAILED: Found {n_nan} NaN values in embeddings"
        )

    def test_random_embeddings_no_inf(self, random_embeddings):
        n_inf = np.isinf(random_embeddings).sum()
        assert n_inf == 0, (
            f"FAILED: Found {n_inf} Inf values in embeddings"
        )

    def test_random_embeddings_finite_norms(self, random_embeddings):
        norms = np.linalg.norm(random_embeddings, axis=1)
        assert np.all(norms > 0), (
            f"FAILED: Found {(norms == 0).sum()} zero-norm embedding vectors"
        )
        assert np.all(np.isfinite(norms)), (
            f"FAILED: Found {(~np.isfinite(norms)).sum()} non-finite norms"
        )
