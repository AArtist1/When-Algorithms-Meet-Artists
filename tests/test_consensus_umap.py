"""Tests for the consensus UMAP pipeline."""

import numpy as np
import pytest

from src.consensus_umap import (
    compute_consensus_average,
    compute_pairwise_ari,
    distance_matrix_consensus,
    l2_normalize,
    procrustes_align,
    run_umap_multi_seed,
)


@pytest.mark.umap
class TestL2Normalize:
    def test_output_unit_norms(self, random_embeddings):
        X_norm = l2_normalize(random_embeddings)
        norms = np.linalg.norm(X_norm, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"FAILED: Expected unit norms, got range [{norms.min():.6f}, {norms.max():.6f}]"
        )


@pytest.mark.umap
class TestProcrustes:
    def test_align_preserves_shape(self):
        rng = np.random.default_rng(42)
        embs = [rng.standard_normal((50, 8)).astype(np.float32) for _ in range(3)]
        aligned = procrustes_align(embs)
        assert aligned.shape == (3, 50, 8), (
            f"FAILED: Expected shape (3, 50, 8), got {aligned.shape}"
        )

    def test_align_rejects_mismatched_shapes(self):
        rng = np.random.default_rng(42)
        embs = [
            rng.standard_normal((50, 8)).astype(np.float32),
            rng.standard_normal((40, 8)).astype(np.float32),
        ]
        with pytest.raises(ValueError, match="shape"):
            procrustes_align(embs)

    def test_align_empty_raises(self):
        with pytest.raises(ValueError):
            procrustes_align([])


@pytest.mark.umap
class TestConsensusAverage:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        aligned = rng.standard_normal((5, 50, 8)).astype(np.float32)
        consensus = compute_consensus_average(aligned)
        assert consensus.shape == (50, 8), (
            f"FAILED: Expected shape (50, 8), got {consensus.shape}"
        )

    def test_consensus_reduces_variance(self):
        rng = np.random.default_rng(42)
        aligned = rng.standard_normal((5, 50, 8)).astype(np.float32)
        consensus = compute_consensus_average(aligned)
        mean_seed_var = np.mean([np.var(aligned[i]) for i in range(5)])
        consensus_var = np.var(consensus)
        assert consensus_var < mean_seed_var, (
            f"FAILED: Consensus variance ({consensus_var:.4f}) should be less than "
            f"mean seed variance ({mean_seed_var:.4f})"
        )


@pytest.mark.umap
class TestDistanceMatrixConsensus:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        embs = [rng.standard_normal((30, 5)).astype(np.float32) for _ in range(3)]
        D = distance_matrix_consensus(embs)
        assert D.shape == (30, 30), (
            f"FAILED: Expected shape (30, 30), got {D.shape}"
        )

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        embs = [rng.standard_normal((30, 5)).astype(np.float32) for _ in range(3)]
        D = distance_matrix_consensus(embs)
        assert np.allclose(D, D.T, atol=1e-6), (
            "FAILED: Distance matrix is not symmetric"
        )

    def test_diagonal_zeros(self):
        rng = np.random.default_rng(42)
        embs = [rng.standard_normal((30, 5)).astype(np.float32) for _ in range(3)]
        D = distance_matrix_consensus(embs)
        assert np.allclose(np.diag(D), 0.0), (
            "FAILED: Distance matrix diagonal is not zero"
        )

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        embs = [rng.standard_normal((30, 5)).astype(np.float32) for _ in range(3)]
        D = distance_matrix_consensus(embs)
        assert np.all(D >= 0), (
            f"FAILED: Found {(D < 0).sum()} negative entries in distance matrix"
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            distance_matrix_consensus([])


@pytest.mark.umap
class TestPairwiseARI:
    def test_output_shape(self, random_labels):
        labels_arr = np.stack([random_labels, random_labels, np.roll(random_labels, 5)])
        ari = compute_pairwise_ari(labels_arr)
        assert ari.shape == (3, 3), (
            f"FAILED: Expected shape (3, 3), got {ari.shape}"
        )

    def test_self_ari_is_zero_on_diagonal(self, random_labels):
        labels_arr = np.stack([random_labels, random_labels])
        ari = compute_pairwise_ari(labels_arr)
        assert ari[0, 0] == 0.0, "FAILED: Diagonal should be 0 (self-ARI not stored)"

    def test_identical_labels_ari_is_1(self, random_labels):
        labels_arr = np.stack([random_labels, random_labels])
        ari = compute_pairwise_ari(labels_arr)
        assert abs(ari[0, 1] - 1.0) < 1e-10, (
            f"FAILED: ARI of identical labelings should be 1.0, got {ari[0, 1]}"
        )

    def test_symmetric(self, random_labels):
        shifted = np.roll(random_labels, 10)
        labels_arr = np.stack([random_labels, shifted])
        ari = compute_pairwise_ari(labels_arr)
        assert abs(ari[0, 1] - ari[1, 0]) < 1e-10, (
            "FAILED: ARI matrix is not symmetric"
        )
