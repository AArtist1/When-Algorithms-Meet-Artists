"""Tests for statistical analysis functions."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    centroid_distance,
    cohens_d,
    compute_salience_ratios,
    js_divergence,
    kl_divergence,
    safe_probabilities,
    topic_counts,
)


@pytest.mark.stats
class TestSafeProbabilities:
    def test_sums_to_one(self):
        counts = np.array([10, 20, 30])
        probs = safe_probabilities(counts)
        assert abs(probs.sum() - 1.0) < 1e-10, (
            f"FAILED: Probabilities sum to {probs.sum()}, expected 1.0"
        )

    def test_no_zeros(self):
        counts = np.array([0, 0, 100])
        probs = safe_probabilities(counts)
        assert np.all(probs > 0), (
            "FAILED: safe_probabilities should have no zero entries"
        )


@pytest.mark.stats
class TestDivergences:
    def test_kl_self_is_zero(self):
        p = np.array([0.3, 0.3, 0.4])
        kl = kl_divergence(p, p)
        assert abs(kl) < 1e-10, (
            f"FAILED: KL(p||p) should be 0, got {kl}"
        )

    def test_kl_positive(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        kl = kl_divergence(p, q)
        assert kl > 0, (
            f"FAILED: KL divergence should be positive for different distributions, got {kl}"
        )

    def test_jsd_symmetric(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.3, 0.6])
        jsd_pq = js_divergence(p, q)
        jsd_qp = js_divergence(q, p)
        assert abs(jsd_pq - jsd_qp) < 1e-10, (
            f"FAILED: JSD should be symmetric — JSD(p,q)={jsd_pq}, JSD(q,p)={jsd_qp}"
        )

    def test_jsd_bounded(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = js_divergence(p, q)
        assert 0 <= jsd <= np.log(2) + 1e-6, (
            f"FAILED: JSD should be in [0, ln(2)≈0.693], got {jsd}"
        )

    def test_jsd_identical_is_zero(self):
        p = np.array([0.5, 0.5])
        jsd = js_divergence(p, p)
        assert abs(jsd) < 1e-10, (
            f"FAILED: JSD of identical distributions should be ~0, got {jsd}"
        )


@pytest.mark.stats
class TestTopicCounts:
    def test_matches_value_counts(self):
        df = pd.DataFrame({"cluster": [0, 0, 1, 1, 1, 2]})
        all_labels = [0, 1, 2]
        counts = topic_counts(df, "cluster", all_labels)
        assert list(counts) == [2, 3, 1], (
            f"FAILED: Expected [2, 3, 1], got {list(counts)}"
        )

    def test_missing_label_gets_zero(self):
        df = pd.DataFrame({"cluster": [0, 0, 1]})
        all_labels = [0, 1, 2]
        counts = topic_counts(df, "cluster", all_labels)
        assert counts[2] == 0, (
            f"FAILED: Label 2 (absent) should have count 0, got {counts[2]}"
        )


@pytest.mark.stats
class TestCentroidDistance:
    def test_same_group_distance_zero(self):
        df = pd.DataFrame({
            "group": ["a", "a", "a"],
            "x": [1.0, 1.0, 1.0],
            "y": [2.0, 2.0, 2.0],
        })
        d = centroid_distance(df, "group", ["x", "y"], "a", "a")
        assert abs(d) < 1e-10, (
            f"FAILED: Distance of group to itself should be 0, got {d}"
        )

    def test_known_distance(self):
        df = pd.DataFrame({
            "group": ["a", "b"],
            "x": [0.0, 3.0],
            "y": [0.0, 4.0],
        })
        d = centroid_distance(df, "group", ["x", "y"], "a", "b")
        assert abs(d - 5.0) < 1e-10, (
            f"FAILED: Expected distance 5.0, got {d}"
        )


@pytest.mark.stats
class TestSalienceRatios:
    def test_uniform_ratio_near_one(self):
        counts = np.array([100, 100, 100])
        ratios = compute_salience_ratios(counts, counts, [0, 1, 2])
        for r in ratios:
            assert abs(r["salience_ratio"] - 1.0) < 0.1, (
                f"FAILED: Uniform distribution should yield ratio ~1.0, got {r['salience_ratio']}"
            )

    def test_concentrated_ratio_high(self):
        artist = np.array([100, 0, 0])
        public = np.array([33, 33, 34])
        ratios = compute_salience_ratios(artist, public, [0, 1, 2])
        assert ratios[0]["salience_ratio"] > 2.0, (
            f"FAILED: Concentrated artist distribution should yield high ratio, "
            f"got {ratios[0]['salience_ratio']}"
        )


@pytest.mark.stats
class TestCohensD:
    def test_identical_groups_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        d = cohens_d(a, a)
        assert abs(d) < 1e-10, (
            f"FAILED: Cohen's d for identical groups should be 0, got {d}"
        )

    def test_large_difference(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(10, 1, 100)
        d = cohens_d(a, b)
        assert abs(d) > 5.0, (
            f"FAILED: Large group difference should yield |d| > 5, got {d}"
        )
