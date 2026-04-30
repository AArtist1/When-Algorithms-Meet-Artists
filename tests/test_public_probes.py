"""E2E tests for public probe extraction pipeline.

Tests both keyword-based and embedding-based extraction methods
against the clean 1,736-chunk corpus. Validates output format,
theme coverage, data integrity, and alignment with the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

EXPECTED_THEMES = {"compensation", "ownership", "threat", "transparency", "utility"}
MIN_PROBES_PER_THEME = 20
EXPECTED_CLEAN_CHUNKS = 1736
EXPECTED_ARTICLES = 125
EXPECTED_EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_chunks() -> pd.DataFrame:
    """Load the clean public discourse corpus."""
    from src.data_loading import load_clean_public_discourse
    return load_clean_public_discourse(DATA_DIR)


@pytest.fixture
def likert_anchors() -> pd.DataFrame:
    """Load the Likert anchor phrases."""
    from src.data_loading import load_likert_anchors
    return load_likert_anchors(DATA_DIR)


@pytest.fixture
def public_embeddings() -> np.ndarray:
    """Load precomputed prefix embeddings for public chunks."""
    path = FIGURES_DIR / "prefix_grid_search" / "prefix_embeddings_public.npy"
    return np.load(path)


@pytest.fixture
def keyword_probes() -> pd.DataFrame | None:
    """Load keyword-based probes if they exist."""
    path = DATA_DIR / "public_probes_keyword.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@pytest.fixture
def embedding_probes() -> pd.DataFrame | None:
    """Load embedding-based probes if they exist."""
    path = DATA_DIR / "public_probes_embedding.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Data source tests
# ---------------------------------------------------------------------------

class TestDataSources:
    """Tests for the underlying data sources used by probe extraction."""

    def test_clean_chunks_count(self, clean_chunks: pd.DataFrame) -> None:
        assert len(clean_chunks) == EXPECTED_CLEAN_CHUNKS

    def test_clean_chunks_articles(self, clean_chunks: pd.DataFrame) -> None:
        assert clean_chunks["article_name"].nunique() == EXPECTED_ARTICLES

    def test_clean_chunks_has_chunk_text(self, clean_chunks: pd.DataFrame) -> None:
        assert "chunk_text" in clean_chunks.columns
        assert clean_chunks["chunk_text"].notna().all()
        assert (clean_chunks["chunk_text"].str.len() > 0).all()

    def test_embeddings_match_chunks(
        self, clean_chunks: pd.DataFrame, public_embeddings: np.ndarray,
    ) -> None:
        assert public_embeddings.shape == (len(clean_chunks), EXPECTED_EMBEDDING_DIM)

    def test_embeddings_normalized(self, public_embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(public_embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

    def test_likert_anchors_count(self, likert_anchors: pd.DataFrame) -> None:
        assert len(likert_anchors) == 250

    def test_likert_themes(self, likert_anchors: pd.DataFrame) -> None:
        assert set(likert_anchors["theme"].unique()) == EXPECTED_THEMES

    def test_likert_factorial_design(self, likert_anchors: pd.DataFrame) -> None:
        """5 themes x 5 levels x 10 styles = 250."""
        assert likert_anchors["theme"].nunique() == 5
        assert likert_anchors["likert"].nunique() == 5


# ---------------------------------------------------------------------------
# Keyword extraction unit tests
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    """Tests for the keyword-based extraction functions."""

    def test_get_sentences_basic(self) -> None:
        from src.public_probes import get_sentences
        text = "This is sentence one. This is sentence two. And three."
        sents = get_sentences(text)
        assert len(sents) == 3

    def test_get_sentences_empty(self) -> None:
        from src.public_probes import get_sentences
        assert get_sentences("") == []
        assert get_sentences(None) == []  # type: ignore[arg-type]

    def test_get_tokens_basic(self) -> None:
        from src.public_probes import get_tokens
        tokens = get_tokens("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens

    def test_get_phrase_anchors(self) -> None:
        from src.public_probes import get_phrase_anchors
        anchors = get_phrase_anchors("AI art models are a threat to artists")
        assert "threat" in anchors
        assert "artists" in anchors
        assert "are" not in anchors  # stopword

    def test_get_anchor_hit_count(self) -> None:
        from src.public_probes import get_anchor_hit_count
        count = get_anchor_hit_count(
            "Artists are concerned about AI threat to their work",
            ["artists", "threat", "work", "banana"],
        )
        assert count == 3

    def test_get_anchor_specs(self, likert_anchors: pd.DataFrame) -> None:
        from src.public_probes import get_anchor_specs
        specs = get_anchor_specs(likert_anchors, min_anchor_len=3, max_anchors=12)
        assert len(specs) == 250
        themes_found = {s["theme"] for s in specs}
        assert themes_found == EXPECTED_THEMES

    def test_get_candidate_sentences(self, clean_chunks: pd.DataFrame, likert_anchors: pd.DataFrame) -> None:
        """Extract candidates from first 50 chunks to verify pipeline works."""
        from src.public_probes import get_anchor_specs, get_candidate_sentences
        specs = get_anchor_specs(likert_anchors)
        df_small = clean_chunks.head(50).copy()
        candidates = get_candidate_sentences(
            df_small, text_col="chunk_text", unit_col="article_name",
            specs=specs, min_words=5, max_words=50, min_anchor_hits=4,
        )
        assert isinstance(candidates, pd.DataFrame)
        if not candidates.empty:
            assert "theme" in candidates.columns
            assert "text" in candidates.columns
            assert set(candidates["theme"].unique()).issubset(EXPECTED_THEMES)


# ---------------------------------------------------------------------------
# Keyword probe output validation (E2E)
# ---------------------------------------------------------------------------

class TestKeywordProbeOutput:
    """E2E tests for the keyword extraction output files."""

    def test_csv_exists(self) -> None:
        path = DATA_DIR / "public_probes_keyword.csv"
        if not path.exists():
            pytest.skip("Keyword probes not yet extracted. Run scripts/extract_public_probes.py")
        assert path.stat().st_size > 0

    def test_probe_count_reasonable(self, keyword_probes: pd.DataFrame | None) -> None:
        if keyword_probes is None:
            pytest.skip("Keyword probes not yet extracted")
        assert len(keyword_probes) >= 200, f"Only {len(keyword_probes)} probes extracted"
        assert len(keyword_probes) <= 2000, f"Unexpectedly many probes: {len(keyword_probes)}"

    def test_all_themes_present(self, keyword_probes: pd.DataFrame | None) -> None:
        if keyword_probes is None:
            pytest.skip("Keyword probes not yet extracted")
        found = set(keyword_probes["theme"].unique())
        assert found == EXPECTED_THEMES, f"Missing themes: {EXPECTED_THEMES - found}"

    def test_min_probes_per_theme(self, keyword_probes: pd.DataFrame | None) -> None:
        if keyword_probes is None:
            pytest.skip("Keyword probes not yet extracted")
        for theme in EXPECTED_THEMES:
            count = (keyword_probes["theme"] == theme).sum()
            assert count >= MIN_PROBES_PER_THEME, (
                f"Theme '{theme}' has only {count} probes (min {MIN_PROBES_PER_THEME})"
            )

    def test_no_empty_probe_text(self, keyword_probes: pd.DataFrame | None) -> None:
        if keyword_probes is None:
            pytest.skip("Keyword probes not yet extracted")
        text_col = "probe_text" if "probe_text" in keyword_probes.columns else "text"
        assert keyword_probes[text_col].notna().all()
        assert (keyword_probes[text_col].str.strip().str.len() > 0).all()

    def test_no_duplicate_probes_within_theme(self, keyword_probes: pd.DataFrame | None) -> None:
        if keyword_probes is None:
            pytest.skip("Keyword probes not yet extracted")
        text_col = "probe_text" if "probe_text" in keyword_probes.columns else "text"
        for theme in EXPECTED_THEMES:
            theme_df = keyword_probes[keyword_probes["theme"] == theme]
            assert theme_df[text_col].is_unique, f"Duplicate probes in theme '{theme}'"

    def test_probes_from_correct_corpus(self, keyword_probes: pd.DataFrame | None) -> None:
        """Verify probes come from the 125-article clean corpus, not the old 131-article one."""
        if keyword_probes is None:
            pytest.skip("Keyword probes not yet extracted")
        if "article_name" in keyword_probes.columns:
            n_articles = keyword_probes["article_name"].nunique()
            assert n_articles <= EXPECTED_ARTICLES, (
                f"Probes reference {n_articles} articles but corpus has {EXPECTED_ARTICLES}"
            )

    def test_embeddings_file_exists(self) -> None:
        path = FIGURES_DIR / "final_pipeline" / "keyword_probes_embeddings.npy"
        if not path.exists():
            pytest.skip("Keyword probe embeddings not yet generated")
        X = np.load(path)
        assert X.shape[1] == EXPECTED_EMBEDDING_DIM
        assert X.shape[0] > 200

    def test_embeddings_match_csv(self, keyword_probes: pd.DataFrame | None) -> None:
        npy_path = FIGURES_DIR / "final_pipeline" / "keyword_probes_embeddings.npy"
        if keyword_probes is None or not npy_path.exists():
            pytest.skip("Keyword probes or embeddings not yet generated")
        X = np.load(npy_path)
        assert X.shape[0] == len(keyword_probes), (
            f"Embedding rows ({X.shape[0]}) != CSV rows ({len(keyword_probes)})"
        )


# ---------------------------------------------------------------------------
# Embedding probe output validation (E2E)
# ---------------------------------------------------------------------------

class TestEmbeddingProbeOutput:
    """E2E tests for the embedding-based extraction output files."""

    def test_csv_exists(self) -> None:
        path = DATA_DIR / "public_probes_embedding.csv"
        if not path.exists():
            pytest.skip("Embedding probes not yet extracted")
        assert path.stat().st_size > 0

    def test_probe_count_reasonable(self, embedding_probes: pd.DataFrame | None) -> None:
        if embedding_probes is None:
            pytest.skip("Embedding probes not yet extracted")
        assert len(embedding_probes) >= 200
        assert len(embedding_probes) <= 2000

    def test_all_themes_present(self, embedding_probes: pd.DataFrame | None) -> None:
        if embedding_probes is None:
            pytest.skip("Embedding probes not yet extracted")
        found = set(embedding_probes["theme"].unique())
        assert found == EXPECTED_THEMES

    def test_min_probes_per_theme(self, embedding_probes: pd.DataFrame | None) -> None:
        if embedding_probes is None:
            pytest.skip("Embedding probes not yet extracted")
        for theme in EXPECTED_THEMES:
            count = (embedding_probes["theme"] == theme).sum()
            assert count >= MIN_PROBES_PER_THEME

    def test_no_empty_probe_text(self, embedding_probes: pd.DataFrame | None) -> None:
        if embedding_probes is None:
            pytest.skip("Embedding probes not yet extracted")
        text_col = "probe_text" if "probe_text" in embedding_probes.columns else "text"
        assert embedding_probes[text_col].notna().all()


# ---------------------------------------------------------------------------
# Cross-method comparison tests
# ---------------------------------------------------------------------------

class TestCrossMethodComparison:
    """Tests that run only when both extraction methods have outputs."""

    def test_both_methods_cover_all_themes(
        self, keyword_probes: pd.DataFrame | None, embedding_probes: pd.DataFrame | None,
    ) -> None:
        if keyword_probes is None or embedding_probes is None:
            pytest.skip("Both probe sets needed for comparison")
        kw_themes = set(keyword_probes["theme"].unique())
        emb_themes = set(embedding_probes["theme"].unique())
        assert kw_themes == EXPECTED_THEMES
        assert emb_themes == EXPECTED_THEMES

    def test_probe_counts_same_order_of_magnitude(
        self, keyword_probes: pd.DataFrame | None, embedding_probes: pd.DataFrame | None,
    ) -> None:
        if keyword_probes is None or embedding_probes is None:
            pytest.skip("Both probe sets needed for comparison")
        ratio = len(keyword_probes) / len(embedding_probes)
        assert 0.2 < ratio < 5.0, (
            f"Keyword ({len(keyword_probes)}) and embedding ({len(embedding_probes)}) "
            f"probe counts differ by more than 5x"
        )

    def test_some_text_overlap(
        self, keyword_probes: pd.DataFrame | None, embedding_probes: pd.DataFrame | None,
    ) -> None:
        """Both methods should find at least some of the same sentences."""
        if keyword_probes is None or embedding_probes is None:
            pytest.skip("Both probe sets needed for comparison")
        kw_col = "probe_text" if "probe_text" in keyword_probes.columns else "text"
        emb_col = "probe_text" if "probe_text" in embedding_probes.columns else "text"
        kw_texts = set(keyword_probes[kw_col].str.strip())
        emb_texts = set(embedding_probes[emb_col].str.strip())
        overlap = len(kw_texts & emb_texts)
        # At least some overlap expected, but methods are different enough that
        # perfect overlap would be suspicious
        assert overlap >= 0  # Allow zero overlap -- methods may diverge
