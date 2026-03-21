"""Tests for data loading: file existence, columns, row counts, dtypes."""

from pathlib import Path

import pytest

from src.data_loading import (
    EXPECTED_ARTIST_COLUMNS,
    EXPECTED_ARTIST_ROWS,
    EXPECTED_LIKERT_COLUMNS,
    EXPECTED_LIKERT_ROWS,
    EXPECTED_PUBLIC_COLUMNS,
    EXPECTED_PUBLIC_ROWS,
)

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.mark.data
class TestFileExistence:
    def test_public_part1_exists(self):
        path = DATA_DIR / "public_discourse_part1.csv"
        assert path.exists(), f"FAILED: File not found at {path}"

    def test_public_part2_exists(self):
        path = DATA_DIR / "public_discourse_part2.csv"
        assert path.exists(), f"FAILED: File not found at {path}"

    def test_artist_perspectives_exists(self):
        path = DATA_DIR / "artist_perspectives.csv"
        assert path.exists(), f"FAILED: File not found at {path}"

    def test_likert_anchors_exists(self):
        path = DATA_DIR / "likert_anchor_phrases.csv"
        assert path.exists(), f"FAILED: File not found at {path}"

    def test_lovato_survey_exists(self):
        path = DATA_DIR / "lovato_survey" / "ai_art_surveydata_cleaned.csv"
        assert path.exists(), f"FAILED: File not found at {path}"


@pytest.mark.data
class TestPublicDiscourse:
    def test_row_count(self, df_public):
        assert len(df_public) == EXPECTED_PUBLIC_ROWS, (
            f"FAILED: Expected {EXPECTED_PUBLIC_ROWS} public discourse rows, got {len(df_public)}"
        )

    def test_all_columns_present(self, df_public):
        missing = set(EXPECTED_PUBLIC_COLUMNS) - set(df_public.columns)
        assert not missing, (
            f"FAILED: Missing columns in public discourse: {sorted(missing)}"
        )

    def test_year_dtype_is_numeric(self, df_public):
        assert df_public["year"].dtype in ("int64", "int32", "float64"), (
            f"FAILED: Expected numeric year dtype, got {df_public['year'].dtype}"
        )

    def test_lexical_diversity_is_float(self, df_public):
        assert df_public["lexical_diversity"].dtype == "float64", (
            f"FAILED: Expected float64 lexical_diversity, got {df_public['lexical_diversity'].dtype}"
        )


@pytest.mark.data
class TestArtistPerspectives:
    def test_row_count(self, df_artist):
        assert len(df_artist) == EXPECTED_ARTIST_ROWS, (
            f"FAILED: Expected {EXPECTED_ARTIST_ROWS} artist rows, got {len(df_artist)}"
        )

    def test_all_columns_present(self, df_artist):
        missing = set(EXPECTED_ARTIST_COLUMNS) - set(df_artist.columns)
        assert not missing, (
            f"FAILED: Missing columns in artist perspectives: {sorted(missing)}"
        )


@pytest.mark.data
class TestLikertAnchors:
    def test_row_count(self, df_likert):
        assert len(df_likert) == EXPECTED_LIKERT_ROWS, (
            f"FAILED: Expected {EXPECTED_LIKERT_ROWS} Likert rows, got {len(df_likert)}"
        )

    def test_all_columns_present(self, df_likert):
        missing = set(EXPECTED_LIKERT_COLUMNS) - set(df_likert.columns)
        assert not missing, (
            f"FAILED: Missing columns in Likert anchors: {sorted(missing)}"
        )
