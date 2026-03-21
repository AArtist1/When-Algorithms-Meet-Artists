"""Tests for data content quality: value ranges, no nulls, valid categories."""

import pytest


@pytest.mark.data
class TestPublicDiscourseContent:
    def test_no_null_text_fields(self, df_public):
        nulls = df_public["chunk_text_clean"].isna().sum()
        assert nulls == 0, (
            f"FAILED: Found {nulls} null values in chunk_text_clean"
        )

    def test_no_empty_text_fields(self, df_public):
        empty = (df_public["chunk_text_clean"].str.strip() == "").sum()
        assert empty == 0, (
            f"FAILED: Found {empty} empty strings in chunk_text_clean"
        )

    def test_year_range(self, df_public):
        min_year = df_public["year"].min()
        max_year = df_public["year"].max()
        assert min_year >= 2013, (
            f"FAILED: Minimum year is {min_year}, expected >= 2013"
        )
        assert max_year <= 2025, (
            f"FAILED: Maximum year is {max_year}, expected <= 2025"
        )

    def test_valid_media_types(self, df_public):
        valid = {"article", "audio", "paper", "video"}
        actual = set(df_public["media_type"].str.strip().str.lower().unique())
        invalid = actual - valid
        assert not invalid, (
            f"FAILED: Invalid media types found: {invalid}. Expected only {valid}"
        )

    def test_unique_article_count_reasonable(self, df_public):
        n = df_public["article_name"].nunique()
        assert 100 <= n <= 140, (
            f"FAILED: Expected 100-140 unique articles, got {n}"
        )

    def test_lexical_diversity_in_range(self, df_public):
        ld = df_public["lexical_diversity"]
        assert ld.min() >= 0.0, (
            f"FAILED: lexical_diversity min is {ld.min()}, expected >= 0"
        )
        assert ld.max() <= 1.0, (
            f"FAILED: lexical_diversity max is {ld.max()}, expected <= 1"
        )

    def test_word_counts_positive(self, df_public):
        min_wc = df_public["chunk_word_count_clean"].min()
        assert min_wc > 0, (
            f"FAILED: Minimum word count is {min_wc}, expected > 0"
        )


@pytest.mark.data
class TestArtistContent:
    def test_no_null_perspective_text(self, df_artist):
        nulls = df_artist["perspective_text"].isna().sum()
        assert nulls == 0, (
            f"FAILED: Found {nulls} null values in perspective_text"
        )

    def test_no_empty_perspective_text(self, df_artist):
        empty = (df_artist["perspective_text"].str.strip() == "").sum()
        assert empty == 0, (
            f"FAILED: Found {empty} empty strings in perspective_text"
        )

    def test_valid_question_groups(self, df_artist):
        expected = {"compensation", "ownership", "threat", "transparency", "utility"}
        actual = set(df_artist["question_group"].str.strip().str.lower().unique())
        assert actual == expected, (
            f"FAILED: Expected question groups {expected}, got {actual}"
        )

    def test_respondent_count(self, df_artist):
        n = df_artist["respondent_id"].nunique()
        assert n == 252, (
            f"FAILED: Expected 252 unique respondents, got {n}"
        )

    def test_all_are_artists(self, df_artist):
        non_artists = (df_artist["Artist"].str.strip() != "Yes").sum()
        assert non_artists == 0, (
            f"FAILED: Found {non_artists} rows where Artist != 'Yes'"
        )


@pytest.mark.data
class TestLikertContent:
    def test_five_themes(self, df_likert):
        themes = set(df_likert["theme"].unique())
        expected = {"compensation", "ownership", "threat", "transparency", "utility"}
        assert themes == expected, (
            f"FAILED: Expected themes {expected}, got {themes}"
        )

    def test_five_likert_levels(self, df_likert):
        levels = set(df_likert["likert"].unique())
        expected = {"strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"}
        assert levels == expected, (
            f"FAILED: Expected Likert levels {expected}, got {levels}"
        )

    def test_no_empty_text(self, df_likert):
        empty = (df_likert["text"].str.strip() == "").sum()
        assert empty == 0, (
            f"FAILED: Found {empty} empty text fields in Likert anchors"
        )

    def test_unique_ids(self, df_likert):
        n_ids = df_likert["id"].nunique()
        n_rows = len(df_likert)
        assert n_ids == n_rows, (
            f"FAILED: Expected {n_rows} unique IDs, got {n_ids} — duplicates exist"
        )

    def test_factorial_design(self, df_likert):
        combos = df_likert.groupby(["theme", "likert"]).size()
        n_combos = len(combos)
        assert n_combos == 25, (
            f"FAILED: Expected 25 theme×likert combinations (5×5), got {n_combos}"
        )
        min_per_combo = combos.min()
        assert min_per_combo >= 1, (
            f"FAILED: Some theme×likert combinations have 0 rows"
        )
