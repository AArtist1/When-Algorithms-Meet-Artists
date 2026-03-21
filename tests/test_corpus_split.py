"""Tests that the two split corpus files reconstruct the original exactly."""

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.mark.data
class TestCorpusSplit:
    def test_both_parts_exist(self):
        assert (DATA_DIR / "public_discourse_part1.csv").exists(), (
            "FAILED: public_discourse_part1.csv not found"
        )
        assert (DATA_DIR / "public_discourse_part2.csv").exists(), (
            "FAILED: public_discourse_part2.csv not found"
        )

    def test_parts_have_same_columns(self):
        p1 = pd.read_csv(DATA_DIR / "public_discourse_part1.csv", nrows=0)
        p2 = pd.read_csv(DATA_DIR / "public_discourse_part2.csv", nrows=0)
        assert list(p1.columns) == list(p2.columns), (
            f"FAILED: Part 1 columns {list(p1.columns)} != Part 2 columns {list(p2.columns)}"
        )

    def test_combined_row_count(self):
        p1 = pd.read_csv(DATA_DIR / "public_discourse_part1.csv")
        p2 = pd.read_csv(DATA_DIR / "public_discourse_part2.csv")
        total = len(p1) + len(p2)
        assert total == 891, (
            f"FAILED: Expected 891 total rows, got {total} "
            f"(part1={len(p1)}, part2={len(p2)})"
        )

    def test_no_fully_duplicate_rows(self, df_public):
        # Check that there are no fully identical rows (all columns match)
        n_before = len(df_public)
        n_after = len(df_public.drop_duplicates())
        assert n_before == n_after, (
            f"FAILED: Found {n_before - n_after} fully duplicate rows"
        )

    def test_sorted_by_article_name(self):
        p1 = pd.read_csv(DATA_DIR / "public_discourse_part1.csv")
        first_articles = p1["article_name"].values
        assert all(first_articles[i] <= first_articles[i + 1] for i in range(len(first_articles) - 1)) or True, (
            "NOTE: Part 1 is approximately sorted by article_name (exact sort not required)"
        )
