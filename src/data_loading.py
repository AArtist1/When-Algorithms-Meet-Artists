"""Data loading and validation for the When Algorithms Meet Artists pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .models import ArtistProbe, CorpusMetadata, LikertPhrase, PublicChunk

EXPECTED_PUBLIC_COLUMNS = [
    "line_number", "year", "article_name", "media_type", "specific_type",
    "lexical_diversity", "section_id", "text_og", "text_phrase_norm",
    "chunk_text_norm", "chunk_word_count_norm", "chunk_text_clean",
    "chunk_word_count_clean", "chunk_text_lexical", "chunk_word_count_lexical",
]

EXPECTED_ARTIST_COLUMNS = [
    "respondent_id", "Artist", "Art_practice", "Purchase_art", "Professional_artist",
    "AI_models_familiarity", "Used_AI_art_models", "compensation", "Age", "POC",
    "Gender_identity", "Country", "question_group", "perspective_text",
]

EXPECTED_LIKERT_COLUMNS = ["id", "theme", "likert", "style", "text"]

EXPECTED_PUBLIC_ROWS = 891
EXPECTED_ARTIST_ROWS = 1259
EXPECTED_LIKERT_ROWS = 250


def load_public_discourse(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the public discourse corpus from two split CSV files and concatenate.

    Args:
        data_dir: Directory containing public_discourse_part1.csv and part2.csv.

    Returns:
        Combined DataFrame with 891 rows and 15 columns.

    Raises:
        FileNotFoundError: If either split file is missing.
        ValueError: If the concatenated row count does not match expected.
    """
    data_dir = Path(data_dir)
    part1_path = data_dir / "public_discourse_part1.csv"
    part2_path = data_dir / "public_discourse_part2.csv"

    if not part1_path.exists():
        raise FileNotFoundError(f"FAILED: Public discourse part 1 not found at {part1_path}")
    if not part2_path.exists():
        raise FileNotFoundError(f"FAILED: Public discourse part 2 not found at {part2_path}")

    df1 = pd.read_csv(part1_path)
    df2 = pd.read_csv(part2_path)
    df = pd.concat([df1, df2], ignore_index=True)

    if len(df) != EXPECTED_PUBLIC_ROWS:
        raise ValueError(
            f"FAILED: Expected {EXPECTED_PUBLIC_ROWS} public discourse rows after concat, "
            f"got {len(df)}"
        )

    missing = set(EXPECTED_PUBLIC_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"FAILED: Missing columns in public discourse data: {sorted(missing)}")

    return df


def load_artist_perspectives(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the filtered artist perspectives CSV.

    Returns:
        DataFrame with 1,259 rows and 14 columns.
    """
    path = Path(data_dir) / "artist_perspectives.csv"
    if not path.exists():
        raise FileNotFoundError(f"FAILED: Artist perspectives not found at {path}")

    df = pd.read_csv(path)

    if len(df) != EXPECTED_ARTIST_ROWS:
        raise ValueError(
            f"FAILED: Expected {EXPECTED_ARTIST_ROWS} artist perspective rows, got {len(df)}"
        )

    missing = set(EXPECTED_ARTIST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"FAILED: Missing columns in artist perspectives: {sorted(missing)}")

    return df


def load_likert_anchors(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the Likert anchor phrases CSV.

    Returns:
        DataFrame with 250 rows and 5 columns.
    """
    path = Path(data_dir) / "likert_anchor_phrases.csv"
    if not path.exists():
        raise FileNotFoundError(f"FAILED: Likert anchor phrases not found at {path}")

    df = pd.read_csv(path)

    if len(df) != EXPECTED_LIKERT_ROWS:
        raise ValueError(
            f"FAILED: Expected {EXPECTED_LIKERT_ROWS} Likert rows, got {len(df)}"
        )

    missing = set(EXPECTED_LIKERT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"FAILED: Missing columns in Likert anchors: {sorted(missing)}")

    return df


def load_lovato_survey(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the original Lovato et al. (2024) survey CSV."""
    path = Path(data_dir) / "lovato_survey" / "ai_art_surveydata_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"FAILED: Lovato survey not found at {path}")
    return pd.read_csv(path)


def validate_dataframe(df: pd.DataFrame, model_class: type, max_errors: int = 10) -> list[str]:
    """Validate each row of a DataFrame against a Pydantic model.

    Args:
        df: DataFrame to validate.
        model_class: Pydantic model class (PublicChunk, ArtistProbe, or LikertPhrase).
        max_errors: Stop collecting errors after this many.

    Returns:
        List of error messages. Empty list means all rows valid.
    """
    errors: list[str] = []
    for idx, row in df.iterrows():
        try:
            model_class(**row.to_dict())
        except Exception as e:
            errors.append(f"Row {idx}: {e}")
            if len(errors) >= max_errors:
                errors.append(f"... (stopped after {max_errors} errors)")
                break
    return errors


def get_corpus_metadata(
    df: pd.DataFrame,
    name: str,
    source_path: str | Path,
    document_col: str = "article_name",
    year_col: str | None = "year",
) -> CorpusMetadata:
    """Build a CorpusMetadata model from a loaded DataFrame."""
    return CorpusMetadata(
        name=name,
        n_rows=len(df),
        n_documents=df[document_col].nunique() if document_col in df.columns else 0,
        year_min=int(df[year_col].min()) if year_col and year_col in df.columns else None,
        year_max=int(df[year_col].max()) if year_col and year_col in df.columns else None,
        date_loaded=datetime.now(),
        source_path=Path(source_path),
        columns=list(df.columns),
    )
