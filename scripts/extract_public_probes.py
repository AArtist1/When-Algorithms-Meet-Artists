"""Extract public probes from the public discourse corpus using keyword matching.

Runs the keyword-based public probe extraction pipeline on the CLEAN
1,736-chunk corpus (April 2026 rebuild):
    1. Load clean chunks + precomputed prefix embeddings
    2. Build anchor keyword specs from 250 Likert phrases
    3. Extract candidate sentences from 1,736 public discourse chunks
    4. Embed candidates with e5-large-v2 + "query: " prefix
    5. Rank by cosine similarity to parent chunk embedding
    6. Select top 1-2 sentences per (chunk, theme)
    7. Deduplicate per theme
    8. Re-embed final probe text with prefix
    9. Save to data/public_probes_keyword.csv

The 250 Likert anchors serve only as retrieval queries and are
discarded after candidate extraction. They are not used in any
downstream analysis.

Output files:
    data/public_probes_keyword.csv
    figures/final_pipeline/keyword_probes_embeddings.npy

Side effects:
    Writes two files to disk. Prints progress to stdout.
    Loads e5-large-v2 model into memory (~1.3GB).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import load_clean_public_discourse, load_likert_anchors
from src.embeddings import embed_chunks
from src.public_probes import (
    get_anchor_specs,
    get_candidate_sentences,
    get_deduplicated_probes,
    get_selected_probes,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT / "data"
OUTPUT_CSV: Path = DATA_DIR / "public_probes_keyword.csv"
OUTPUT_NPY: Path = ROOT / "figures" / "final_pipeline" / "keyword_probes_embeddings.npy"
PREFIX_PUBLIC_NPY: Path = ROOT / "figures" / "prefix_grid_search" / "prefix_embeddings_public.npy"

MODEL_NAME: str = "intfloat/e5-large-v2"
PREFIX: str = "query: "
TEXT_COL: str = "chunk_text"
UNIT_COL: str = "article_name"
MIN_WORDS: int = 5
MAX_WORDS: int = 50
MIN_ANCHOR_HITS: int = 4
MIN_ANCHOR_LEN: int = 3
MAX_ANCHORS: int = 12


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def get_public_chunk_embeddings(npy_path: Path, expected_rows: int) -> np.ndarray:
    """Load precomputed prefix e5-large-v2 embeddings for public discourse chunks.

    Args:
        npy_path: Path to the .npy file.
        expected_rows: Expected number of rows (must match clean chunk count).

    Returns:
        Array of shape (expected_rows, 1024).

    Raises:
        ValueError: If shape mismatch.

    Side effects:
        Reads from disk.
    """
    X = np.load(npy_path)
    if X.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding shape {X.shape} does not match expected {expected_rows} rows. "
            f"Embeddings may be from a different corpus version."
        )
    return X


def set_embeddings_on_dataframe(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    col_name: str = "embeddings",
) -> pd.DataFrame:
    """Attach embedding vectors as a column on a DataFrame.

    Args:
        df: Input DataFrame (must have same number of rows as embeddings).
        embeddings: Array of shape (n, d).
        col_name: Name for the embedding column.

    Returns:
        DataFrame with embedding column added.

    Side effects:
        None (returns a copy).
    """
    df = df.copy()
    df[col_name] = [embeddings[i] for i in range(len(embeddings))]
    return df


def get_embedded_candidates(
    df_candidates: pd.DataFrame,
    model_name: str = MODEL_NAME,
    prefix: str = PREFIX,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Embed candidate sentence text with e5-large-v2 + prefix.

    Args:
        df_candidates: DataFrame with 'text' column.
        model_name: Sentence transformer model name.
        prefix: Prefix to prepend to each text.

    Returns:
        Tuple of (df with embeddings column, raw embedding array).

    Side effects:
        Loads model into memory. Prints progress.
    """
    X = embed_chunks(df_candidates, text_col="text", model_name=model_name,
                     batch_size=32, prefix=prefix)
    df_out = set_embeddings_on_dataframe(df_candidates, X)
    return df_out, X


def get_final_probe_embeddings(
    df_probes: pd.DataFrame,
    model_name: str = MODEL_NAME,
    prefix: str = PREFIX,
) -> np.ndarray:
    """Re-embed the final probe text with prefix.

    Args:
        df_probes: DataFrame with 'probe_text' column.
        model_name: Sentence transformer model name.
        prefix: Prefix to prepend to each text.

    Returns:
        Embedding array of shape (n_probes, 1024).

    Side effects:
        Loads model into memory. Prints progress.
    """
    df_temp = pd.DataFrame({"text": df_probes["probe_text"].values})
    return embed_chunks(df_temp, text_col="text", model_name=model_name,
                        batch_size=32, prefix=prefix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the keyword-based public probe extraction pipeline on clean 1,736-chunk data.

    Side effects:
        Writes CSV and NPY files. Prints progress. Loads model.
    """
    t0 = time.time()

    print("=" * 60)
    print("KEYWORD-BASED PUBLIC PROBE EXTRACTION")
    print("(Clean 1,736-chunk corpus, prefix embeddings)")
    print("=" * 60)

    # Step 1: Load clean data and precomputed prefix embeddings
    print("\nStep 1: Loading clean public discourse data...")
    df_public = load_clean_public_discourse(DATA_DIR)
    df_likert = load_likert_anchors(DATA_DIR)
    print(f"  Public chunks: {len(df_public)}")
    print(f"  Unique articles: {df_public['article_name'].nunique()}")
    print(f"  Likert anchors: {len(df_likert)}")

    print("\nStep 1b: Loading precomputed prefix embeddings...")
    X_public_parent = get_public_chunk_embeddings(PREFIX_PUBLIC_NPY, len(df_public))
    df_public = set_embeddings_on_dataframe(df_public, X_public_parent)
    print(f"  Parent embeddings: {X_public_parent.shape}")

    # Step 2: Build anchor specs
    print("\nStep 2: Building anchor specs from Likert phrases...")
    specs = get_anchor_specs(df_likert, min_anchor_len=MIN_ANCHOR_LEN, max_anchors=MAX_ANCHORS)
    print(f"  Anchor specs: {len(specs)} (from {len(df_likert)} phrases)")

    # Step 3: Extract candidate sentences
    print("\nStep 3: Extracting candidate sentences via keyword matching...")
    df_candidates = get_candidate_sentences(
        df_public, text_col=TEXT_COL, unit_col=UNIT_COL,
        specs=specs, min_words=MIN_WORDS, max_words=MAX_WORDS,
        min_anchor_hits=MIN_ANCHOR_HITS,
    )
    print(f"  Candidates: {len(df_candidates)}")
    if not df_candidates.empty:
        print(f"  Per theme: {dict(df_candidates['theme'].value_counts())}")

    if df_candidates.empty:
        print("ERROR: No candidates extracted. Check text column and parameters.")
        return

    # Step 4: Embed candidate sentences
    print("\nStep 4: Embedding candidate sentences with prefix...")
    df_candidates, X_cand = get_embedded_candidates(df_candidates)
    print(f"  Embedded: {X_cand.shape}")

    # Step 5: Select best probes by similarity to parent
    print("\nStep 5: Selecting probes by parent similarity...")
    df_selected = get_selected_probes(
        df_public, df_candidates,
        public_emb_col="embeddings", cand_emb_col="embeddings",
        max_sentences=2, redundancy_cosine=0.92,
    )
    print(f"  Selected: {len(df_selected)}")

    # Step 6: Deduplicate
    print("\nStep 6: Deduplicating per theme...")
    df_final = get_deduplicated_probes(df_selected)
    print(f"  Final probes: {len(df_final)}")
    print(f"  Per theme:")
    for theme, count in df_final["theme"].value_counts().items():
        print(f"    {theme}: {count}")

    # Step 7: Re-embed final probe text
    print("\nStep 7: Re-embedding final probe text with prefix...")
    X_final = get_final_probe_embeddings(df_final)
    print(f"  Final embeddings: {X_final.shape}")

    # Step 8: Save
    print("\nStep 8: Saving outputs...")
    output_cols = [
        "theme", "likert", "text", "probe_text", "probe_sent_ids",
        "row_id", "unit_id", "article_name",
        "anchor_hits_max", "n_phrase_matches", "probe_sim_mean",
    ]
    save_cols = [c for c in output_cols if c in df_final.columns]
    df_final[save_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV: {OUTPUT_CSV} ({len(df_final)} rows)")

    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_NPY, X_final)
    print(f"  NPY: {OUTPUT_NPY} {X_final.shape}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {len(df_final)} keyword-based public probes extracted.")


if __name__ == "__main__":
    main()
