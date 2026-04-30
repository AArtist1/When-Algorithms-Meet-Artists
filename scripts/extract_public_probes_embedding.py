"""Extract public probes using embedding-based retrieval (cosine similarity to theme centroids).

Pipeline:
    1. Load 1,736 clean public discourse chunks and precomputed prefix embeddings
    2. Load 250 Likert anchor phrases and embed them with e5-large-v2 + "query: " prefix
    3. Compute theme centroids (average embedding per theme) from Likert anchors
    4. Split all chunks into individual sentences, filter to 5-50 words
    5. Embed all filtered sentences with e5-large-v2 + "query: " prefix (batch_size=64)
    6. Compute cosine similarity of each sentence to each of the 5 theme centroids
    7. Assign each sentence to its best-matching theme; take top N per theme
    8. Deduplicate within theme and save to data/public_probes_embedding.csv
    9. Re-embed final probes and save to figures/final_pipeline/embedding_probes_embeddings.npy

Unlike scripts/extract_public_probes.py (keyword matching), this script uses
pure embedding-space retrieval: sentences are ranked by cosine similarity to
theme centroids derived from the Likert anchor phrase embeddings.

Output files:
    data/public_probes_embedding.csv
    figures/final_pipeline/embedding_probes_embeddings.npy

Side effects:
    Writes two files to disk. Prints progress to stdout.
    Loads e5-large-v2 model into memory (~1.3GB).
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import load_clean_public_discourse, load_likert_anchors
from src.embeddings import embed_chunks
from src.public_probes import (
    get_embedding_based_probes,
    get_sentences,
    get_sentence_theme_similarities,
    get_theme_centroids,
    get_tokens,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT / "data"
FIGURES_DIR: Path = ROOT / "figures" / "final_pipeline"
PRECOMPUTED_PUBLIC_NPY: Path = ROOT / "figures" / "prefix_grid_search" / "prefix_embeddings_public.npy"

OUTPUT_CSV: Path = DATA_DIR / "public_probes_embedding.csv"
OUTPUT_NPY: Path = FIGURES_DIR / "embedding_probes_embeddings.npy"

MODEL_NAME: str = "intfloat/e5-large-v2"
PREFIX: str = "query: "
BATCH_SIZE: int = 64
MIN_WORDS: int = 5
MAX_WORDS: int = 50
TOP_N_PER_THEME: int = 150


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def get_precomputed_public_embeddings(path: Path, expected_rows: int) -> np.ndarray:
    """Load precomputed prefix embeddings for the public discourse chunks.

    Args:
        path: Path to the .npy file.
        expected_rows: Expected number of rows (must match loaded array).

    Returns:
        Array of shape (expected_rows, 1024).

    Raises:
        FileNotFoundError: If the .npy file does not exist.
        ValueError: If row count does not match expected.

    Side effects:
        Reads from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Precomputed embeddings not found at {path}")
    X = np.load(path)
    if X.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding rows ({X.shape[0]}) != expected ({expected_rows})"
        )
    return X


def get_anchor_embeddings(
    df_likert: pd.DataFrame,
    model_name: str = MODEL_NAME,
    prefix: str = PREFIX,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed Likert anchor phrases with e5-large-v2 + prefix.

    Args:
        df_likert: DataFrame with 'text' column containing anchor phrases.
        model_name: SentenceTransformer model identifier.
        prefix: Prefix to prepend (e.g., "query: " for e5-large-v2).
        batch_size: Encoding batch size.

    Returns:
        Array of shape (n_anchors, 1024).

    Side effects:
        Loads model into memory. Prints progress via embed_chunks.
    """
    return embed_chunks(
        df_likert, text_col="text", model_name=model_name,
        batch_size=batch_size, prefix=prefix,
    )


def get_filtered_sentences(
    df_public: pd.DataFrame,
    text_col: str = "chunk_text",
    unit_col: str = "article_name",
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> pd.DataFrame:
    """Split all public chunks into sentences and filter by word count.

    Args:
        df_public: Public discourse DataFrame with text and article columns.
        text_col: Column containing chunk text.
        unit_col: Column identifying the source article.
        min_words: Minimum words per sentence (inclusive).
        max_words: Maximum words per sentence (inclusive).

    Returns:
        DataFrame with columns: row_id, article_name, sent_id, text, n_words.
        Each row is one sentence from one chunk.

    Side effects:
        None.
    """
    rows: list[dict[str, Any]] = []
    for row_id, r in df_public.iterrows():
        article_name = r.get(unit_col, "")
        text = r.get(text_col, "")
        if not isinstance(text, str) or not text.strip():
            continue

        sents = get_sentences(text)
        for sent_id, sent in enumerate(sents):
            n_words = len(get_tokens(sent))
            if n_words < min_words or n_words > max_words:
                continue
            rows.append({
                "row_id": int(row_id),  # type: ignore[arg-type]
                "article_name": str(article_name),
                "sent_id": sent_id,
                "text": sent,
                "n_words": n_words,
            })

    return pd.DataFrame(rows)


def get_sentence_embeddings(
    df_sentences: pd.DataFrame,
    model_name: str = MODEL_NAME,
    prefix: str = PREFIX,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed sentence text with e5-large-v2 + prefix.

    Args:
        df_sentences: DataFrame with 'text' column.
        model_name: SentenceTransformer model identifier.
        prefix: Prefix to prepend to each sentence.
        batch_size: Encoding batch size.

    Returns:
        Array of shape (n_sentences, 1024).

    Side effects:
        Loads model into memory. Prints progress via embed_chunks.
    """
    return embed_chunks(
        df_sentences, text_col="text", model_name=model_name,
        batch_size=batch_size, prefix=prefix,
    )


def get_final_probe_embeddings(
    df_probes: pd.DataFrame,
    model_name: str = MODEL_NAME,
    prefix: str = PREFIX,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Re-embed the final selected probe text.

    Args:
        df_probes: DataFrame with 'probe_text' column.
        model_name: SentenceTransformer model identifier.
        prefix: Prefix to prepend to each probe.
        batch_size: Encoding batch size.

    Returns:
        Array of shape (n_probes, 1024).

    Side effects:
        Loads model into memory. Prints progress via embed_chunks.
    """
    df_temp = pd.DataFrame({"text": df_probes["probe_text"].values})
    return embed_chunks(
        df_temp, text_col="text", model_name=model_name,
        batch_size=batch_size, prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the embedding-based public probe extraction pipeline.

    Side effects:
        Writes CSV and NPY files to disk. Prints progress to stdout.
        Loads e5-large-v2 model into memory (~1.3GB).
    """
    t0 = time.time()

    print("=" * 60)
    print("EMBEDDING-BASED PUBLIC PROBE EXTRACTION PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load clean public chunks and precomputed embeddings
    # ------------------------------------------------------------------
    print("\nStep 1: Loading clean public discourse chunks...")
    t1 = time.time()
    df_public = load_clean_public_discourse(DATA_DIR)
    print(f"  Chunks loaded: {len(df_public)}")

    X_public = get_precomputed_public_embeddings(PRECOMPUTED_PUBLIC_NPY, len(df_public))
    print(f"  Precomputed embeddings: {X_public.shape} ({X_public.dtype})")
    print(f"  Step 1 time: {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Load and embed Likert anchor phrases
    # ------------------------------------------------------------------
    print("\nStep 2: Loading and embedding Likert anchor phrases...")
    t2 = time.time()
    df_likert = load_likert_anchors(DATA_DIR)
    print(f"  Likert anchors: {len(df_likert)}")
    themes_list: list[str] = df_likert["theme"].tolist()
    unique_themes = sorted(set(themes_list))
    print(f"  Themes: {unique_themes}")

    X_anchors = get_anchor_embeddings(df_likert)
    print(f"  Anchor embeddings: {X_anchors.shape}")
    print(f"  Step 2 time: {time.time() - t2:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Compute theme centroids
    # ------------------------------------------------------------------
    print("\nStep 3: Computing theme centroids...")
    t3 = time.time()
    centroids = get_theme_centroids(X_anchors, themes_list)
    for theme, centroid in centroids.items():
        print(f"  {theme}: norm={np.linalg.norm(centroid):.4f}, shape={centroid.shape}")
    print(f"  Step 3 time: {time.time() - t3:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Split chunks into filtered sentences
    # ------------------------------------------------------------------
    print("\nStep 4: Splitting chunks into sentences (filtering {}-{} words)...".format(
        MIN_WORDS, MAX_WORDS
    ))
    t4 = time.time()
    df_sentences = get_filtered_sentences(df_public, text_col="chunk_text")
    print(f"  Filtered sentences: {len(df_sentences)}")
    print(f"  From {df_sentences['row_id'].nunique()} unique chunks")
    print(f"  Word count range: {df_sentences['n_words'].min()}-{df_sentences['n_words'].max()}")
    print(f"  Step 4 time: {time.time() - t4:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Embed all filtered sentences (expensive step)
    # ------------------------------------------------------------------
    print(f"\nStep 5: Embedding {len(df_sentences)} sentences (batch_size={BATCH_SIZE})...")
    t5 = time.time()
    X_sentences = get_sentence_embeddings(df_sentences)
    print(f"  Sentence embeddings: {X_sentences.shape}")
    print(f"  Step 5 time: {time.time() - t5:.1f}s")

    # ------------------------------------------------------------------
    # Step 6: Compute cosine similarity to theme centroids
    # ------------------------------------------------------------------
    print("\nStep 6: Computing sentence-to-centroid similarities...")
    t6 = time.time()
    similarities = get_sentence_theme_similarities(X_sentences, centroids)
    print(f"  Similarity matrix: {similarities.shape}")
    print(f"  Theme assignment distribution:")
    theme_counts = similarities["best_theme"].value_counts()
    for theme, count in theme_counts.items():
        mean_sim = similarities.loc[similarities["best_theme"] == theme, "best_sim"].mean()
        print(f"    {theme}: {count} sentences (mean sim={mean_sim:.4f})")
    print(f"  Step 6 time: {time.time() - t6:.1f}s")

    # ------------------------------------------------------------------
    # Step 7: Select top-N per theme
    # ------------------------------------------------------------------
    print(f"\nStep 7: Selecting top {TOP_N_PER_THEME} per theme...")
    t7 = time.time()
    df_probes = get_embedding_based_probes(
        df_sentences, similarities, top_n_per_theme=TOP_N_PER_THEME,
    )
    # Add probe_text column (same as text for single-sentence probes)
    df_probes["probe_text"] = df_probes["text"]
    print(f"  Selected probes: {len(df_probes)}")
    print(f"  Per theme:")
    for theme, count in df_probes["theme"].value_counts().sort_index().items():
        mean_sim = df_probes.loc[df_probes["theme"] == theme, "sim_to_centroid"].mean()
        print(f"    {theme}: {count} probes (mean sim={mean_sim:.4f})")
    print(f"  Step 7 time: {time.time() - t7:.1f}s")

    # ------------------------------------------------------------------
    # Step 8: Re-embed final probes
    # ------------------------------------------------------------------
    print(f"\nStep 8: Re-embedding {len(df_probes)} final probes...")
    t8 = time.time()
    X_final = get_final_probe_embeddings(df_probes)
    print(f"  Final probe embeddings: {X_final.shape}")
    print(f"  Step 8 time: {time.time() - t8:.1f}s")

    # ------------------------------------------------------------------
    # Step 9: Save outputs
    # ------------------------------------------------------------------
    print("\nStep 9: Saving outputs...")
    t9 = time.time()

    output_cols = ["theme", "text", "probe_text", "row_id", "article_name", "sim_to_centroid"]
    save_cols = [c for c in output_cols if c in df_probes.columns]
    df_probes[save_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV: {OUTPUT_CSV} ({len(df_probes)} rows)")

    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_NPY, X_final)
    print(f"  NPY: {OUTPUT_NPY} {X_final.shape}")
    print(f"  Step 9 time: {time.time() - t9:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed:.0f}s")
    print(f"  Total probes: {len(df_probes)}")
    print(f"  Themes: {sorted(df_probes['theme'].unique())}")
    print(f"  Articles represented: {df_probes['article_name'].nunique()}")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"  NPY: {OUTPUT_NPY}")
    print("=" * 60)


if __name__ == "__main__":
    main()
