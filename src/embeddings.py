"""Embedding generation for the When Algorithms Meet Artists pipeline."""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

PHRASE_MAP = {
    r"\bartificial intelligence\b": "artificial_intelligence",
    r"\bai\b": "artificial_intelligence",
    r"\bmachine learning\b": "machine_learning",
    r"\bml\b": "machine_learning",
    r"\bsci fi\b": "sci_fi",
    r"\bscience fiction\b": "sci_fi",
    r"\bdeep learning\b": "deep_learning",
    r"\bneural network\b": "neural_network",
}


def normalize_phrases(text: str) -> str:
    """Normalize multi-word phrases into single tokens for embedding consistency."""
    text = text.lower()
    for pattern, replacement in PHRASE_MAP.items():
        text = re.sub(pattern, replacement, text)
    return text


def clean_for_embedding(text: str) -> str:
    """Prepare text for embedding: lowercase and normalize phrases."""
    return normalize_phrases(text).lower()


def embed_chunks(
    df: pd.DataFrame,
    text_col: str = "chunk_text_clean",
    model_name: str = "intfloat/e5-large-v2",
    batch_size: int = 32,
    prefix: str | None = None,
) -> np.ndarray:
    """Encode text using a SentenceTransformer model.

    Args:
        df: DataFrame containing the text column.
        text_col: Column name with text to embed.
        model_name: SentenceTransformer model identifier.
        batch_size: Encoding batch size.
        prefix: Optional prefix to prepend to each text before encoding.
            For e5-large-v2, use "query: " for clustering/classification tasks.

    Returns:
        numpy array of shape (n_rows, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    texts = df[text_col].astype(str).tolist()
    if prefix is not None:
        texts = [prefix + t for t in texts]
        print(f"Prefix '{prefix}' applied to all texts.")
    print(f"Number of texts to encode: {len(texts)}")

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embeddings)


def load_or_embed(
    df: pd.DataFrame,
    text_col: str,
    model_name: str,
    precomputed_npy: str | None = None,
    batch_size: int = 32,
    prefix: str | None = None,
) -> np.ndarray:
    """Load precomputed embeddings or generate them.

    Args:
        df: DataFrame with text.
        text_col: Column name for text.
        model_name: SentenceTransformer model name.
        precomputed_npy: Path to precomputed .npy file (if available).
        batch_size: Batch size for encoding.
        prefix: Optional prefix to prepend to each text before encoding.

    Returns:
        Embedding array of shape (n_rows, embedding_dim).

    Raises:
        ValueError: If precomputed file row count doesn't match DataFrame.
    """
    if precomputed_npy is not None and os.path.exists(precomputed_npy):
        print(f"Loading precomputed embeddings from {precomputed_npy}")
        embeddings = np.load(precomputed_npy)
        if embeddings.shape[0] != len(df):
            raise ValueError(
                f"FAILED: Embedding rows ({embeddings.shape[0]}) != DataFrame rows ({len(df)})"
            )
        return embeddings

    return embed_chunks(df, text_col=text_col, model_name=model_name,
                        batch_size=batch_size, prefix=prefix)
