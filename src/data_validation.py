"""Data validation utilities for the analysis pipeline.

Provides pre-embedding and pre-analysis checks to ensure data quality.
Designed to be reusable for other datasets beyond this project.

Usage:
    from src.data_validation import validate_chunks, validate_embeddings

    # Before embedding
    issues = validate_chunks(df, text_col="chunk_text")
    if issues:
        raise ValueError(f"Data quality issues: {issues}")

    # After embedding, before analysis
    issues = validate_embeddings(X, df, text_col="chunk_text")
    if issues:
        raise ValueError(f"Embedding quality issues: {issues}")
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pre-embedding: text chunk validation
# ---------------------------------------------------------------------------

def validate_chunks(
    df: pd.DataFrame,
    text_col: str = "chunk_text",
    min_words: int = 30,
    max_words: int = 300,
    max_token_estimate: int = 512,
    near_dupe_threshold: float = 0.85,
    near_dupe_sample: int = 10000,
    random_seed: int = 42,
) -> list[str]:
    """Validate text chunks before embedding.

    Checks:
        1. No exact duplicate texts
        2. No near-duplicate texts (high Jaccard similarity)
        3. All texts within word count bounds
        4. Estimated token count within model limit
        5. No empty or whitespace-only texts
        6. No encoding artifacts (control chars, private Unicode)
        7. Sufficient lexical diversity

    Args:
        df: DataFrame with text column.
        text_col: Name of the text column.
        min_words: Minimum words per chunk.
        max_words: Maximum words per chunk (soft limit).
        max_token_estimate: Maximum estimated tokens (words * 1.3).
        near_dupe_threshold: Jaccard threshold for near-duplicates.
        near_dupe_sample: Number of random pairs to check.
        random_seed: Seed for random pair sampling.

    Returns:
        List of issue strings. Empty list means all checks passed.
    """
    issues = []
    texts = df[text_col].astype(str).tolist()

    # 1. Exact duplicates
    n_unique = len(set(texts))
    if n_unique < len(texts):
        n_dupes = len(texts) - n_unique
        issues.append(f"EXACT_DUPLICATES: {n_dupes} duplicate texts found")

    # 2. Near-duplicates (random sample)
    rng = np.random.RandomState(random_seed)
    near_dupes = 0
    if len(texts) > 1:
        n_pairs = min(near_dupe_sample, len(texts) * (len(texts) - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.randint(0, len(texts)), rng.randint(0, len(texts))
            if i == j:
                continue
            words_i = set(texts[i].lower().split())
            words_j = set(texts[j].lower().split())
            if words_i and words_j:
                jaccard = len(words_i & words_j) / len(words_i | words_j)
                if jaccard > near_dupe_threshold:
                    near_dupes += 1
        if near_dupes > 0:
            issues.append(f"NEAR_DUPLICATES: {near_dupes} pairs above "
                          f"{near_dupe_threshold} Jaccard threshold (sampled {n_pairs} pairs)")

    # 3. Word count bounds
    word_counts = [len(t.split()) for t in texts]
    too_short = sum(1 for wc in word_counts if wc < min_words)
    too_long = sum(1 for wc in word_counts if wc > max_words)
    if too_short > 0:
        issues.append(f"TOO_SHORT: {too_short} chunks have <{min_words} words")
    if too_long > 0:
        issues.append(f"TOO_LONG: {too_long} chunks have >{max_words} words")

    # 4. Token limit
    over_limit = sum(1 for wc in word_counts if wc * 1.3 > max_token_estimate)
    if over_limit > 0:
        issues.append(f"TOKEN_LIMIT: {over_limit} chunks may exceed {max_token_estimate} tokens")

    # 5. Empty texts
    empty = sum(1 for t in texts if not t.strip())
    if empty > 0:
        issues.append(f"EMPTY: {empty} empty or whitespace-only texts")

    # 6. Encoding artifacts
    all_text = ''.join(texts)
    control_chars = set(c for c in all_text if ord(c) < 32 and c not in '\n\r\t ')
    private_unicode = bool(re.search(r'[\ue000-\uf8ff]', all_text))
    if control_chars:
        issues.append(f"CONTROL_CHARS: {len(control_chars)} unique control characters found")
    if private_unicode:
        issues.append("PRIVATE_UNICODE: Private-use Unicode characters found")

    # 7. Lexical diversity
    low_diversity = 0
    for t in texts:
        words = t.lower().split()
        if len(words) > 20:
            ratio = len(set(words)) / len(words)
            if ratio < 0.25:
                low_diversity += 1
    if low_diversity > 0:
        issues.append(f"LOW_DIVERSITY: {low_diversity} chunks with <25% unique words")

    return issues


# ---------------------------------------------------------------------------
# Post-embedding: embedding validation
# ---------------------------------------------------------------------------

def validate_embeddings(
    X: np.ndarray,
    df: pd.DataFrame,
    text_col: str = "chunk_text",
    max_identical_pairs: int = 0,
    identity_threshold: float = 0.9999,
    sample_pairs: int = 10000,
    random_seed: int = 42,
) -> list[str]:
    """Validate embeddings after generation, before analysis.

    Checks:
        1. Shape matches DataFrame row count
        2. No NaN or Inf values
        3. All vectors are L2-normalized
        4. No identical embedding vectors (cosine > identity_threshold)
        5. Different texts produce different embeddings
        6. Reasonable similarity distribution

    Args:
        X: Embedding array of shape (n, d).
        df: Source DataFrame.
        text_col: Text column name.
        max_identical_pairs: Maximum allowed near-identical pairs.
        identity_threshold: Cosine threshold for "identical" embeddings.
        sample_pairs: Number of random pairs to check.
        random_seed: Seed for sampling.

    Returns:
        List of issue strings. Empty list means all checks passed.
    """
    issues = []

    # 1. Shape
    if X.shape[0] != len(df):
        issues.append(f"SHAPE_MISMATCH: Embedding rows ({X.shape[0]}) != "
                      f"DataFrame rows ({len(df)})")

    # 2. NaN / Inf
    if np.any(np.isnan(X)):
        issues.append(f"NAN: {np.isnan(X).sum()} NaN values in embeddings")
    if np.any(np.isinf(X)):
        issues.append(f"INF: {np.isinf(X).sum()} Inf values in embeddings")

    # 3. L2 normalization
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        issues.append(f"NOT_NORMALIZED: Norms range [{norms.min():.4f}, {norms.max():.4f}], "
                      f"expected ~1.0")

    # 4. No identical embeddings
    rng = np.random.RandomState(random_seed)
    n_identical = 0
    identical_examples = []
    n_to_check = min(sample_pairs, X.shape[0] * (X.shape[0] - 1) // 2)
    for _ in range(n_to_check):
        i, j = rng.randint(0, X.shape[0]), rng.randint(0, X.shape[0])
        if i == j:
            continue
        sim = float(np.dot(X[i], X[j]))
        if sim > identity_threshold:
            n_identical += 1
            if len(identical_examples) < 3:
                identical_examples.append((i, j, sim))
    if n_identical > max_identical_pairs:
        issues.append(f"IDENTICAL_EMBEDDINGS: {n_identical} pairs with cosine > "
                      f"{identity_threshold} (sampled {n_to_check})")
        for i, j, sim in identical_examples:
            issues.append(f"  Example: rows {i} & {j}, cosine={sim:.6f}")

    # 5. Similarity distribution
    sims = []
    for _ in range(min(5000, n_to_check)):
        i, j = rng.randint(0, X.shape[0]), rng.randint(0, X.shape[0])
        if i != j:
            sims.append(float(np.dot(X[i], X[j])))
    sims = np.array(sims)

    if sims.std() < 0.01:
        issues.append(f"LOW_VARIANCE: Similarity std={sims.std():.4f} — "
                      f"embeddings may be collapsed")
    if sims.mean() > 0.98:
        issues.append(f"HIGH_MEAN_SIM: Mean similarity={sims.mean():.4f} — "
                      f"embeddings may not differentiate texts")

    return issues


# ---------------------------------------------------------------------------
# Cluster size validation
# ---------------------------------------------------------------------------

def validate_cluster_sizes(
    labels: np.ndarray,
    n_samples: int,
    min_cluster_size: int = 10,
    min_pct_of_corpus: float = 0.005,
) -> list[str]:
    """Validate cluster sizes meet minimum requirements.

    For topic modeling with c-TF-IDF, clusters need enough documents
    for reliable term frequency estimation. The floor ensures:
        - Multiple source documents per cluster (not single-article clusters)
        - Enough text for discriminative term extraction
        - Statistical reliability for downstream metrics

    Args:
        labels: Cluster assignments, shape (n,).
        n_samples: Total number of samples in the corpus.
        min_cluster_size: Absolute minimum chunks per cluster.
        min_pct_of_corpus: Minimum cluster size as fraction of corpus.

    Returns:
        List of issue strings. Empty list means all checks passed.
    """
    issues = []
    sizes = Counter(int(l) for l in labels)

    # Absolute minimum
    dynamic_floor = max(min_cluster_size, int(n_samples * min_pct_of_corpus))

    violations = {k: v for k, v in sizes.items() if v < dynamic_floor}
    if violations:
        issues.append(
            f"SMALL_CLUSTERS: {len(violations)} clusters below floor of "
            f"{dynamic_floor} (min_cluster_size={min_cluster_size}, "
            f"min_pct={min_pct_of_corpus:.1%} of {n_samples}): "
            f"{dict(sorted(violations.items(), key=lambda x: x[1]))}"
        )

    # Check total
    total = sum(sizes.values())
    if total != n_samples:
        issues.append(f"COUNT_MISMATCH: Label count ({total}) != n_samples ({n_samples})")

    return issues
