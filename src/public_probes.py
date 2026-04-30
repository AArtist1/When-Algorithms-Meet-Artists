"""Public probe extraction pipeline.

Extracts style-matched public probe sentences from the public discourse corpus
using Likert anchor phrases as keyword-based retrieval queries, then ranks
candidates by embedding similarity to their parent chunk.

Pipeline:
    1. Build keyword anchor specs from Likert phrases
    2. Split public chunks into sentences
    3. Filter sentences by anchor keyword hits
    4. Rank candidates by cosine similarity to parent chunk embedding
    5. Select top 1-2 sentences per (chunk, theme) with redundancy filtering
    6. Deduplicate per theme
    7. Re-embed final probe text

The 250 Likert anchor phrases serve ONLY as retrieval queries. They are
discarded after step 3 and never used in downstream analysis.

Functions follow a functional pattern: get_* for pure computation,
no hidden side effects unless documented.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "or", "a", "an", "of", "in", "on", "for", "to", "that",
    "this", "these", "those", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "being", "been", "it", "its", "at", "about", "into", "over",
    "under", "up", "down", "out", "off", "so", "such", "just", "very",
    "really", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "i", "our", "any",
})

GENERIC_DOMAIN_WORDS: frozenset[str] = frozenset({"artificial", "intelligence"})


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def get_tokens(text: str) -> list[str]:
    """Tokenize text into lowercase alphabetic tokens.

    Args:
        text: Input string.

    Returns:
        List of lowercase token strings.

    Side effects:
        None.
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z']+", text.lower())


def get_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation.

    Uses a simple regex split on period, question mark, or exclamation mark
    followed by whitespace. Suitable for edited text (news, academic, legal).

    Args:
        text: Input text string.

    Returns:
        List of non-empty sentence strings.

    Side effects:
        None.
    """
    if not isinstance(text, str):
        return []
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Anchor spec construction
# ---------------------------------------------------------------------------

def get_phrase_anchors(
    text: str,
    max_anchors: int = 10,
    stopwords: frozenset[str] = STOPWORDS,
    generic_words: frozenset[str] = GENERIC_DOMAIN_WORDS,
) -> list[str]:
    """Extract keyword anchors from a Likert phrase by removing stopwords.

    Args:
        text: Likert phrase text.
        max_anchors: Maximum anchors to return.
        stopwords: Words to exclude.
        generic_words: Domain-generic words to exclude.

    Returns:
        List of unique anchor tokens (up to max_anchors).

    Side effects:
        None.
    """
    anchors: list[str] = []
    for tok in get_tokens(text):
        if tok in stopwords or tok in generic_words:
            continue
        if tok not in anchors:
            anchors.append(tok)
        if len(anchors) >= max_anchors:
            break
    return anchors


def get_anchor_hit_count(sentence: str, anchors: list[str]) -> int:
    """Count how many anchor keywords appear in a sentence (substring match).

    Args:
        sentence: The sentence to search.
        anchors: List of anchor keywords.

    Returns:
        Number of anchors found in the sentence.

    Side effects:
        None.
    """
    s = (sentence or "").lower()
    return sum(1 for a in anchors if a in s)


def get_anchor_specs(
    df_likert: pd.DataFrame,
    min_anchor_len: int = 3,
    max_anchors: int = 10,
) -> list[dict[str, Any]]:
    """Build anchor spec dicts from Likert anchor phrases.

    Each spec contains the theme, likert level, original text, phrase ID,
    and extracted keyword anchors. Phrases with too few anchors are skipped.

    Args:
        df_likert: DataFrame with columns [id, theme, likert, text].
        min_anchor_len: Minimum number of anchors required to keep a phrase.
        max_anchors: Maximum anchors per phrase.

    Returns:
        List of spec dicts with keys: id, theme, likert, text, anchors.

    Side effects:
        None.
    """
    specs: list[dict[str, Any]] = []
    for _, row in df_likert.iterrows():
        anchors = get_phrase_anchors(str(row["text"]), max_anchors=max_anchors)
        if len(anchors) < min_anchor_len:
            continue
        specs.append({
            "id": row["id"],
            "theme": row["theme"],
            "likert": row["likert"],
            "text": row["text"],
            "anchors": anchors,
        })
    return specs


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def get_candidate_sentences(
    df_public: pd.DataFrame,
    text_col: str,
    unit_col: str,
    specs: list[dict[str, Any]],
    min_words: int = 5,
    max_words: int = 50,
    min_anchor_hits: int = 4,
) -> pd.DataFrame:
    """Extract candidate public probe sentences using anchor keyword matching.

    For each public chunk, splits into sentences, then checks each sentence
    against each anchor spec. Sentences with enough keyword hits are kept
    as candidates with full provenance.

    Args:
        df_public: Public discourse DataFrame (one row per chunk).
        text_col: Column containing the text for each chunk.
        unit_col: Column identifying the source article/document.
        specs: List of anchor spec dicts from get_anchor_specs().
        min_words: Minimum words per candidate sentence.
        max_words: Maximum words per candidate sentence.
        min_anchor_hits: Minimum anchor hits to keep a sentence.

    Returns:
        DataFrame with columns: row_id, unit_id, sent_id, theme, likert,
        phrase_id, phrase_text, anchor_hits, text, n_words, prev_sent, next_sent.

    Side effects:
        None.
    """
    rows: list[dict[str, Any]] = []

    for row_id, r in df_public.iterrows():
        unit_id = r.get(unit_col, None)
        text = str(r.get(text_col, "") or "")

        sents = get_sentences(text)
        toks = [get_tokens(s) for s in sents]

        for sent_id, sent in enumerate(sents):
            n_words = len(toks[sent_id])
            if n_words < min_words or n_words > max_words:
                continue

            prev_sent = sents[sent_id - 1] if sent_id > 0 else ""
            next_sent = sents[sent_id + 1] if sent_id + 1 < len(sents) else ""

            for spec in specs:
                hits = get_anchor_hit_count(sent, spec["anchors"])
                if hits >= min_anchor_hits:
                    rows.append({
                        "row_id": int(row_id),  # type: ignore[arg-type]
                        "unit_id": unit_id,
                        "sent_id": sent_id,
                        "theme": spec["theme"],
                        "likert": spec["likert"],
                        "phrase_id": spec["id"],
                        "phrase_text": spec["text"],
                        "anchor_hits": hits,
                        "text": sent,
                        "n_words": n_words,
                        "prev_sent": prev_sent,
                        "next_sent": next_sent,
                    })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    # Collapse multiple phrase matches for the same sentence+theme
    grp_cols = ["row_id", "unit_id", "sent_id", "theme", "text", "prev_sent", "next_sent"]
    agg = (
        raw.sort_values("anchor_hits", ascending=False)
        .groupby(grp_cols, sort=False)
        .agg(
            n_phrase_matches=("phrase_id", "nunique"),
            anchor_hits_max=("anchor_hits", "max"),
            likerts=("likert", lambda x: ",".join(sorted(set(x)))),
            phrase_ids=("phrase_id", lambda x: list(sorted(set(x)))),
        )
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Similarity-based selection
# ---------------------------------------------------------------------------

def get_l2_normalized(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization.

    Args:
        X: Input array, shape (n, d).
        eps: Small value to avoid division by zero.

    Returns:
        Normalized array, same shape.

    Side effects:
        None.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def get_selected_probes(
    df_public: pd.DataFrame,
    df_candidates: pd.DataFrame,
    public_emb_col: str = "embeddings",
    cand_emb_col: str = "embeddings",
    max_sentences: int = 2,
    redundancy_cosine: float = 0.92,
) -> pd.DataFrame:
    """Select the best 1-2 candidate sentences per (chunk, theme) by similarity to parent.

    For each (row_id, theme) group, ranks candidate sentences by cosine
    similarity to their parent chunk embedding and selects the top 1-2,
    enforcing a redundancy threshold between selected sentences.

    Args:
        df_public: Public chunks DataFrame with embedding column.
        df_candidates: Candidate sentences DataFrame with embedding column.
        public_emb_col: Name of the embedding column in df_public.
        cand_emb_col: Name of the embedding column in df_candidates.
        max_sentences: Maximum sentences to select per (chunk, theme).
        redundancy_cosine: Maximum cosine similarity between selected sentences.

    Returns:
        DataFrame of selected probes with sim_to_parent column added.

    Side effects:
        None.
    """
    if df_candidates.empty:
        return df_candidates.copy()

    # Build parent embedding lookup
    parent = df_public[[public_emb_col]].copy()
    parent["row_id"] = parent.index

    # Merge parent embeddings into candidates
    df = df_candidates.merge(parent, on="row_id", how="left", suffixes=("", "_parent"))
    df = df.dropna(subset=[public_emb_col, cand_emb_col])

    # Compute cosine similarity: candidate sentence vs parent chunk
    X_sent = get_l2_normalized(np.vstack(df[cand_emb_col].values))
    X_parent = get_l2_normalized(np.vstack(df[public_emb_col].values))
    df["sim_to_parent"] = np.sum(X_sent * X_parent, axis=1)

    # Select top 1-2 per (row_id, theme) with redundancy filter
    chosen_rows: list[dict] = []

    for (row_id, theme), g in df.groupby(["row_id", "theme"], sort=False):
        g = g.sort_values("sim_to_parent", ascending=False).reset_index(drop=True)

        first = g.iloc[0].to_dict()
        selected = [first]

        if max_sentences >= 2 and len(g) > 1:
            first_emb = g.iloc[0][cand_emb_col]
            first_emb = first_emb / (np.linalg.norm(first_emb) + 1e-12)

            for j in range(1, len(g)):
                cand_emb = g.iloc[j][cand_emb_col]
                cand_emb = cand_emb / (np.linalg.norm(cand_emb) + 1e-12)
                redundancy = float(np.dot(first_emb, cand_emb))
                if redundancy < redundancy_cosine:
                    selected.append(g.iloc[j].to_dict())
                    break

        # Build probe text from selected sentences
        probe_text = " ".join(s["text"] for s in selected)
        probe_sent_ids = [s["sent_id"] for s in selected]
        sim_values = [s["sim_to_parent"] for s in selected]

        out = dict(selected[0])  # Start from first selected
        out["probe_text"] = probe_text
        out["probe_sent_ids"] = probe_sent_ids
        out["probe_sim_mean"] = float(np.mean(sim_values))
        out["probe_sim_max"] = float(np.max(sim_values))
        chosen_rows.append(out)

    return pd.DataFrame(chosen_rows)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def get_deduplicated_probes(df_probes: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate probes per theme, removing exact text duplicates.

    Args:
        df_probes: DataFrame with columns including 'theme' and 'text'.

    Returns:
        Deduplicated DataFrame.

    Side effects:
        None.
    """
    parts = []
    for theme in df_probes["theme"].unique():
        theme_df = df_probes[df_probes["theme"] == theme].copy()
        theme_df = theme_df.drop_duplicates(subset=["theme", "text"])
        parts.append(theme_df)

    if not parts:
        return df_probes.iloc[0:0]  # Empty with same schema
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Embedding-based probe retrieval
# ---------------------------------------------------------------------------

def get_theme_centroids(
    anchor_embeddings: np.ndarray,
    themes: list[str],
) -> dict[str, np.ndarray]:
    """Compute mean embedding per theme from Likert anchor embeddings.

    Groups the anchor embeddings by their corresponding theme label and
    returns the L2-normalized centroid (mean vector) for each theme.

    Args:
        anchor_embeddings: Array of shape (n_anchors, d) with one embedding
            per Likert anchor phrase.
        themes: List of length n_anchors with the theme label for each
            anchor phrase (e.g., "threat", "utility", ...).

    Returns:
        Dict mapping theme name to its L2-normalized centroid vector
        of shape (d,).

    Side effects:
        None.
    """
    unique_themes = sorted(set(themes))
    centroids: dict[str, np.ndarray] = {}
    for theme in unique_themes:
        mask = np.array([t == theme for t in themes])
        mean_vec = anchor_embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-12:
            mean_vec = mean_vec / norm
        centroids[theme] = mean_vec
    return centroids


def get_sentence_theme_similarities(
    sentence_embeddings: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute cosine similarity of each sentence to each theme centroid.

    Normalizes sentence embeddings and computes dot-product similarity
    against each L2-normalized theme centroid.

    Args:
        sentence_embeddings: Array of shape (n_sentences, d).
        centroids: Dict mapping theme name to L2-normalized centroid
            vector of shape (d,).

    Returns:
        DataFrame with n_sentences rows and one column per theme,
        plus columns 'best_theme' (str) and 'best_sim' (float).

    Side effects:
        None.
    """
    X_norm = get_l2_normalized(sentence_embeddings)

    theme_names = sorted(centroids.keys())
    C = np.vstack([centroids[t] for t in theme_names])  # (n_themes, d)

    # (n_sentences, d) @ (d, n_themes) -> (n_sentences, n_themes)
    sims = X_norm @ C.T

    df = pd.DataFrame(sims, columns=pd.Index(theme_names))
    df["best_theme"] = df[theme_names].idxmax(axis=1)
    df["best_sim"] = df[theme_names].max(axis=1)
    return df


def get_embedding_based_probes(
    df_sentences: pd.DataFrame,
    similarities: pd.DataFrame,
    top_n_per_theme: int = 150,
) -> pd.DataFrame:
    """Select top-N sentences per theme by similarity to centroid.

    For each theme, takes the sentences assigned to that theme
    (highest cosine similarity) and selects the top_n_per_theme
    by similarity score. Deduplicates exact text matches within
    each theme.

    Args:
        df_sentences: DataFrame with sentence metadata. Must have at
            least a 'text' column. The index must align with
            similarities.
        similarities: DataFrame from get_sentence_theme_similarities()
            with columns for each theme, 'best_theme', and 'best_sim'.
        top_n_per_theme: Maximum sentences to keep per theme.

    Returns:
        DataFrame with columns from df_sentences plus 'theme' and
        'sim_to_centroid'. Sorted by theme then descending similarity.

    Side effects:
        None.
    """
    combined = pd.concat(
        [df_sentences.reset_index(drop=True), similarities.reset_index(drop=True)],
        axis=1,
    )

    theme_names = sorted(
        c for c in similarities.columns if c not in ("best_theme", "best_sim")
    )

    parts: list[pd.DataFrame] = []
    for theme in theme_names:
        mask = combined["best_theme"] == theme
        theme_df: pd.DataFrame = combined.loc[mask].copy()
        theme_df = theme_df.sort_values(by="best_sim", ascending=False)
        theme_df = theme_df.drop_duplicates(subset="text")
        theme_df = theme_df.head(top_n_per_theme)
        theme_df["theme"] = theme
        theme_df["sim_to_centroid"] = theme_df["best_sim"]
        parts.append(theme_df)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(["theme", "sim_to_centroid"], ascending=[True, False]).reset_index(
        drop=True
    )
