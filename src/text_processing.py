"""Minimal text cleaning and chunking for the embedding pipeline.

Design principles:
    - Preserve natural language structure for transformer embedding models
    - NO lemmatization (keeps "data science" not "datum science")
    - NO stopword removal (transformers need function words for context)
    - Remove only artifacts that would confuse embeddings: URLs, markdown
      links/formatting, image references, navigation menus, boilerplate
    - 250-word chunks with 25-word overlap, matching the original chunking
      strategy but applied to clean natural text

All functions are pure (no side effects) unless documented.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_article_text(text: str) -> str:
    """Apply minimal cleaning to raw article text for embedding.

    Removes:
        - Markdown links [text](url) → keeps text
        - Bare URLs (http/https/www)
        - Markdown formatting (**, *, #, ===, ---)
        - Image file references (*.jpg, *.png, etc.)
        - Markdown navigation bullets (* [text](url))
        - Excessive whitespace

    Preserves:
        - All natural language words
        - Punctuation (periods, commas, quotes, etc.)
        - Numbers and dates
        - Proper capitalization

    Args:
        text: Raw article text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove markdown image links: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", text)

    # Convert markdown links to just the text: [text](url) → text
    text = re.sub(r"\[([^\]]*)\]\([^\)]+\)", r"\1", text)

    # Remove bare URLs
    text = re.sub(r"https?://[^\s\)]+", " ", text)
    text = re.sub(r"www\.[^\s\)]+", " ", text)

    # Remove image file references (word.jpg, word.png, etc.)
    text = re.sub(r"\b\S+\.(?:jpg|jpeg|png|gif|svg|webp)\b", " ", text, flags=re.IGNORECASE)

    # Remove markdown heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown horizontal rules and emphasis markers
    text = re.sub(r"^[=\-]{3,}\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^\*]+)\*{1,3}", r"\1", text)  # bold/italic → text

    # Remove markdown bullet navigation lines (lines starting with * [ )
    text = re.sub(r"^\s*\*\s+\[.*$", " ", text, flags=re.MULTILINE)

    # Remove standalone markdown bullets at line start
    text = re.sub(r"^\s*[\*\-]\s+", " ", text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", " ", text)

    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"(?i)skip to content",
        r"(?i)toggle child menu",
        r"(?i)primary menu",
        r"(?i)sign up for (?:our|the) newsletter",
        r"(?i)subscribe to (?:our|the)",
        r"(?i)Previous article Previous article\w*",
        r"(?i)Next article Next article\w*",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, " ", text)

    # Fix camelCase/PascalCase merges from removed punctuation
    # e.g., "articlepoliticsRunaway" → "article politics Runaway"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Fix known compound artifacts (no space after period/semicolon removal)
    text = re.sub(r"([a-z]{3,})([A-Z][a-z])", r"\1 \2", text)

    # Collapse multiple whitespace (spaces, tabs, newlines) to single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 250,
    overlap: int = 25,
) -> list[str]:
    """Split text into word-level chunks with overlap.

    Args:
        text: Input text string.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk strings. Empty list if text is empty.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    words = text.split()
    if len(words) == 0:
        return []

    step = chunk_size - overlap  # 225 by default
    chunks = []

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Stop if we've reached the end
        if end >= len(words):
            break

    return chunks


def process_article(
    text: str,
    article_name: str,
    year: int,
    media_type: str,
    specific_type: str,
    lexical_diversity: float,
    line_number: int,
    chunk_size: int = 250,
    overlap: int = 25,
    min_words: int = 30,
) -> list[dict]:
    """Clean and chunk a single article into records.

    Args:
        text: Raw article text (text_og).
        article_name: Article identifier.
        year: Publication year.
        media_type: Article media type.
        specific_type: Article specific type.
        lexical_diversity: Article lexical diversity score.
        line_number: Original line number.
        chunk_size: Words per chunk.
        overlap: Overlap words.
        min_words: Minimum words for a chunk to be kept.

    Returns:
        List of record dicts, one per chunk.
    """
    cleaned = clean_article_text(text)
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)

    records = []
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        if word_count < min_words:
            continue
        records.append({
            "line_number": line_number,
            "year": year,
            "article_name": article_name,
            "media_type": media_type.strip(),
            "specific_type": specific_type.strip(),
            "lexical_diversity": lexical_diversity,
            "section_id": i + 1,
            "chunk_text": chunk,
            "chunk_word_count": word_count,
        })

    return records
