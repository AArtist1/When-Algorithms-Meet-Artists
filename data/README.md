# Data Dictionary

## Public Discourse Corpus

### `public_discourse_part1.csv` + `public_discourse_part2.csv`

The public discourse corpus, split into two files for GitHub compatibility.
Load with:

```python
from src.data_loading import load_public_discourse
df = load_public_discourse("data/")
```

**Total rows**: 891 (sorted by `article_name`, `section_id`; split at row 445)
**Source**: 131 documents spanning 2013-2025 (news articles, podcasts, panel discussions, legal filings, research papers), segmented into ~250-word chunks with 25-word overlap.

| Column | Type | Description |
|--------|------|-------------|
| `line_number` | int | Source document ID |
| `year` | int | Publication year (2013-2025) |
| `article_name` | str | Full title of source document |
| `media_type` | str | Category: article, audio, paper, video |
| `specific_type` | str | Sub-category: interview, opinion, journal, etc. |
| `lexical_diversity` | float | Type-token ratio for source document (0-1) |
| `section_id` | int | Chunk index within document |
| `text_og` | str | Original extracted text |
| `text_phrase_norm` | str | Phrase-normalized text (e.g., "AI" → "artificial_intelligence") |
| `chunk_text_norm` | str | Normalized chunk (stopwords removed, lowercased) |
| `chunk_word_count_norm` | int | Word count of normalized chunk |
| `chunk_text_clean` | str | Cleaned chunk (additional short-word removal) |
| `chunk_word_count_clean` | int | Word count of cleaned chunk |
| `chunk_text_lexical` | str | Lexical-diversity-optimized chunk |
| `chunk_word_count_lexical` | int | Word count of lexical chunk |

## Artist Perspectives

### `artist_perspectives.csv`

Artist stakeholder probes derived from Lovato et al. (2024) survey data.
Filtered to US-based, self-identified practicing artists.

**Rows**: 1,259
**Respondents**: 252 unique artists
**Dimensions**: 5 (threat, utility, ownership, transparency, compensation)

| Column | Type | Description |
|--------|------|-------------|
| `respondent_id` | int | Anonymous respondent identifier |
| `Artist` | str | Self-identifies as artist (Yes) |
| `Art_practice` | str | Primary art practice |
| `Purchase_art` | str | Purchases art (Yes/No/Not sure) |
| `Professional_artist` | str | Professional artist (Yes/No) |
| `AI_models_familiarity` | str | Familiarity level with AI art models |
| `Used_AI_art_models` | str | Has used AI art models (Yes/No) |
| `compensation` | str | Compensation stance category |
| `Age` | str | Age bracket |
| `POC` | str | Person of color (Yes/No) |
| `Gender_identity` | str | Gender identity |
| `Country` | str | Country of residence |
| `question_group` | str | Survey dimension: threat/utility/ownership/transparency/compensation |
| `perspective_text` | str | Declarative probe text (Likert-anchored) |

## Likert Anchor Phrases

### `likert_anchor_phrases.csv`

LLM-generated Likert-style anchor statements used for style-matched public probe extraction.

**Rows**: 250 (5 themes x 5 agreement levels x 10 discourse styles)

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Unique identifier (e.g., "utility_strongly_disagree_01") |
| `theme` | str | Survey dimension: utility/ownership/transparency/threat/compensation |
| `likert` | str | Agreement level: strongly_disagree to strongly_agree |
| `style` | str | Discourse style: blog_opinion, news_editorial, etc. |
| `text` | str | Generated anchor statement |

## Source Survey Data

### `lovato_survey/ai_art_surveydata_cleaned.csv`

Original cleaned survey data from Lovato et al. (2024).
See their paper for the full codebook.

**Rows**: 513
