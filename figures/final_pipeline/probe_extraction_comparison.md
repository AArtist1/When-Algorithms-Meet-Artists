# Public Probe Extraction: Before vs After

**Date:** April 9, 2026
**Change:** Keyword-based probe extraction updated from old 891-chunk corpus to new 1,736-chunk clean corpus

---

## Summary

| Metric | Old Pipeline | New Pipeline | Change |
|--------|-------------|-------------|--------|
| Corpus chunks | 891 | 1,736 | +95% |
| Source articles | 131 | 125 | -6 (duplicates removed) |
| Total probes | 557 | 906 | +63% |
| Text column | text_phrase_norm (lemmatized) | chunk_text (natural English) |  |
| Embedding prefix | None | "query: " | Fixed |
| Data alignment | Misaligned (891 chunks vs 1,736 embeddings) | Matched (1,736 x 1,024) | Fixed |

## Theme Distribution

| Theme | Old | New | Change | % of New Total |
|-------|-----|-----|--------|---------------|
| compensation | 144 | 297 | +153 (+106%) | 32.8% |
| threat | 106 | 190 | +84 (+79%) | 21.0% |
| utility | 146 | 189 | +43 (+29%) | 20.9% |
| transparency | 91 | 131 | +40 (+44%) | 14.5% |
| ownership | 70 | 99 | +29 (+41%) | 10.9% |
| **TOTAL** | **557** | **906** | **+349 (+63%)** | **100%** |

## Key Observations

1. **All themes gained probes.** The larger corpus provides more extraction surface.
2. **Ownership improved.** From 70 to 99 probes (+41%). This is critical because ownership is the most compressed theme (H3).
3. **Compensation saw the largest gain.** +153 probes (+106%), likely because the natural-language chunks preserve compensation-related vocabulary that lemmatization was stripping.
4. **The distribution is still uneven.** Compensation has 3x more probes than ownership. This reflects genuine differences in how often these themes appear in public discourse, which is itself evidence for the study's claims.

## Pipeline Diagram

```
  Keyword-Based Extraction Pipeline (New)
  ========================================

  data/public_discourse_clean_chunks.csv
           |
           | 1,736 chunks (natural English)
           v
  +------------------+     +---------------------------+
  | Split into       |     | data/likert_anchor_phrases |
  | sentences        |     | 250 Likert phrases        |
  | (5-50 words)     |     +---------------------------+
  +------------------+              |
           |                        | Extract keywords
           v                        v
  +-------------------------------------------+
  | Keyword matching: sentence vs anchors      |
  | (min 4 keyword hits per sentence)          |
  +-------------------------------------------+
           |
           | 1,114 candidates
           v
  +-------------------------------------------+
  | Embed candidates with e5-large-v2          |
  | + "query: " prefix                         |
  +-------------------------------------------+
           |
           v
  +-------------------------------------------+
  | Rank by cosine similarity to parent chunk  |
  | Select top 1-2 per (chunk, theme)          |
  | Redundancy filter (cosine < 0.92)          |
  +-------------------------------------------+
           |
           v
  +-------------------------------------------+
  | Deduplicate within theme                   |
  +-------------------------------------------+
           |
           | 906 probes
           v
  +-------------------------------------------+
  | Re-embed final probes with prefix          |
  +-------------------------------------------+
           |
           v
  data/public_probes_keyword.csv (906 rows)
  figures/final_pipeline/keyword_probes_embeddings.npy (906, 1024)
```

## Files

| File | Description |
|------|-------------|
| `data/public_probes_keyword.csv` | 906 keyword-extracted probes |
| `figures/final_pipeline/keyword_probes_embeddings.npy` | Probe embeddings (906, 1024) |
| `figures/final_pipeline/keyword_probe_comparison.json` | Structured comparison data |
| `figures/final_pipeline/probe_extraction_report.html` | Visual HTML comparison report |
