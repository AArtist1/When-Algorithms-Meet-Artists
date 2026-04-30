# When Algorithms Meet Artists

**Semantic Compression and Stakeholder Marginalization in Public AI-Art Discourse (2013-2025)**

## Overview

This repository contains the code, data, and supplementary materials for our study of stakeholder representation in public discourse about generative AI and art. We analyze whether artist concerns achieve proportional representation in the public discourse shaping AI governance.

**Method:** We construct a semantic reference map from 125 public documents (1,736 text chunks, 2013-2025) using consensus UMAP and KMeans clustering (k=20), then project 1,259 artist survey probes from 252 US-based practicing artists into this space to quantify representational alignment.

**Key findings:**
- **Semantic compression:** 99.9% of artist concerns cluster in 4 of 20 public discourse topics
- **15 topics** (58.2% of discourse volume) contain zero artist perspective. Six of those zero-artist clusters carry artist-themed labels (e.g., AI Copyright and Legal Protection, Artist Defense Tools Against AI, AI Art Authorship and Copyright Debates) and total 532 chunks, fully 30.6% of the corpus
- **Compression operates at four distinct levels:**
  1. *Topical exclusion* — 15 of 20 clusters contain zero artist voice
  2. *Frame redirection* — all 252 ownership probes and all 252 utility probes flow into a single cluster pre-framed as a debate about aesthetic authenticity
  3. *Binary simplification* — the threat dimension splits into two clusters whose labels invert plain readings of the underlying positions ("AI is a threat" lands in a cluster labeled "AI as Creative Collaborator")
  4. *Voice collapse* — opposite positions on the same question group together by first-person practice register rather than by content
- Marginalization is **semantic, not stylistic**: style controls reduce Cramer's V from 0.740 to 0.734 (a 0.8% reduction) and JSD from 0.364 to 0.308 (a 15.4% reduction). Format differences account for a modest portion of the raw divergence; the rest is semantic.

## Repository Structure

```
When-Algorithms-Meet-Artists/
├── README.md
├── LICENSE                          # GNU GPL v3
├── pyproject.toml                   # Dependencies and project config
│
├── data/                            # Analysis-ready datasets
│   ├── README.md                    # Data dictionary
│   ├── public_discourse_clean_chunks.csv  # Clean corpus (1,736 chunks)
│   ├── public_discourse_part1.csv   # Original corpus (part 1)
│   ├── public_discourse_part2.csv   # Original corpus (part 2)
│   ├── artist_perspectives.csv      # 1,259 artist probes
│   ├── likert_anchor_phrases.csv    # 250 Likert anchor statements
│   ├── public_probes_keyword.csv    # 906 keyword-extracted probes
│   ├── public_probes_embedding.csv  # 750 embedding-extracted probes
│   └── lovato_survey/               # Source survey from Lovato et al. (2024)
│
├── src/                             # Analysis pipeline modules
│   ├── models.py                    # Pydantic data models
│   ├── data_loading.py              # Load + validate data
│   ├── text_processing.py           # Minimal text cleaning and chunking
│   ├── data_validation.py           # Pre/post-embedding validation gates
│   ├── cluster_quality.py           # Multi-source cluster quality assessment
│   ├── embeddings.py                # SentenceTransformer embedding generation
│   ├── consensus_umap.py            # Multi-seed UMAP + distance consensus
│   ├── clustering.py                # KMeans clustering
│   ├── projection.py                # MLP projection head
│   ├── compression_metrics.py       # Entropy, JSD, chi-square, permutation tests
│   ├── public_probes.py             # Public probe extraction (keyword + embedding)
│   ├── analysis.py                  # Statistical tests
│   └── visualization.py             # Figure generation
│
├── tests/                           # Test suite (66+ tests)
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_data_loading.py
│   ├── test_data_validation.py
│   ├── test_corpus_split.py
│   ├── test_compression_metrics.py
│   ├── test_public_probes.py        # Probe extraction E2E tests
│   ├── test_e2e.py
│   └── test_models.py
│
├── scripts/                         # Pipeline and analysis scripts
│   ├── final_pipeline.py            # Full pipeline (Config A)
│   ├── extract_public_probes.py     # Keyword-based probe extraction
│   ├── extract_public_probes_embedding.py  # Embedding-based extraction
│   ├── compare_probe_methods.py     # Method comparison
│   ├── validate_free_text.py        # Free-text validation
│   ├── prefix_umap_grid_search.py   # 4-stage grid search
│   └── ...
│
├── supplementary/                   # Materials referenced in manuscript
│   ├── table_S1_search_queries.md
│   ├── table_S2_full_topic_inventory.md
│   ├── table_S3_likert_design_matrix.md
│   └── consensus_umap_implementation.md
│
├── figures/                         # Generated figures and reports
│   └── final_pipeline/              # Pipeline outputs and comparisons
│
└── trace_provenance/                # AI-human collaboration provenance (TRACE v0.3)
```

## Quick Start

### Installation

```bash
git clone https://github.com/AArtist1/When-Algorithms-Meet-Artists.git
cd When-Algorithms-Meet-Artists
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest                       # all tests
pytest tests/test_public_probes.py -v  # probe extraction tests
pytest -m "not slow" -v      # skip tests requiring model downloads
```

### Running the Pipeline

```bash
python scripts/final_pipeline.py     # full analysis pipeline
python scripts/extract_public_probes.py           # keyword probes
python scripts/extract_public_probes_embedding.py  # embedding probes
python scripts/compare_probe_methods.py           # method comparison
```

## Method Summary

1. **Corpus construction:** 125 public documents segmented into 1,736 text chunks (~250 words, 25-word overlap), minimal preprocessing (URLs, HTML, boilerplate removed; no lemmatization)
2. **Embedding:** e5-large-v2 with "query: " prefix (1024-dim, L2-normalized)
3. **Consensus UMAP:** 30 seeds, distance-matrix consensus, nn=53, md=0.01, nc=5 (ARI: 0.71 vs 0.56 for coordinate averaging)
4. **Clustering:** KMeans (k=20, selected via 4-stage grid search prioritizing cluster quality) yielding 20 discourse topics across 5 macro-themes
5. **Projection:** MLP projection head (1024, 512, 256, 128, R-squared = 0.904, trustworthiness = 0.916) maps artist probes into reference space
6. **Style control:** Public probes extracted via both embedding-based (750 probes, primary) and keyword-based (906 probes, robustness) retrieval, with 14% text overlap confirming complementary coverage
7. **Analysis:** Shannon entropy, chi-square tests, Jensen-Shannon divergence, Cramer's V, permutation-based centroid distance tests
8. **2D visualization:** PCA on 5D consensus coordinates (PC1: 66.6%, PC2: 14.2%)

## Supplementary Materials

Online materials referenced in the manuscript:

- **Table S1:** [Search queries](supplementary/table_S1_search_queries.md) (17 Google Search phrases)
- **Table S2:** [Full topic inventory](supplementary/table_S2_full_topic_inventory.md) (20 topics with labels, keywords, macro-themes)
- **Table S3:** [Likert design matrix](supplementary/table_S3_likert_design_matrix.md) (5 x 5 x 10 factorial design)
- **Consensus UMAP:** [Implementation details](supplementary/consensus_umap_implementation.md) (seeds, parameters, stability validation, probe extraction)

## TRACE Provenance

This project uses TRACE (v0.3), a framework for transparent documentation of AI-human collaboration during manuscript preparation. Machine-readable provenance logs, including decision proposals, contribution attribution, and corrections, are available in `trace_provenance/`.

## Citation

[To be added after publication]

## License

GNU General Public License v3.0

## Authors

Ariya Mukherjee-Gandhi and Oliver Muellerklein (University of California, Berkeley)
