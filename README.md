# When Algorithms Meet Artists

**Semantic Compression and Stakeholder Marginalization in Public AI-Art Discourse (2013-2025)**

## Overview

This repository contains the code, data, and supplementary materials for our study of stakeholder representation in public discourse about generative AI and art. We analyze whether artist concerns achieve proportional representation in the public discourse shaping AI governance.

**Method:** We construct a semantic reference map from 131 public documents (891 text chunks, 2013-2025) using consensus UMAP and HDBSCAN clustering (22 topics), then project 1,259 artist survey probes from 252 US-based practicing artists into this space to quantify representational alignment.

**Key findings:**
- **Semantic compression:** 95.4% of artist concerns cluster in 4 of 22 public discourse topics
- **14 topics** (62.4% of discourse volume) contain zero artist perspective
- **Governance themes** (ownership, transparency) are 7x underrepresented; affective themes (threat, utility) show only 1.4x underrepresentation after style controls
- Marginalization is **semantic, not stylistic** — divergence persists after controlling for format differences

## Repository Structure

```
When-Algorithms-Meet-Artists/
├── README.md
├── LICENSE                          # GNU GPL v3
├── pyproject.toml                   # Dependencies and project config
│
├── data/                            # Analysis-ready datasets
│   ├── README.md                    # Data dictionary
│   ├── public_discourse_part1.csv   # Public corpus (first half, ~27 MB)
│   ├── public_discourse_part2.csv   # Public corpus (second half, ~27 MB)
│   ├── artist_perspectives.csv      # 1,259 artist probes
│   ├── likert_anchor_phrases.csv    # 250 Likert anchor statements
│   └── lovato_survey/               # Source survey from Lovato et al. (2024)
│
├── src/                             # Analysis pipeline modules
│   ├── models.py                    # Pydantic data models
│   ├── data_loading.py              # Load + validate data
│   ├── embeddings.py                # SentenceTransformer embedding generation
│   ├── consensus_umap.py            # Multi-seed UMAP + distance consensus
│   ├── clustering.py                # HDBSCAN + KMeans clustering
│   ├── projection.py                # MLP projection head
│   ├── analysis.py                  # Salience ratios, statistical tests
│   └── visualization.py             # Figure generation
│
├── notebooks/                       # Analysis notebooks (import from src/)
│   ├── 01_preprocessing.ipynb       # Data verification
│   ├── 02_main_analysis.ipynb       # Full pipeline
│   └── 03_consensus_umap_details.ipynb  # Methodology deep-dive
│
├── tests/                           # Comprehensive test suite (114 tests)
│   ├── conftest.py
│   ├── test_models.py               # Pydantic validation
│   ├── test_data_loading.py         # File existence + structure
│   ├── test_data_validation.py      # Content quality
│   ├── test_corpus_split.py         # Split file integrity
│   ├── test_embeddings.py           # Embedding shape + quality
│   ├── test_consensus_umap.py       # UMAP consensus pipeline
│   ├── test_clustering.py           # Clustering correctness
│   ├── test_analysis.py             # Statistical test functions
│   └── test_e2e.py                  # End-to-end integration
│
├── supplementary/                   # Materials referenced in manuscript
│   ├── table_S1_search_queries.md
│   ├── table_S2_full_topic_inventory.md
│   ├── table_S3_likert_design_matrix.md
│   └── consensus_umap_implementation.md
│
├── scripts/                         # Analysis and comparison scripts
│   ├── pca_vs_umap_2d_comparison.py
│   ├── generate_pca_manuscript_figure.py
│   └── three_way_pca_comparison.py
│
├── figures/                         # Publication figures
│
└── trace_provenance/                # AI-human collaboration provenance (TRACE v0.3.0)
    ├── README.md
    ├── project_summary.json
    └── sessions/
```

## Quick Start

### Installation

```bash
git clone https://github.com/[user]/When-Algorithms-Meet-Artists.git
cd When-Algorithms-Meet-Artists
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest                       # all 114 tests
pytest -m e2e -v             # end-to-end pipeline tests
pytest -m "not slow" -v      # skip tests requiring model downloads
```

### Running the Analysis

```bash
jupyter lab notebooks/
```

Execute notebooks in order: `01_preprocessing` → `02_main_analysis` → `03_consensus_umap_details`.

## Method Summary

1. **Corpus construction:** 131 public documents segmented into 891 text chunks (~250 words, 25-word overlap)
2. **Embedding:** sentence-transformers `e5-large-v2` (1024-dim)
3. **Consensus UMAP:** 31 seeds → pairwise distance matrices → averaged → 8D embedding (ARI: 0.71 vs 0.56 for coordinate averaging)
4. **Clustering:** HDBSCAN → 22 discourse topics across 5 macro-themes
5. **Projection:** MLP projection head (R² = 0.73, k-NN preservation = 82.4%) maps artist probes into reference space
6. **Analysis:** Salience ratios, chi-square tests, Jensen-Shannon divergence, permutation-based centroid distance tests
7. **2D visualization:** PCA on 8D consensus coordinates (PC1: 34.7%, PC2: 31.9%, total: 66.5% variance)

## Data

See `data/README.md` for the full data dictionary. The public discourse corpus is split into two files for GitHub compatibility — `src/data_loading.py` handles automatic concatenation.

## Supplementary Materials

Online materials referenced in the manuscript:

- **Table S1:** [Search queries](supplementary/table_S1_search_queries.md) — 17 Google Search phrases used for corpus construction
- **Table S2:** [Full topic inventory](supplementary/table_S2_full_topic_inventory.md) — All 22 topics with labels, keywords, macro-themes
- **Table S3:** [Likert design matrix](supplementary/table_S3_likert_design_matrix.md) — 5×5×10 factorial design with representative examples
- **Consensus UMAP:** [Implementation details](supplementary/consensus_umap_implementation.md) — Seeds, parameters, stability validation

## TRACE Provenance

This project uses [TRACE](https://trace-protocol.org) v0.3.0 for transparent documentation of AI-human collaboration during manuscript preparation. Machine-readable provenance logs — including decision proposals, contribution attribution, and corrections — are available in `trace_provenance/`.

## Citation

[To be added after publication]

## License

GNU General Public License v3.0

## Authors

[Anonymous for review]
