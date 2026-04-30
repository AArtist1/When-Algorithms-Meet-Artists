# Consensus UMAP Implementation Details

## Seed Pool and Multi-Run Generation

30 UMAP projections were generated using different random seeds with the following parameters:

| Parameter | Value |
|---|---|
| `n_components` | 5 |
| `n_neighbors` | 53 |
| `min_dist` | 0.01 |
| `metric` | cosine |
| Number of seeds | 30 |

Seeds: `[137, 85, 127, 59, 195, 243, 170, 77, 186, 79, 69, 42, 240, 105, 199, 91, 151, 82, 177, 234, 46, 101, 34, 175, 108, 81, 176, 241, 20, 53]`.

Parameters were selected via a 4-stage grid search (497 total configurations across Stages 1, 1b, 1c, and 2) on the 1,736-chunk corpus with prefix embeddings. Config A (nn=53, md=0.01, nc=5) was selected based on highest consensus silhouette (0.657), fewest single-article clusters (3 of 20), and highest valid topic rate (85%). See `scripts/prefix_umap_grid_search.py` for the full search protocol.

## Distance-Matrix Consensus Method

For each seeded UMAP projection, the pairwise Euclidean distance matrix in 5D space was computed. These 30 matrices were element-wise averaged to produce a consensus distance matrix. A final UMAP was fit using this consensus distance structure as the precomputed distance input (`metric="precomputed"`).

This approach is rotation-invariant (unlike coordinate averaging) because pairwise distances do not depend on the arbitrary rotation of each UMAP embedding.

## Comparison: Distance Averaging vs. Coordinate Averaging

| Method | Mean ARI (seed vs. consensus) |
|---|---|
| Naive coordinate averaging (Procrustes) | 0.56 |
| **Distance-matrix consensus** | **0.71** |
| Relative improvement | ~27% |

## Projection Head Architecture

To project new data (artist probes, public probes) into the consensus space without distorting the reference map, we trained an MLP projection head:

| Component | Specification |
|---|---|
| Architecture | 1024, 512, 256, 128 (4 layers) |
| Output | 5 dimensions (matching consensus UMAP) |
| Activation | ReLU |
| Regularization | L2 (alpha = 0.0001) |
| Training | StandardScaler on inputs and outputs, adam optimizer, early stopping |
| Max iterations | 1000 |
| Validation R-squared | 0.904 |
| Trustworthiness (k=15) | 0.916 |

The projection head was validated via sensitivity analysis across 28 architectural configurations. The top-4 cluster concentration of artist probes exceeds 87% across all tested configurations.

## 2D Visualization: PCA on Consensus Coordinates

For 2D visualization, we apply PCA to the 5-dimensional consensus coordinates rather than running a second UMAP reduction. This avoids compounding nonlinear distortions and provides deterministic, interpretable axes.

PCA captures 80.8% of the variance in the first two components (PC1: 66.6%, PC2: 14.2%).

The comparison analysis is in `scripts/pca_vs_umap_2d_comparison.py`.

## Public Probe Extraction

To control for stylistic differences between survey statements and media discourse, we extracted style-matched public probes from the clean corpus using two complementary methods:

**Keyword-based retrieval:** Using 250 synthetic Likert anchor phrases as keyword sources, sentences matching at least 4 theme-relevant keywords were extracted and ranked by cosine similarity to their parent chunk embedding. This produced 906 public probes.

**Embedding-based retrieval (primary method):** Theme centroids were computed from the 250 Likert anchor embeddings. All 12,313 sentences in the corpus were embedded and ranked by cosine similarity to the nearest theme centroid. The top 150 per theme were selected, yielding 750 public probes.

The two methods share 14% text overlap (101 sentences), confirming they are complementary: keyword matching finds explicit mentions while embedding retrieval captures semantically similar sentences that lack keyword markers.

The 250 Likert anchors served only as retrieval queries and were discarded after extraction.

## H2 Style Control Results

| Comparison | JSD | Cramer's V | Centroid Distance (5D) |
|---|---|---|---|
| Public vs Artist (raw) | 0.364 | 0.740 | 2.003 |
| Public Probes vs Artist (style-controlled) | 0.245 | 0.648 | 2.699 |

The style control reduces Cramer's V by 12.4% (0.740 to 0.648) and JSD by 33% (0.364 to 0.245), confirming that format differences contribute to but do not account for the observed divergence.

## H3 Differential Compression

| Theme | Frames | Entropy (norm) | Topics (any / 5+) | Articles | FCR |
|---|---|---|---|---|---|
| Ownership | 24 | 0.000 | 1 / 1 | 34 (27%) | 24.0 |
| Transparency | 3 | 0.137 | 2 / 2 | 46 (37%) | 1.5 |
| Threat | 3 | 0.148 | 2 / 2 | 46 (37%) | 1.5 |
| Utility | 3 | 0.000 | 1 / 1 | 34 (27%) | 3.0 |
| Compensation | 37 | 0.295 | 5 / 4 | 52 (42%) | 7.4 |

Entropy is normalized (0 = all probes in one topic, 1 = uniform across all 20 topics). FCR = frame-to-topic compression ratio.

## Free-Text Validation

To validate probe construction, 38 free-text responses from the Lovato et al. (2024) survey (Fair_compensation_tax_text field) were embedded with e5-large-v2 and projected into the consensus space. 84% of free-text responses land in the same cluster region as the corresponding templated compensation probes, confirming that the probe templates capture the same semantic territory as organic artist language.

## Configuration Summary

| Parameter | Value |
|---|---|
| Corpus | 1,736 chunks from 125 articles |
| Embedding | e5-large-v2, "query: " prefix |
| UMAP | nn=53, md=0.01, nc=5, 30 seeds |
| Clustering | KMeans, k=20 |
| Projection | MLP (1024, 512, 256, 128), R-squared=0.904 |
| Public probes | 750 (embedding, primary) + 906 (keyword, robustness check) |

## Implementation

See `src/consensus_umap.py` and `src/projection.py` for the implementation code. The full pipeline is in `scripts/final_pipeline.py`.
