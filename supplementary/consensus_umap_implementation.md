# Consensus UMAP Implementation Details

## Seed Pool and Multi-Run Generation

31 UMAP projections were generated using different random seeds with the following parameters:

| Parameter | Value |
|---|---|
| `n_components` | 8 |
| `n_neighbors` | 27 |
| `min_dist` | 0.1 |
| `metric` | cosine |
| Number of seeds | 31 |

Seeds were selected to maximize diversity: `[137, 85, 127, 59, 195, 243, 170, 77, 186, 79, 69, 42, 240, 105, 199, 91, 151, 195, 77, 82, 177, 234, 46, 101, 34, 175, 108, 81, 176, 241, 20]`.

## Distance-Matrix Consensus Method

For each seeded UMAP projection, the pairwise Euclidean distance matrix in 8D space was computed. These 31 matrices were element-wise averaged to produce a consensus distance matrix. A final UMAP was fit using this consensus distance structure as the precomputed distance input (`metric="precomputed"`).

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
| Architecture | 1024 → 512 → 128 → 8 |
| Activation | ReLU |
| Regularization | L2 (alpha = 0.001) |
| Training | StandardScaler on inputs and outputs; early stopping |
| Validation R² | 0.73 |
| k-NN neighborhood preservation | 82.4% (k = 72) |

## 2D Visualization: PCA on Consensus Coordinates

For 2D visualization, we apply PCA to the 8-dimensional consensus coordinates rather than running a second UMAP reduction. This avoids compounding nonlinear distortions ("double-UMAP") and provides deterministic, interpretable axes.

| Metric | UMAP 8D→2D | PCA 8D→2D |
|---|---|---|
| Variance explained (PC1+PC2) | N/A | **67.4%** |
| Trustworthiness (k=15) | 0.988 | 0.982 |
| k-NN preservation (k=15) | 0.562 | **0.731** |
| Silhouette (22 clusters) | 0.301 | **0.423** |
| Public-Artist centroid distance | 2.553 | **2.570** |

PCA outperforms UMAP on k-NN preservation (+30%), silhouette score (+41%), and public-artist centroid distance, while trustworthiness is marginally lower (0.006 difference). PCA captures 67.4% of the variance in the first two components (PC1: 34.7%, PC2: 31.9%).

The comparison analysis is in `scripts/pca_vs_umap_2d_comparison.py`.

## 3-Way Comparison: Public Discourse vs. Artist Probes vs. Public Probes

To validate that the observed divergence between artist probes and public discourse is semantic rather than stylistic (H2), we compare all three corpora in the consensus space:

1. **Public discourse** (891 chunks) — the reference map
2. **Artist probes** (1,259) — survey-derived stakeholder perspectives
3. **Public probes** (250 Likert anchor phrases) — style-matched control statements covering the same 5 themes at 5 agreement levels in 10 discourse styles

### Pairwise Divergence Metrics

| Comparison | JSD | Cramér's V | Centroid Distance (8D) | Same-Source kNN (k=15) |
|---|---|---|---|---|
| Public vs Artist | 0.310 | 0.716 | 2.685 | 0.986 |
| Public vs Public Probes | 0.237 | 0.539 | 2.248 | 0.926 |
| Artist vs Public Probes | **0.056** | 0.330 | **1.685** | 0.946 |

**Key finding:** Artist probes and public probes are much closer to each other (JSD = 0.056) than either is to the full public discourse (JSD > 0.23). This confirms that when both corpora express the same themes in the same declarative format, they converge — the residual gap between public discourse and either probe set is semantic, not stylistic.

### Cluster Concentration

| Dataset | Top 4 clusters | Clusters with 0 probes | % of discourse absent |
|---|---|---|---|
| Artist probes | 81.3% | 13 of 22 | 54.8% |
| Public probes | 73.2% | 8 of 22 | 25.6% |

Public probes spread to more clusters (14 vs 9) because the Likert design covers the same themes in 10 different discourse styles, giving them broader stylistic reach. Yet they still avoid 8 clusters — confirming that even style-matched text on artist-relevant themes does not permeate the institutional, technical, and philosophical regions of public discourse.

### Theme-Specific Salience Ratios (Max per Theme)

| Theme | Artist Probes | Public Probes |
|---|---|---|
| Compensation | 6.9× | 8.1× |
| Ownership | 11.5× | 5.8× |
| Transparency | 13.2× | 8.8× |
| Threat | 12.1× | 5.6× |
| Utility | 12.4× | 5.0× |

The 3-way comparison analysis is in `scripts/three_way_pca_comparison.py`. Visualizations are in `figures/three_way_comparison/`.

## Implementation

See `src/consensus_umap.py` and `src/projection.py` for the implementation code. The full pipeline is demonstrated in `notebooks/03_consensus_umap_details.ipynb`.
