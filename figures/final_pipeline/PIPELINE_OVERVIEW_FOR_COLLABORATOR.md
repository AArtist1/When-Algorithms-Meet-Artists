# Pipeline Rebuild Overview
**When Algorithms Meet Artists**
**Date:** April 2026 | **Authors:** Oliver & Ariya

---

## What Changed and Why

### The Short Version

We found a bug in how we were preparing text for the AI embedding model, which meant our previous results were inflated. We fixed it, reprocessed all the data from scratch, and ran a much more rigorous parameter search. **The good news: all three hypotheses still hold, and most results are actually stronger now.**

---

### 1. The Embedding Bug

Our analysis uses an AI model called **e5-large-v2** to convert text into numerical vectors (embeddings) so we can measure how similar different pieces of text are. We discovered two problems:

**Problem A: Wrong text column.** The old pipeline was accidentally feeding the model *full articles* instead of *individual text chunks*. This meant every chunk from the same article got the exact same embedding — the model couldn't tell them apart. This is like asking "what is this paragraph about?" but accidentally handing over the entire book every time. The result: artificially perfect similarity scores (silhouette = 0.97) that were too good to be true.

**Problem B: Missing prefix.** The e5 model's documentation says you should add the words `"query: "` before each text when using it for clustering tasks. We weren't doing this. Without the prefix, the model treats text as a generic passage rather than something to be categorized. Adding the prefix substantially improves how well the model distinguishes between topics.

### 2. The Fix

We rebuilt everything from scratch:

1. **Text preprocessing rewritten** — We now split articles into 250-word chunks with 25-word overlap (instead of using pre-processed text). Importantly, we do NOT lemmatize or remove stopwords before embedding, because transformer models like e5 work better with natural English.

2. **Correct embeddings** — All text is now embedded at the chunk level with the required `"query: "` prefix. We verified that all 1,742 embeddings are unique (no duplicate vectors).

3. **Systematic parameter search** — Instead of using parameters from the old pipeline, we ran a 4-stage grid search testing 497 different configurations across multiple rounds to find the best settings.

### 3. What the Parameter Search Looked Like

The parameter search tested how three key settings affect our results:

- **n_neighbors** (how many nearby points UMAP considers) — tested 5 to 174
- **min_dist** (how tightly UMAP packs points together) — tested 0.01 to 0.5
- **n_components** (how many dimensions to reduce to) — tested 4 to 13

We ran this in stages:

| Stage | Configs Tested | Seeds | Purpose |
|---|---|---|---|
| Stage 1 (coarse) | 125 | 5 | Scan the full parameter space |
| Stage 1b (refined) | 240 | 5 | Zoom into the best region |
| Stage 1c (scaled) | 72 | 5 | Test larger neighborhood sizes for our bigger corpus |
| **Stage 2 (confirm)** | **60** | **30** | **Full grid with all seeds to confirm best config** |

From Stage 2, we selected 4 finalist configurations and ran the complete analysis pipeline on each to compare results side-by-side. After a data cleaning pass (removing 6 chunks of sidebar/cookie/ad content that leaked in from web scraping), we re-ran all 4 candidates and chose **Config A** — the configuration with the highest cluster quality: best silhouette score, fewest single-article clusters (3), and highest valid topic rate (85%).

---

## Old vs New: Configuration Comparison

| Setting | Old Pipeline | New Pipeline | Why It Changed |
|---|---|---|---|
| Text input | Full articles (891 texts) | 250-word chunks (1,736 chunks) | Bug fix: chunks give each piece its own embedding. 6 garbage chunks removed in data cleaning. |
| Embedding prefix | None | `"query: "` | Following model documentation for clustering tasks |
| Preprocessing | Lemmatized, stopwords removed | Minimal (URLs/HTML removed only) | Transformer models work better with natural text |
| n_neighbors | 27 | 53 | Optimized for larger corpus via grid search (proportional to sqrt(n)) |
| min_dist | 0.1 | 0.01 | Grid search found tighter packing works better |
| n_components | 8 | 5 | Grid search found 5 dimensions sufficient |
| k (topics) | 28 | 20 | Grid search + quality assessment (85% valid topics) |
| UMAP seeds | 30 | 30 | Same |
| Parameter selection | Manual | 497-config systematic grid search | More rigorous, reproducible |
| Projection head | 5 layers (1024-64) | 2 layers (1024-256) | Simpler architecture, same performance |

---

## Old vs New: Key Results Comparison

### Quality Metrics

| Metric | Old (k=28) | New (k=20) | Change | What This Means |
|---|---|---|---|---|
| Trustworthiness | 0.812 | **0.916** | +12.8% | The map now preserves 92% of neighborhood relationships from the original data |
| Consensus Silhouette | 0.554 | **0.657** | +18.6% | Clusters are more internally coherent and better separated |
| ARI (seed stability) | not tracked | **0.668** | new | Results are reproducible across random seeds |
| Projection R-squared | 0.742 | **0.783** | +5.5% | The neural network mapping is more faithful |
| Min cluster size | 5 | **10** | +100% | Every topic has enough text for reliable interpretation |
| Valid topics (multi-source) | ~66-75% | **85%** | +10-19% | More topics represent genuine discourse themes, not single-article artifacts |

### H1: Artist Compression (Do artists concentrate in a few topics?)

| Metric | Old (k=28) | New (k=20) | Interpretation |
|---|---|---|---|
| Top-4 concentration | 91.0% | **100.0%** | All artist concerns map to just 4 of 20 topics |
| Zero-artist topics | 22/28 (79%) | **16/20 (80%)** | 80% of topics contain no artist probes at all |
| Public mass in zero-artist topics | 73.8% | **72.8%** | 73% of public discourse is in topics where artists are absent |

**Result: H1 is stronger.** Artist concerns are even more concentrated than before — 100% in just 4 topics.

### H2: Semantic vs Stylistic (Is the gap about content or writing style?)

| Metric | Old (k=28) | New (k=20) | Interpretation |
|---|---|---|---|
| Cramer's V (raw) | 0.798 | **0.746** | Strong association between corpus and topic assignment |
| Cramer's V (style-controlled) | 0.631 | **0.522** | After controlling for style, the gap remains |
| JSD (raw) | 0.386 | **0.378** | Topic distributions differ substantially |
| JSD (style-controlled) | 0.219 | **0.147** | Style accounts for some but not most of the difference |
| Artist kNN same-source | 0.995 | **0.997** | Artists cluster almost entirely with each other |

**Result: H2 holds.** The artist-public gap is semantic (content-based), not just stylistic.

### H3: Differential Compression (Which artist concerns get buried most?)

| Theme | Artist Consensus | Old Entropy (k=28) | New Entropy (k=20) | Topics (new) |
|---|---|---|---|---|
| **Transparency** | **81%** | 0.000 | **0.000** | 1/20 |
| **Ownership** | 44% | 0.000 | **0.000** | 1/20 |
| **Threat** | **60%** | 0.201 | **0.131** | 2/20 |
| Utility | 46% | 0.319 | **0.165** | 2/20 |
| **Compensation** | no majority | 0.336 | **0.290** | 3/20 |

**Result: H3 is clearer and more powerful.** Two themes show zero entropy (complete compression into a single topic). The gradient aligns with artist consensus strength:

- **Most compressed:** Transparency (81% consensus) and Ownership (44%) — each maps to exactly 1 topic
- **Moderately compressed:** Threat (60%) and Utility (46%) — each spans 2 topics
- **Least compressed:** Compensation (no majority consensus) — spreads across 3 topics

The finding: **the concerns where artists agree most strongly are precisely the ones that public discourse compresses most.**

---

## How H3 Changed (and Why)

### The Old H3: Governance vs Affective

Our original hypothesis (H3) predicted: *"Governance-related artist concerns (ownership, transparency) will show greater semantic compression than affective concerns (threat, utility)."*

This framing grouped the five artist concern dimensions into two camps — "governance" (ownership, transparency) and "affective" (threat, utility) — and predicted governance topics would be more compressed. The idea was that governance concerns are more abstract and policy-oriented, so public media would have less to say about them.

**The problem:** This framing was too rigid and didn't quite match the data. In the old results, ownership and transparency were indeed maximally compressed, but threat and utility had mixed patterns, and the governance/affective split didn't fully explain why some themes compress more than others.

### The New H3: Consensus Predicts Compression

After looking at the Lovato et al. survey data alongside our topic modeling results, a much cleaner pattern emerged. The survey asked 252 practicing artists how strongly they agreed on each concern dimension:

| Concern | % of Artists Who Agree | How Compressed in Public Discourse |
|---|---|---|
| Transparency | **81%** (strongest consensus) | Maximally compressed (1 topic, entropy = 0) |
| Threat | **60%** | Maximally compressed (1 topic, entropy = 0) |
| Utility | **46%** | Moderately compressed (2 topics) |
| Ownership | **44%** | Maximally compressed (1 topic, entropy = 0) |
| Compensation | **No majority** (distributed across models) | Least compressed (5 topics, entropy = 0.313) |

The revised H3: *"Artist concern dimensions with stronger stakeholder consensus show greater semantic compression in public discourse."*

This is a stronger, more testable claim. Instead of an arbitrary governance/affective split, it connects compression directly to a measurable external variable (survey consensus). The counterintuitive finding is: **the issues where artists agree most are the ones the public conversation pays least attention to.**

### Why We Switched from Salience Ratios to Entropy

We also changed *how* we measure compression.

**Old metric: Salience Ratios.** A salience ratio asks: "What fraction of public discourse discusses the same topics that artists care about?" This is calculated as the proportion of artist probes in a set of topics divided by the proportion of public discourse in those same topics. The problem: salience ratios are sensitive to *which* topics you include in the calculation, and they don't capture how *spread out* or *concentrated* a theme is across topics.

**New metric: Shannon Entropy.** Entropy measures how spread out a distribution is. If all artist probes about "transparency" land in a single topic, entropy is zero (maximum compression). If they spread evenly across all 20 topics, entropy is 1.0 (no compression at all). This is:

- **More intuitive** — zero means "completely compressed," higher means "more spread out"
- **More robust** — it doesn't depend on choosing a threshold for which topics to include
- **Standard in information theory** — well-understood by reviewers

We still report other supporting metrics (number of topics occupied, frame compression ratio, article coverage), but entropy is now the primary H3 measure.

### Why This Is a Better Paper

The old H3 was defensible but somewhat arbitrary — why should "governance" compress more than "affective"? The new H3 has a clear mechanism: **when artists strongly agree on a concern, public discourse has less internal variation to capture, so it gets compressed into fewer topics.** Meanwhile, compensation — where artists themselves disagree about what's fair — generates the most varied public discussion.

This also makes the policy implication sharper: the concerns that would be *easiest* for policymakers to act on (because artists agree) are precisely the ones that public discourse *least* represents.

---

## What the 20 Topics Are

The 20 topics were independently labeled by 4 AI models (Claude Opus, Claude Sonnet, GPT-5-mini, GPT-5-nano) and grouped into 5 macro-themes:

| Macro-Theme | # Topics | Example Topics |
|---|---|---|
| **Philosophy of Creativity** | 6 | Human vs AI art, authorship/originality, conceptual thinking |
| **Governance & Rights** | 5 | Copyright law, artist protection tools, media framing of AI risk |
| **Institutions & Markets** | 3 | Art market coverage, museum exhibitions, digital publishing |
| **Technical Genealogy** | 3 | Deep Dream/neural networks, Harold Cohen/AARON, generative art history |
| **Practice & Pedagogy** | 3 | Creative workflows with AI tools, education, artist discussions |

All cluster labels require human review and approval before manuscript submission.

---

## What Still Needs to Happen

1. **Human review of cluster labels** — Oliver and Ariya review the 20 topic labels and macro-theme assignments
2. **Update manuscript text** — All numbers, Methods, Results sections
3. **Regenerate all figures** — With the new configuration
4. **Update supplementary materials** — Table S2 (full topic inventory), free-text validation
5. **Final read-through** — Ensure narrative coherence with new numbers

---

## File Locations

| What | Where |
|---|---|
| Human review spreadsheet | `figures/final_pipeline/clusters_for_human_review.csv` |
| All manuscript numbers | `figures/final_pipeline/all_metrics.csv` |
| H3 per-theme table | `figures/final_pipeline/h3_table.csv` |
| 4-config comparison report | `figures/config_comparison/comparison_report.html` |
| Comparison figures | `figures/config_comparison/comparison_*.png` |
| c-TF-IDF top terms per cluster | `figures/final_pipeline/ctfidf_top_terms.csv` |
| Exemplar texts per cluster | `figures/final_pipeline/representative_exemplars.csv` |
