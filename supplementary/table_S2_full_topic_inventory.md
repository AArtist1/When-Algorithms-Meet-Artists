# Table S2: Full Topic Inventory

20 topics identified via KMeans clustering (k=20, selected by 4-stage hyperparameter validation with cluster quality prioritization) of the consensus UMAP embedding. Labels assigned by multi-annotator consensus (2 human coders + 4 LLMs: Claude Opus, Claude Sonnet, GPT-5-mini, GPT-5-nano). Combined confidence scores reflect inter-annotator agreement.

## Macro-Thematic Groupings

| Macro-theme | Share of Corpus | Topics |
|---|---|---|
| Philosophy of Creativity | 38.1% | 1, 4, 11, 13, 17 |
| Practice and Pedagogy | 34.8% | 3, 7, 14, 16 |
| Technical Genealogy | 14.7% | 2, 5, 15, 19 |
| Governance and Rights | 7.5% | 8, 10, 18 |
| Institutions and Markets | 4.9% | 0, 6, 9, 12 |

## Full Topic Inventory

| Topic | Label | Top Keywords | Macro-theme | Confidence |
|---|---|---|---|---|
| 0 | Decentralized Infrastructure for Art Ecosystems | decentralized, technologies, cultural, statements | Institutions and Markets | 0.4 |
| 1 | AI as Creative Collaborator | ai, art, human, data | Philosophy of Creativity | 0.4 |
| 2 | Machine Learning Art Theory and Practice | MIT, machine learning, Microsoft | Technical Genealogy | 0.4 |
| 3 | Personal Reflections on AI and Art | know, ai, art, really, yeah | Practice and Pedagogy | 0.4 |
| 4 | Mental Models and Abstract Thought | sort, kind, mental images, language | Philosophy of Creativity | 0.6 |
| 5 | Harold Cohen and AARON Legacy | Cohen, AARON, computer, creativity, 2010 | Technical Genealogy | 0.5 |
| 6 | Future of Arts Journalism and Museums | museums, publishing, writing, journalism | Institutions and Markets | 0.4 |
| 7 | Conversational Reflections on Art Practice | know, art, really, kind, lot | Practice and Pedagogy | 0.4 |
| 8 | AI Copyright and Legal Protection | copyright, coders, software, law, protection | Governance and Rights | 0.6 |
| 9 | Digital Art Exhibition and Display | internet, MoMA, exhibition | Institutions and Markets | 0.5 |
| 10 | Artist Defense Tools Against AI | Nightshade, Glaze, tools, Rachel | Governance and Rights | 0.5 |
| 11 | AI Authorship and Creative Agency | Chung, authorship, agency | Philosophy of Creativity | 0.6 |
| 12 | Media Coverage and AI Panic | ai, effective altruism, media, humanity | Institutions and Markets | 0.4 |
| 13 | AI Art Authenticity and Human Creativity | ai, art, artists, image, human | Philosophy of Creativity | 0.6 |
| 14 | Informal AI Creative Tool Discourse | gonna, students, Adobe, Firefly | Practice and Pedagogy | 0.4 |
| 15 | Deep Dream and Neural Network Visualization | Mordvintsev, neural, deep, Deep Dream | Technical Genealogy | 0.6 |
| 16 | Artist Reflections on Technology | know, art, really, artists | Practice and Pedagogy | 0.3 |
| 17 | Artist-Centered AI Design and Ethics | fi, prompts, design, ethics | Philosophy of Creativity | 0.6 |
| 18 | AI Art Authorship and Copyright Debates | Midjourney, Allen, copyright | Governance and Rights | 0.6 |
| 19 | Generative Art History and Pioneers | ai art, generative art, Barrat, computer art | Technical Genealogy | 0.4 |

## Configuration

- Corpus: 1,736 chunks from 125 articles (2013-2025)
- Embedding: e5-large-v2 with "query: " prefix, L2-normalized
- UMAP: consensus from 30 seeds, n_neighbors=53, min_dist=0.01, n_components=5
- Clustering: KMeans with k=20
- Minimum cluster size: 10 chunks
- Labeling: 4 LLMs independently + 2 human coders, consensus voting
