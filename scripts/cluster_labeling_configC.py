"""Cluster labeling for Config C (nn=45, md=0.03, nc=8, k=20).

Pipeline:
    1. Load Config C consensus coords + labels from config_comparison/C/
    2. Build c-TF-IDF matrix (1-4 grams) per cluster
    3. Sample representative exemplars (core + boundary + diverse)
    4. Compute within vs across cluster similarity
    5. Send c-TF-IDF + exemplars to 4 LLMs (Claude Opus, Claude Sonnet, GPT-5-mini, GPT-5-nano)
    6. Compute agreement metrics and macro-theme consensus
    7. Generate human review CSV

Output:
    figures/final_pipeline/ctfidf_top_terms.csv
    figures/final_pipeline/representative_exemplars.csv
    figures/final_pipeline/within_vs_across_similarity.csv
    figures/final_pipeline/quad_llm_labels.csv
    figures/final_pipeline/macro_theme_fit.csv
    figures/final_pipeline/clusters_for_human_review.csv

Side effects:
    Writes files. Calls Anthropic + OpenAI APIs. Prints report.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv("/Users/echoes/Documents/Berkeley/Research/corp-sus-report-extractor/.env")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "figures" / "config_comparison" / "A"
OUTPUT_DIR = ROOT / "figures" / "final_pipeline"

K = 20

MACRO_THEMES = [
    "Institutions & Markets",
    "Governance & Rights",
    "Technical Genealogy",
    "Practice & Pedagogy",
    "Philosophy of Creativity",
]

MODELS = {
    "claude_opus": {"provider": "anthropic", "model": "claude-opus-4-20250514"},
    "claude_sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "gpt5_mini": {"provider": "openai", "model": "gpt-5-mini"},
    "gpt5_nano": {"provider": "openai", "model": "gpt-5-nano"},
}


# ---------------------------------------------------------------------------
# c-TF-IDF
# ---------------------------------------------------------------------------

def build_ctfidf_matrix(
    texts: list[str],
    labels: np.ndarray,
    ngram_range: tuple[int, int] = (1, 4),
    min_df: int = 2,
    max_df: float = 0.95,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build c-TF-IDF matrix. Returns (matrix [K,V], vocab, unique_labels)."""
    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)

    class_docs = []
    class_sizes = []
    for c in unique_labels:
        mask = labels == c
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        class_docs.append(" ".join(cluster_texts))
        class_sizes.append(int(mask.sum()))

    effective_min_df = min(min_df, max(1, n_clusters // 2))

    vec = CountVectorizer(
        ngram_range=ngram_range,
        min_df=effective_min_df,
        max_df=max_df,
        stop_words="english",
    )
    raw_counts = vec.fit_transform(class_docs).toarray().astype(np.float64)
    vocab = vec.get_feature_names_out().tolist()

    row_sums = raw_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    tf = raw_counts / row_sums

    avg_class_size = float(np.mean(class_sizes))
    df_counts = np.sum(raw_counts > 0, axis=0).astype(np.float64)
    idf = np.log(avg_class_size / (df_counts + 1.0))

    ctfidf_matrix = tf * idf
    return ctfidf_matrix, vocab, unique_labels


def get_top_terms(
    ctfidf_matrix: np.ndarray,
    vocab: list[str],
    unique_labels: list[int],
    top_n: int = 20,
) -> pd.DataFrame:
    """Get top c-TF-IDF terms per cluster with discriminativeness scores."""
    eps = 1e-10
    n_clusters = ctfidf_matrix.shape[0]
    rows = []

    for i, c in enumerate(unique_labels):
        other_mask = np.ones(n_clusters, dtype=bool)
        other_mask[i] = False
        max_other = ctfidf_matrix[other_mask].max(axis=0)

        disc = ctfidf_matrix[i] / (max_other + eps)

        term_data = []
        for v in range(len(vocab)):
            if ctfidf_matrix[i, v] > 0:
                term_data.append((vocab[v], float(ctfidf_matrix[i, v]), float(disc[v])))
        term_data.sort(key=lambda x: x[1], reverse=True)

        for rank, (term, score, d) in enumerate(term_data[:top_n]):
            rows.append({
                "cluster": c,
                "rank": rank + 1,
                "term": term,
                "ctfidf_score": score,
                "discriminativeness": d,
                "n_gram": len(term.split()),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exemplar sampling
# ---------------------------------------------------------------------------

def sample_exemplars(
    X_embed: np.ndarray,
    consensus_8d: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    articles: list[str],
    cluster_id: int,
    n_core: int = 3,
    n_boundary: int = 1,
    n_diverse: int = 1,
) -> list[dict]:
    """Sample representative exemplars from a cluster."""
    mask = labels == cluster_id
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    X_cluster = X_embed[indices]
    centroid = X_cluster.mean(axis=0, keepdims=True)
    sims = cosine_similarity(X_cluster, centroid).flatten()
    dists = 1.0 - sims

    total = n_core + n_boundary + n_diverse
    if len(indices) <= total:
        order = np.argsort(dists)
        return [{"text": texts[indices[i]][:500], "article": articles[indices[i]][:80],
                 "cohort": "core" if r < n_core else "boundary",
                 "sim_to_centroid": float(sims[i]),
                 "original_index": int(indices[i]), "cluster": cluster_id}
                for r, i in enumerate(order)]

    order = np.argsort(dists)
    selected: set[int] = set()
    exemplars = []

    for idx in order[:n_core]:
        idx = int(idx)
        selected.add(idx)
        exemplars.append({"text": texts[indices[idx]][:500], "article": articles[indices[idx]][:80],
                          "cohort": "core", "sim_to_centroid": float(sims[idx]),
                          "original_index": int(indices[idx]), "cluster": cluster_id})

    for idx in reversed(order):
        if len([e for e in exemplars if e["cohort"] == "boundary"]) >= n_boundary:
            break
        idx = int(idx)
        if idx in selected:
            continue
        selected.add(idx)
        exemplars.append({"text": texts[indices[idx]][:500], "article": articles[indices[idx]][:80],
                          "cohort": "boundary", "sim_to_centroid": float(sims[idx]),
                          "original_index": int(indices[idx]), "cluster": cluster_id})

    remaining = [i for i in range(len(indices)) if i not in selected]
    if remaining and n_diverse > 0:
        current = remaining[0]
        selected.add(current)
        exemplars.append({"text": texts[indices[current]][:500], "article": articles[indices[current]][:80],
                          "cohort": "diverse", "sim_to_centroid": float(sims[current]),
                          "original_index": int(indices[current]), "cluster": cluster_id})

    return exemplars


# ---------------------------------------------------------------------------
# Within vs across cluster similarity
# ---------------------------------------------------------------------------

def compute_within_across_similarity(
    X_embed: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 50,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Compute within-cluster and across-cluster cosine similarity."""
    rng = np.random.RandomState(rng_seed)
    unique = sorted(np.unique(labels))

    sampled = {}
    for c in unique:
        indices = np.where(labels == c)[0]
        if len(indices) <= sample_size:
            sampled[c] = indices
        else:
            sampled[c] = rng.choice(indices, sample_size, replace=False)

    rows = []
    for ci in unique:
        Xi = X_embed[sampled[ci]]

        if len(Xi) >= 2:
            sim_within = cosine_similarity(Xi)
            triu = sim_within[np.triu_indices(len(Xi), k=1)]
            within_mean = float(np.mean(triu))
        else:
            within_mean = 1.0

        across_means = []
        for cj in unique:
            if cj == ci:
                continue
            Xj = X_embed[sampled[cj]]
            sim_across = cosine_similarity(Xi, Xj)
            across_means.append(float(np.mean(sim_across)))

        across_mean = float(np.mean(across_means))

        rows.append({
            "cluster": ci,
            "n_points": len(sampled[ci]),
            "within_similarity": within_mean,
            "across_similarity": across_mean,
            "separation_ratio": within_mean / (across_mean + 1e-10),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

def build_cluster_prompt(cluster_id: int, n_chunks: int,
                         top_terms: pd.DataFrame,
                         exemplars: pd.DataFrame) -> str:
    terms_block = "\n".join([
        f"  - {r['term']} (score={r['ctfidf_score']:.4f}, "
        f"disc={r['discriminativeness']:.2f}, {r['n_gram']}g)"
        for _, r in top_terms.iterrows()
    ])

    exemplar_block = ""
    for _, ex in exemplars.iterrows():
        exemplar_block += (
            f"\n--- {ex['cohort']} exemplar (sim={ex['sim_to_centroid']:.3f}) "
            f"from: {str(ex['article'])[:80]} ---\n"
            f"{str(ex['text'])[:400]}\n"
        )

    return f"""You are analyzing cluster {cluster_id} from a topic model of 1,742 public discourse text chunks about AI and art (2013-2025).
This cluster contains {n_chunks} text chunks from news articles, podcasts, panel discussions, legal filings, and academic papers.

Statistically distinctive terms (c-TF-IDF, 1-4 grams):
{terms_block}

Representative documents:
{exemplar_block}

The existing macro-themes in this dataset are:
1. Institutions & Markets — industry economics, platform dynamics, market impact
2. Governance & Rights — copyright, regulation, legal frameworks, transparency
3. Technical Genealogy — AI models, training data, technical capabilities and processes
4. Practice & Pedagogy — creative process, education, workflow integration, craft
5. Philosophy of Creativity — authorship, originality, human vs machine aesthetics

If none of these macro-themes fit well, you may suggest "Other: [brief description]" and explain why.

Respond with ONLY a JSON object (no markdown, no explanation):
{{"label": "concise topic label (3-8 words)", "description": "1-2 sentence description of what this cluster is about", "macro_theme": "one of the 5 macro-themes above, or Other: [description]", "confidence": 0.0-1.0}}"""


def call_anthropic(prompt: str, model: str) -> dict | None:
    import anthropic
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"      {model} error: {e}")
    return None


def call_openai(prompt: str, model: str) -> dict | None:
    from openai import OpenAI
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        if text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
    except Exception as e:
        print(f"      {model} error: {e}")
    return None


def call_model(prompt: str, model_key: str) -> dict | None:
    spec = MODELS[model_key]
    if spec["provider"] == "anthropic":
        return call_anthropic(prompt, spec["model"])
    else:
        return call_openai(prompt, spec["model"])


def compute_label_similarity(label_a: str, label_b: str) -> float:
    words_a = set(label_a.lower().split())
    words_b = set(label_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def majority_macro_theme(themes: list[str]) -> tuple[str, float]:
    valid = [t for t in themes if t and t != ""]
    if not valid:
        return "", 0.0
    counts = Counter(valid)
    best, best_count = counts.most_common(1)[0]
    return best, best_count / len(valid)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("CLUSTER LABELING — Config C (nn=45, md=0.03, nc=8, k=20)")
    print(f"  Models: {', '.join(MODELS.keys())}")
    print(f"  Macro-themes: {len(MACRO_THEMES)}")
    print("=" * 70)

    # Load data
    X_pub = np.load(ROOT / "figures" / "prefix_comparison" / "prefix_embeddings_public.npy")
    consensus_8d = np.load(CONFIG_DIR / "consensus_coords.npy")
    labels_pub = np.load(CONFIG_DIR / "labels_pub.npy").astype(int)
    df_pub = pd.read_csv(DATA_DIR / "public_discourse_clean_chunks.csv")

    texts = df_pub["chunk_text"].tolist()
    articles = df_pub["article_name"].tolist()

    print(f"  Chunks: {len(texts)}, Clusters: {len(np.unique(labels_pub))}")

    # Cluster sizes
    sizes = Counter(int(l) for l in labels_pub)
    print(f"  Cluster sizes: min={min(sizes.values())}, max={max(sizes.values())}, "
          f"avg={np.mean(list(sizes.values())):.1f}")

    # ===== STEP 1: c-TF-IDF =====
    print(f"\n{'='*70}")
    print("STEP 1: c-TF-IDF (1-4 grams)")
    print(f"{'='*70}")

    ctfidf_matrix, vocab, unique_labels = build_ctfidf_matrix(texts, labels_pub)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Matrix shape: {ctfidf_matrix.shape}")

    df_terms = get_top_terms(ctfidf_matrix, vocab, unique_labels, top_n=20)
    df_terms.to_csv(OUTPUT_DIR / "ctfidf_top_terms.csv", index=False)
    print(f"  Saved ctfidf_top_terms.csv ({len(df_terms)} rows)")

    # Show top 5 discriminative terms per cluster
    for c in unique_labels:
        c_terms = df_terms[df_terms["cluster"] == c]
        disc_terms = c_terms[c_terms["discriminativeness"] > 1.5].head(5)
        if disc_terms.empty:
            disc_terms = c_terms.head(5)
        terms_str = ", ".join(disc_terms["term"].tolist())
        print(f"    Cluster {c:2d} ({sizes[c]:3d} chunks): {terms_str}")

    # ===== STEP 2: Exemplar Sampling =====
    print(f"\n{'='*70}")
    print("STEP 2: Exemplar Sampling (core + boundary + diverse)")
    print(f"{'='*70}")

    all_exemplars = []
    for c in unique_labels:
        exs = sample_exemplars(X_pub, consensus_8d, labels_pub, texts, articles, c)
        all_exemplars.extend(exs)

    df_exemplars = pd.DataFrame(all_exemplars)
    df_exemplars.to_csv(OUTPUT_DIR / "representative_exemplars.csv", index=False)
    print(f"  Saved representative_exemplars.csv ({len(df_exemplars)} rows)")

    # ===== STEP 3: Within vs Across Similarity =====
    print(f"\n{'='*70}")
    print("STEP 3: Within vs Across Cluster Similarity")
    print(f"{'='*70}")

    df_sim = compute_within_across_similarity(X_pub, labels_pub)
    df_sim.to_csv(OUTPUT_DIR / "within_vs_across_similarity.csv", index=False)

    weak_clusters = df_sim[df_sim["separation_ratio"] < 1.1]
    print(f"  Mean within-cluster sim: {df_sim['within_similarity'].mean():.4f}")
    print(f"  Mean across-cluster sim: {df_sim['across_similarity'].mean():.4f}")
    print(f"  Mean separation ratio: {df_sim['separation_ratio'].mean():.2f}")
    if len(weak_clusters) > 0:
        print(f"  WARNING: {len(weak_clusters)} clusters with separation ratio < 1.1:")
        for _, r in weak_clusters.iterrows():
            print(f"    Cluster {int(r['cluster'])}: ratio={r['separation_ratio']:.2f}")

    # ===== STEP 4: 4-Model LLM Labeling =====
    print(f"\n{'='*70}")
    print("STEP 4: 4-Model LLM Labeling")
    print(f"{'='*70}")

    results = []
    prompts_log = {}

    for c in unique_labels:
        n_chunks = sizes[c]
        c_terms = df_terms[df_terms["cluster"] == c].head(15)
        c_exemplars = df_exemplars[df_exemplars["cluster"] == c]

        prompt = build_cluster_prompt(c, n_chunks, c_terms, c_exemplars)
        prompts_log[str(c)] = prompt
        print(f"\nCluster {c:2d} ({n_chunks:3d} chunks):")

        row: dict = {"cluster": c, "n_chunks": n_chunks}

        disc_terms = c_terms[c_terms["discriminativeness"] > 1.5].head(5)
        if disc_terms.empty:
            disc_terms = c_terms.head(5)
        row["top_terms"] = ", ".join(disc_terms["term"].tolist())

        labels_dict: dict[str, str] = {}
        macros: dict[str, str] = {}
        confs: dict[str, float] = {}

        for model_key in MODELS:
            result = call_model(prompt, model_key)
            if result:
                label = result.get("label", "ERROR")
                macro = result.get("macro_theme", "")
                conf = float(result.get("confidence", 0.0))
                desc = result.get("description", "")
            else:
                label, macro, conf, desc = "API_ERROR", "", 0.0, ""

            labels_dict[model_key] = label
            macros[model_key] = macro
            confs[model_key] = conf

            row[f"{model_key}_label"] = label
            row[f"{model_key}_macro"] = macro
            row[f"{model_key}_conf"] = conf
            row[f"{model_key}_desc"] = desc

            status = "✓" if label != "API_ERROR" else "✗"
            print(f"    {status} {model_key:15s}: {label[:60]} [{macro}] (conf={conf})")

        # Agreement metrics
        valid_labels = [l for l in labels_dict.values() if l != "API_ERROR"]
        valid_macros = [m for m in macros.values() if m and m != ""]
        valid_confs = [co for co in confs.values() if co > 0]

        pair_sims = []
        model_keys = list(MODELS.keys())
        for i in range(len(model_keys)):
            for j in range(i + 1, len(model_keys)):
                la = labels_dict[model_keys[i]]
                lb = labels_dict[model_keys[j]]
                if la != "API_ERROR" and lb != "API_ERROR":
                    pair_sims.append(compute_label_similarity(la, lb))
        mean_label_sim = float(np.mean(pair_sims)) if pair_sims else 0.0

        consensus_macro, macro_agreement = majority_macro_theme(valid_macros)
        row["consensus_macro_theme"] = consensus_macro
        row["macro_agreement"] = macro_agreement

        mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
        row["mean_label_similarity"] = mean_label_sim
        row["mean_model_confidence"] = mean_conf
        row["n_models_succeeded"] = len(valid_labels)
        row["combined_confidence"] = mean_conf * (0.5 + 0.5 * mean_label_sim) * (0.5 + 0.5 * macro_agreement)

        print(f"    → Consensus macro: {consensus_macro} ({macro_agreement:.0%} agree), "
              f"label_sim={mean_label_sim:.2f}, combined_conf={row['combined_confidence']:.2f}")

        results.append(row)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "quad_llm_labels.csv", index=False)
    print(f"\nSaved quad_llm_labels.csv")

    # Save prompts for reproducibility
    with open(OUTPUT_DIR / "llm_prompts.json", "w") as f:
        json.dump(prompts_log, f, indent=2)

    # ===== MACRO-THEME FIT ANALYSIS =====
    print(f"\n{'='*70}")
    print("MACRO-THEME FIT ANALYSIS")
    print(f"{'='*70}")

    all_macros_seen = set()
    for mk in MODELS:
        all_macros_seen.update(df_results[f"{mk}_macro"].unique())
    all_macros_seen.update(df_results["consensus_macro_theme"].unique())

    macro_fit = []
    for mt in MACRO_THEMES + sorted(all_macros_seen - set(MACRO_THEMES)):
        counts = {}
        for mk in MODELS:
            counts[mk] = int((df_results[f"{mk}_macro"] == mt).sum())
        consensus_count = int((df_results["consensus_macro_theme"] == mt).sum())
        row_mt = {"macro_theme": mt, "consensus_count": consensus_count}
        row_mt.update(counts)
        macro_fit.append(row_mt)
        print(f"  {mt:35s}: consensus={consensus_count}, "
              + ", ".join(f"{mk}={counts[mk]}" for mk in MODELS))

    df_macro = pd.DataFrame(macro_fit)
    df_macro.to_csv(OUTPUT_DIR / "macro_theme_fit.csv", index=False)

    overall_macro_agree = df_results["macro_agreement"].mean()
    print(f"\n  Overall macro-theme agreement: {overall_macro_agree:.1%}")

    # Check if any models suggested "Other" themes
    other_themes = []
    for mk in MODELS:
        others = df_results[df_results[f"{mk}_macro"].str.startswith("Other", na=False)]
        for _, r in others.iterrows():
            other_themes.append({"cluster": r["cluster"], "model": mk,
                                 "suggested_macro": r[f"{mk}_macro"]})
    if other_themes:
        print(f"\n  {len(other_themes)} 'Other' macro-theme suggestions:")
        for ot in other_themes:
            print(f"    Cluster {ot['cluster']} ({ot['model']}): {ot['suggested_macro']}")

    # ===== HUMAN REVIEW FILE =====
    print(f"\n{'='*70}")
    print("GENERATING HUMAN REVIEW FILE")
    print(f"{'='*70}")

    review_cols = ["cluster", "n_chunks", "top_terms"]
    for mk in MODELS:
        review_cols.extend([f"{mk}_label", f"{mk}_macro", f"{mk}_conf"])
    review_cols.extend([
        "consensus_macro_theme", "macro_agreement",
        "mean_label_similarity", "combined_confidence", "n_models_succeeded",
    ])

    df_review = df_results[review_cols].copy()
    df_review["human_approved_label"] = ""
    df_review["human_approved_macro_theme"] = ""
    df_review["human_notes"] = ""
    df_review.to_csv(OUTPUT_DIR / "clusters_for_human_review.csv", index=False)
    print(f"Saved clusters_for_human_review.csv")

    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for mk in MODELS:
        n_ok = (df_results[f"{mk}_label"] != "API_ERROR").sum()
        print(f"  {mk:15s}: {n_ok}/{K} succeeded")

    print(f"\n  Mean label similarity (pairwise): {df_results['mean_label_similarity'].mean():.2f}")
    print(f"  Mean model confidence: {df_results['mean_model_confidence'].mean():.2f}")
    print(f"  Mean combined confidence: {df_results['combined_confidence'].mean():.2f}")
    print(f"  Macro-theme agreement: {overall_macro_agree:.1%}")

    high = (df_results["combined_confidence"] >= 0.5).sum()
    med = ((df_results["combined_confidence"] >= 0.3) & (df_results["combined_confidence"] < 0.5)).sum()
    low = (df_results["combined_confidence"] < 0.3).sum()
    print(f"  High confidence (>=0.5): {high}")
    print(f"  Medium confidence (0.3-0.5): {med}")
    print(f"  Low confidence (<0.3): {low}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
