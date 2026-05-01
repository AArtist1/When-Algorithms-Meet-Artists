"""Validate public probe theme assignments using dual-LLM classification.

Sends each public probe to both gpt-5.4-mini (OpenAI) and Claude Sonnet 4.6
(Anthropic, model id `claude-sonnet-4-6`) with an identical classification
prompt. Compares LLM-assigned themes to the extraction-assigned themes to
measure precision and inter-rater agreement.

Requirements:
    OPENAI_API_KEY environment variable
    ANTHROPIC_API_KEY environment variable
    pip install openai anthropic

Output files:
    figures/final_pipeline/probe_theme_validation.csv      (per-probe results)
    figures/final_pipeline/probe_theme_validation_summary.csv  (per-theme summary)
    figures/final_pipeline/probe_theme_validation_report.html  (visual report)

Side effects:
    Makes API calls to OpenAI and Anthropic. Writes files. Prints progress.
    Estimated cost: ~$0.50 total for 750 probes x 2 models.
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures" / "final_pipeline"

PROBE_CSV = DATA_DIR / "public_probes.csv"
VALID_THEMES = ["threat", "utility", "ownership", "transparency", "compensation"]

CLASSIFICATION_PROMPT = """Classify the following sentence into exactly one of these five themes based on its primary topic:

- threat: concerns about AI threatening artists, jobs, or creative practice
- utility: AI as a useful or positive tool for art and creativity
- ownership: who should own AI-generated artwork, intellectual property rights
- transparency: disclosure of training data, consent, how AI models use artist work
- compensation: payment, revenue sharing, financial models for artists whose work trains AI

If the sentence does not clearly fit any theme, respond with "none".

Sentence: "{text}"

Respond with ONLY the theme name (one word, lowercase). Do not explain."""

# Rate limiting
OPENAI_DELAY = 0.1   # seconds between calls
ANTHROPIC_DELAY = 0.15


# ---------------------------------------------------------------------------
# LLM Classification Functions
# ---------------------------------------------------------------------------

def classify_with_openai(
    texts: list[str],
    model: str = "gpt-5.4-mini",
    delay: float = OPENAI_DELAY,
) -> list[str]:
    """Classify texts using OpenAI API.

    Args:
        texts: List of sentence strings to classify.
        model: OpenAI model identifier.
        delay: Seconds between API calls.

    Returns:
        List of theme labels (one per text).

    Side effects:
        Makes API calls. Prints progress.
    """
    from openai import OpenAI
    client = OpenAI()
    results = []

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  OpenAI: {i}/{len(texts)}...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(text=text)}],
                max_completion_tokens=256,
            )
            label = response.choices[0].message.content.strip().lower()
            label = label.replace('"', '').replace("'", "").strip()
            if label not in VALID_THEMES:
                label = "none"
            results.append(label)
        except Exception as e:
            print(f"  OpenAI error on item {i}: {e}")
            results.append("error")
        time.sleep(delay)

    return results


def classify_with_anthropic(
    texts: list[str],
    model: str = "claude-sonnet-4-6",
    delay: float = ANTHROPIC_DELAY,
) -> list[str]:
    """Classify texts using Anthropic API.

    Args:
        texts: List of sentence strings to classify.
        model: Anthropic model identifier.
        delay: Seconds between API calls.

    Returns:
        List of theme labels (one per text).

    Side effects:
        Makes API calls. Prints progress.
    """
    from anthropic import Anthropic
    client = Anthropic()
    results = []

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  Anthropic: {i}/{len(texts)}...")
        try:
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(text=text)}],
            )
            label = response.content[0].text.strip().lower()
            label = label.replace('"', '').replace("'", "").strip()
            if label not in VALID_THEMES:
                label = "none"
            results.append(label)
        except Exception as e:
            print(f"  Anthropic error on item {i}: {e}")
            results.append("error")
        time.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# Agreement Metrics
# ---------------------------------------------------------------------------

def get_accuracy(assigned: list[str], classified: list[str]) -> float:
    """Compute simple accuracy between two label lists."""
    matches = sum(1 for a, c in zip(assigned, classified) if a == c)
    return matches / len(assigned) if assigned else 0.0


def get_cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa between two raters.

    Side effects: None.
    """
    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Confusion matrix
    matrix = {}
    for cat in categories:
        matrix[cat] = {c: 0 for c in categories}
    for a, b in zip(labels_a, labels_b):
        matrix[a][b] += 1

    # Observed agreement
    p_o = sum(matrix[c][c] for c in categories) / n

    # Expected agreement
    p_e = 0.0
    for c in categories:
        row_sum = sum(matrix[c].values()) / n
        col_sum = sum(matrix[r][c] for r in categories) / n
        p_e += row_sum * col_sum

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def get_per_theme_metrics(
    df: pd.DataFrame,
    assigned_col: str,
    classified_col: str,
) -> pd.DataFrame:
    """Compute per-theme precision and recall.

    Side effects: None.
    """
    rows = []
    for theme in VALID_THEMES:
        assigned_mask = df[assigned_col] == theme
        classified_mask = df[classified_col] == theme
        tp = (assigned_mask & classified_mask).sum()
        fp = (~assigned_mask & classified_mask).sum()
        fn = (assigned_mask & ~classified_mask).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "theme": theme,
            "n_assigned": int(assigned_mask.sum()),
            "n_classified": int(classified_mask.sum()),
            "tp": int(tp),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_validation_report(
    df: pd.DataFrame,
    openai_metrics: pd.DataFrame,
    anthropic_metrics: pd.DataFrame,
    overall: dict,
    output_path: Path,
) -> None:
    """Generate HTML validation report.

    Side effects: Writes HTML file.
    """
    def metrics_table(metrics_df: pd.DataFrame, model_name: str) -> str:
        rows = ""
        for _, r in metrics_df.iterrows():
            rows += f"<tr><td>{r['theme']}</td><td>{r['n_assigned']}</td><td>{r['n_classified']}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td></tr>\n"
        return f"""
        <h3>{model_name}</h3>
        <table><thead><tr><th>Theme</th><th>Assigned</th><th>Classified</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Probe Theme Validation</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.6rem; }} h2 {{ margin-top: 2rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
h3 {{ margin-top: 1.5rem; color: #2d3436; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #2d3436; color: white; }}
.card {{ background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }}
.big {{ font-size: 1.8rem; font-weight: 700; text-align: center; }}
.label {{ font-size: 0.8rem; color: #666; text-align: center; }}
.good {{ color: #00b894; }} .ok {{ color: #fdcb6e; }} .bad {{ color: #e17055; }}
</style></head><body>
<h1>Public Probe Theme Validation: Dual-LLM Classification</h1>
<p>Each of {overall['n_probes']} probes classified by both gpt-5.4-mini and Claude Sonnet 4.6</p>

<div class="cards">
<div class="card"><div class="big {'good' if overall['openai_accuracy'] > 0.8 else 'ok'}">{overall['openai_accuracy']:.1%}</div><div class="label">GPT Accuracy</div></div>
<div class="card"><div class="big {'good' if overall['anthropic_accuracy'] > 0.8 else 'ok'}">{overall['anthropic_accuracy']:.1%}</div><div class="label">Claude Accuracy</div></div>
<div class="card"><div class="big {'good' if overall['consensus_accuracy'] > 0.8 else 'ok'}">{overall['consensus_accuracy']:.1%}</div><div class="label">Consensus Accuracy</div></div>
<div class="card"><div class="big">{overall['inter_llm_kappa']:.3f}</div><div class="label">Inter-LLM Kappa</div></div>
</div>

<h2>Overall Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total probes</td><td>{overall['n_probes']}</td></tr>
<tr><td>OpenAI accuracy vs extraction</td><td>{overall['openai_accuracy']:.3f}</td></tr>
<tr><td>Anthropic accuracy vs extraction</td><td>{overall['anthropic_accuracy']:.3f}</td></tr>
<tr><td>Inter-LLM agreement (kappa)</td><td>{overall['inter_llm_kappa']:.3f}</td></tr>
<tr><td>Consensus accuracy (both agree with extraction)</td><td>{overall['consensus_accuracy']:.3f}</td></tr>
<tr><td>Both LLMs agree with each other</td><td>{overall['inter_llm_agreement']:.3f}</td></tr>
</table>

<h2>Per-Theme Results</h2>
{metrics_table(openai_metrics, "gpt-5.4-mini")}
{metrics_table(anthropic_metrics, "Claude Sonnet 4.6")}

<h2>Interpretation</h2>
<div class="card">
<p>Accuracy > 80% indicates that the extraction method assigns themes consistent with
independent LLM judgment. Inter-LLM kappa > 0.6 indicates substantial agreement
between the two independent classifiers. Together, these confirm that the probes
are genuinely about their assigned themes and not random keyword matches.</p>
</div>

</body></html>"""

    output_path.write_text(html)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run dual-LLM theme validation on public probes."""
    print("=" * 60)
    print("DUAL-LLM PROBE THEME VALIDATION")
    print("=" * 60)

    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Export it before running.")
        print("  export OPENAI_API_KEY='sk-...'")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Export it before running.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    # Load probes
    df = pd.read_csv(PROBE_CSV)
    text_col = "probe_text" if "probe_text" in df.columns else "text"
    texts = df[text_col].tolist()
    assigned_themes = df["theme"].tolist()
    print(f"\nLoaded {len(df)} probes")
    print(f"Theme distribution: {dict(df['theme'].value_counts())}")

    # Classify with both LLMs
    print("\nClassifying with OpenAI gpt-5.4-mini...")
    t0 = time.time()
    openai_labels = classify_with_openai(texts)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\nClassifying with Anthropic Claude Sonnet 4.6...")
    t0 = time.time()
    anthropic_labels = classify_with_anthropic(texts)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Build results DataFrame
    df["openai_theme"] = openai_labels
    df["anthropic_theme"] = anthropic_labels
    df["openai_match"] = [a == o for a, o in zip(assigned_themes, openai_labels)]
    df["anthropic_match"] = [a == c for a, c in zip(assigned_themes, anthropic_labels)]
    df["llms_agree"] = [o == c for o, c in zip(openai_labels, anthropic_labels)]
    df["consensus_match"] = [a == o == c for a, o, c in zip(assigned_themes, openai_labels, anthropic_labels)]

    # Compute metrics
    openai_acc = get_accuracy(assigned_themes, openai_labels)
    anthropic_acc = get_accuracy(assigned_themes, anthropic_labels)
    inter_kappa = get_cohens_kappa(openai_labels, anthropic_labels)
    inter_agreement = sum(df["llms_agree"]) / len(df)
    consensus_acc = sum(df["consensus_match"]) / len(df)

    openai_metrics = get_per_theme_metrics(df, "theme", "openai_theme")
    anthropic_metrics = get_per_theme_metrics(df, "theme", "anthropic_theme")

    overall = {
        "n_probes": len(df),
        "openai_accuracy": openai_acc,
        "anthropic_accuracy": anthropic_acc,
        "inter_llm_kappa": inter_kappa,
        "inter_llm_agreement": inter_agreement,
        "consensus_accuracy": consensus_acc,
    }

    # Print results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  OpenAI accuracy: {openai_acc:.3f}")
    print(f"  Anthropic accuracy: {anthropic_acc:.3f}")
    print(f"  Inter-LLM kappa: {inter_kappa:.3f}")
    print(f"  Inter-LLM agreement: {inter_agreement:.3f}")
    print(f"  Consensus accuracy: {consensus_acc:.3f}")

    print(f"\n  Per-theme (OpenAI):")
    for _, r in openai_metrics.iterrows():
        print(f"    {r['theme']:<16} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")

    print(f"\n  Per-theme (Anthropic):")
    for _, r in anthropic_metrics.iterrows():
        print(f"    {r['theme']:<16} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "probe_theme_validation.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'probe_theme_validation.csv'}")

    summary = pd.concat([
        openai_metrics.assign(model="gpt-5.4-mini"),
        anthropic_metrics.assign(model="claude-sonnet-4-6"),
    ])
    summary.to_csv(OUTPUT_DIR / "probe_theme_validation_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'probe_theme_validation_summary.csv'}")

    generate_validation_report(df, openai_metrics, anthropic_metrics, overall,
                               OUTPUT_DIR / "probe_theme_validation_report.html")

    print(f"\nDone. Overall consensus accuracy: {consensus_acc:.1%}")


if __name__ == "__main__":
    main()
