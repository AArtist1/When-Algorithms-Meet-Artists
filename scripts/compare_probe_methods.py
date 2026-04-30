"""Compare keyword-based and embedding-based public probe extraction methods.

Generates comparison tables, visualizations, and an HTML report showing
how the two extraction methods agree or differ. This script runs after
both extract_public_probes.py and extract_public_probes_embedding.py
have completed.

Output files:
    figures/final_pipeline/probe_method_comparison.csv
    figures/final_pipeline/probe_method_comparison.html
    figures/final_pipeline/probe_method_comparison.png

Side effects:
    Writes files to disk. Prints comparison stats.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures" / "final_pipeline"

KEYWORD_CSV = DATA_DIR / "public_probes_keyword.csv"
EMBEDDING_CSV = DATA_DIR / "public_probes_embedding.csv"
THEMES = ["compensation", "ownership", "threat", "transparency", "utility"]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def get_theme_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count probes per theme."""
    return {t: int((df["theme"] == t).sum()) for t in THEMES}


def get_text_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, float]:
    """Compute text overlap between two probe sets, overall and per theme."""
    col_a = "probe_text" if "probe_text" in df_a.columns else "text"
    col_b = "probe_text" if "probe_text" in df_b.columns else "text"

    texts_a = set(df_a[col_a].str.strip().str.lower())
    texts_b = set(df_b[col_b].str.strip().str.lower())
    overlap = texts_a & texts_b

    result = {
        "overall_overlap": len(overlap),
        "pct_of_keyword": len(overlap) / len(texts_a) * 100 if texts_a else 0,
        "pct_of_embedding": len(overlap) / len(texts_b) * 100 if texts_b else 0,
    }

    for theme in THEMES:
        ta = set(df_a[df_a["theme"] == theme][col_a].str.strip().str.lower())
        tb = set(df_b[df_b["theme"] == theme][col_b].str.strip().str.lower())
        theme_overlap = ta & tb
        result[f"{theme}_overlap"] = len(theme_overlap)
        result[f"{theme}_pct_kw"] = len(theme_overlap) / len(ta) * 100 if ta else 0

    return result


def get_article_coverage(df: pd.DataFrame) -> int:
    """Count unique articles represented in probe set."""
    for col in ["article_name", "unit_id"]:
        if col in df.columns:
            return int(df[col].nunique())
    return 0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_comparison_figure(
    kw_counts: dict[str, int],
    emb_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Generate side-by-side bar chart comparing both methods.

    Side effects: Writes PNG file.
    """
    _fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(THEMES))
    width = 0.35

    kw_vals = [kw_counts[t] for t in THEMES]
    emb_vals = [emb_counts[t] for t in THEMES]

    bars1 = ax.bar(x - width / 2, kw_vals, width, label="Keyword", color="#e17055", alpha=0.85)
    bars2 = ax.bar(x + width / 2, emb_vals, width, label="Embedding", color="#0984e3", alpha=0.85)

    ax.set_xlabel("Theme", fontsize=11)
    ax.set_ylabel("Probe Count", fontsize=11)
    ax.set_title("Probe Extraction: Keyword vs Embedding Method", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(THEMES, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.bar_label(bars1, padding=2, fontsize=9)
    ax.bar_label(bars2, padding=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def generate_html_report(
    kw_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    kw_counts: dict[str, int],
    emb_counts: dict[str, int],
    overlap: dict[str, float],
    output_path: Path,
) -> None:
    """Generate HTML comparison report.

    Side effects: Writes HTML file.
    """
    theme_rows = ""
    for t in THEMES:
        kc = kw_counts[t]
        ec = emb_counts[t]
        ov = overlap.get(f"{t}_overlap", 0)
        pct = overlap.get(f"{t}_pct_kw", 0)
        theme_rows += f"""
        <tr>
            <td>{t}</td>
            <td>{kc}</td>
            <td>{ec}</td>
            <td>{ov}</td>
            <td>{pct:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Probe Method Comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f8f9fa; color: #1a1a2e; padding: 2rem; max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
  h2 {{ font-size: 1.2rem; margin: 1.5rem 0 0.8rem; border-bottom: 2px solid #dfe6e9; padding-bottom: 0.3rem; }}
  .subtitle {{ color: #636e72; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; }}
  th {{ background: #2d3436; color: white; font-size: 0.85rem; }}
  td {{ border-bottom: 1px solid #f1f2f6; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{ background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .card .big {{ font-size: 2rem; font-weight: 700; }}
  .card .label {{ font-size: 0.8rem; color: #636e72; }}
  .kw {{ color: #e17055; }}
  .emb {{ color: #0984e3; }}
  .note {{ background: #dfe6e9; border-radius: 6px; padding: 1rem; margin: 1rem 0; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Probe Method Comparison: Keyword vs Embedding</h1>
<p class="subtitle">Both methods run on the clean 1,736-chunk corpus with prefix embeddings</p>

<div class="cards">
  <div class="card"><div class="big kw">{len(kw_df)}</div><div class="label">Keyword Probes</div></div>
  <div class="card"><div class="big emb">{len(emb_df)}</div><div class="label">Embedding Probes</div></div>
  <div class="card"><div class="big">{overlap['overall_overlap']}</div><div class="label">Overlapping Texts</div></div>
</div>

<h2>Per-Theme Comparison</h2>
<table>
  <thead><tr><th>Theme</th><th>Keyword</th><th>Embedding</th><th>Overlap</th><th>Overlap % (of KW)</th></tr></thead>
  <tbody>{theme_rows}
  <tr style="font-weight:600; background:#f8f9fa">
    <td>TOTAL</td><td>{len(kw_df)}</td><td>{len(emb_df)}</td>
    <td>{overlap['overall_overlap']}</td><td>{overlap['pct_of_keyword']:.1f}%</td>
  </tr></tbody>
</table>

<div class="note">
  <strong>Interpretation:</strong> Low overlap between methods is expected and healthy. Keyword matching
  finds sentences with explicit theme vocabulary. Embedding retrieval finds sentences that are
  semantically similar to theme centroids even without keyword matches. The two methods complement
  each other. High overlap on a theme (>30%) suggests the theme has strong lexical markers.
  Low overlap (<10%) suggests the theme is expressed through indirect language that keywords miss.
</div>

<h2>Article Coverage</h2>
<table>
  <thead><tr><th>Method</th><th>Unique Articles</th><th>% of Corpus (125)</th></tr></thead>
  <tbody>
    <tr><td>Keyword</td><td>{get_article_coverage(kw_df)}</td><td>{get_article_coverage(kw_df)/125*100:.1f}%</td></tr>
    <tr><td>Embedding</td><td>{get_article_coverage(emb_df)}</td><td>{get_article_coverage(emb_df)/125*100:.1f}%</td></tr>
  </tbody>
</table>

<img src="probe_method_comparison.png" style="width:100%; max-width:700px; margin:1rem 0" alt="Comparison chart">

<footer style="margin-top:2rem; font-size:0.8rem; color:#b2bec3; border-top:1px solid #dfe6e9; padding-top:0.5rem">
  Generated April 2026 | When Algorithms Meet Artists
</footer>
</body>
</html>"""

    output_path.write_text(html)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Compare keyword and embedding probe extraction outputs."""
    print("=" * 60)
    print("PROBE METHOD COMPARISON")
    print("=" * 60)

    if not KEYWORD_CSV.exists():
        print(f"ERROR: {KEYWORD_CSV} not found. Run extract_public_probes.py first.")
        return
    if not EMBEDDING_CSV.exists():
        print(f"ERROR: {EMBEDDING_CSV} not found. Run extract_public_probes_embedding.py first.")
        return

    kw_df = pd.read_csv(KEYWORD_CSV)
    emb_df = pd.read_csv(EMBEDDING_CSV)
    print(f"\nKeyword probes: {len(kw_df)}")
    print(f"Embedding probes: {len(emb_df)}")

    kw_counts = get_theme_counts(kw_df)
    emb_counts = get_theme_counts(emb_df)
    overlap = get_text_overlap(kw_df, emb_df)

    print(f"\nOverlapping texts: {overlap['overall_overlap']}")
    print(f"  As % of keyword set: {overlap['pct_of_keyword']:.1f}%")
    print(f"  As % of embedding set: {overlap['pct_of_embedding']:.1f}%")

    print(f"\n{'Theme':<18} {'Keyword':<10} {'Embedding':<10} {'Overlap':<10}")
    print("-" * 48)
    for t in THEMES:
        print(f"{t:<18} {kw_counts[t]:<10} {emb_counts[t]:<10} {overlap.get(f'{t}_overlap', 0):<10}")

    # Save comparison CSV
    rows = []
    for t in THEMES:
        rows.append({
            "theme": t,
            "keyword_count": kw_counts[t],
            "embedding_count": emb_counts[t],
            "overlap_count": overlap.get(f"{t}_overlap", 0),
            "overlap_pct_kw": overlap.get(f"{t}_pct_kw", 0),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "probe_method_comparison.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'probe_method_comparison.csv'}")

    # Generate visuals
    generate_comparison_figure(kw_counts, emb_counts, OUTPUT_DIR / "probe_method_comparison.png")
    generate_html_report(kw_df, emb_df, kw_counts, emb_counts, overlap,
                         OUTPUT_DIR / "probe_method_comparison.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
