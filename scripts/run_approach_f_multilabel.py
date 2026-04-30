"""Approach F — multi-label LLM classification with revised prompts.

This script implements a redesigned LLM validation that addresses the
limitations of the single-label approaches in
`scripts/compare_llm_validation_approaches.py`. Key changes:

1. **Multi-label output** — the LLM lists ALL themes that apply to a
   sentence, not just one. Public-discourse sentences routinely entangle
   multiple themes (e.g., "companies should disclose data and pay artists"
   → transparency + compensation).

2. **Theme descriptions grounded in the Lovato et al. 2024 Likert
   templates** — the themes are written in the exact form artists were
   asked about in the survey, so the LLM classifies against the ground
   truth of the original survey instrument.

3. **Multi-theme examples drawn from actual language** — three example
   sentences illustrating how multi-theme framing appears in practice.

4. **6 artist-probe exemplars per theme** (30 total in-context examples)
   instead of 3 (the pilot used 3). Drawn from `artist_perspectives.csv`.

5. **No "none" option** — removed. Forces the LLM to commit to at least
   one theme. The theme list now always yields a valid classification.

6. **Primary metric: multi-label recall** — fraction of probes where the
   extraction's assigned theme appears in the LLM's predicted theme list.

7. **Two runs**: (a) balanced 50 public probes (10 per theme), and
   (b) 30 compensation-only public probes (to pin down the theme
   that failed most in the pilot).

Outputs:
    figures/final_pipeline/approach_f_multilabel_results.csv
    figures/final_pipeline/approach_f_per_probe_balanced.csv
    figures/final_pipeline/approach_f_per_probe_compensation.csv

Side effects:
    Reads public_probes.csv and artist_perspectives.csv.
    Loads OPENAI_API_KEY and ANTHROPIC_API_KEY from .env.
    Makes ~160 API calls (80 probes × 2 models).
    Estimated runtime: ~8-12 min.
    Estimated cost: ~$5-7 (Opus 4.6 dominates).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures" / "final_pipeline"
PROBE_CSV = DATA_DIR / "public_probes.csv"
ARTIST_CSV = DATA_DIR / "artist_perspectives.csv"

VALID_THEMES = ["threat", "utility", "ownership", "transparency", "compensation"]

# Run on ALL public probes (no sampling) for publication-grade N=150 per theme
N_FEWSHOT_PER_THEME = 6     # 30 exemplars total in prompt
SEED = 42

OPENAI_DELAY = 0.1
ANTHROPIC_DELAY = 0.15

OPENAI_MODEL = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def load_env_file() -> None:
    """Load API keys from .env files if not already set.

    Side effects: mutates os.environ.
    """
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"):
        return
    candidate_paths = [
        ROOT / ".env",
        ROOT / "When-Algorithms-Meet-Artists-EDIT" / ".env",
        Path.home() / ".env",
    ]
    for p in candidate_paths:
        if not p.exists():
            continue
        try:
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") and val:
                    os.environ.setdefault(key, val)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REVISED THEME DESCRIPTIONS
# Grounded in the exact Lovato et al. 2024 survey Likert templates.
# ---------------------------------------------------------------------------

THEME_DEFINITIONS_REVISED = """**threat**
The core question from the survey: "Are AI art models a threat to art workers?"
Includes sentences about: AI harming artists' livelihoods, job displacement, creative workers losing income to generative AI, artist anxiety or fear about AI, the survival of art as a profession, devaluation of human creative skill, and the existential/economic threat AI poses to working artists.

**utility**
The core question from the survey: "Are AI art models a positive development in the field of art?"
Includes sentences about: AI as a useful tool for artists, AI-assisted creativity, AI as a creative collaborator, AI democratizing art-making, AI expanding expressive possibilities, AI helping artists iterate faster, AI as a positive development for the art field, and any endorsement of AI's creative benefits.

**ownership**
The core question from the survey: "Who should own AI-generated artwork in an artist's style — the AI user, the original artist, or nobody?"
Includes sentences about: copyright of AI-generated artwork, intellectual property rights, legal authorship of AI outputs, style ownership, derivative-work rights, who holds rights to AI-generated images, infringement claims, and the legal/moral status of the output.

**transparency**
The core question from the survey: "Should AI model creators be required to disclose in detail what art and images they use to train their AI models?"
Includes sentences about: disclosure of training datasets, consent for data scraping, opt-in / opt-out mechanisms, data provenance, dataset cards, audit requirements, the right to know what training data was used, and obligations on AI developers to reveal sources.

**compensation**
The core question from the survey: "How, if at all, should artists be paid when their work is used to train AI art models?"
Includes sentences about: royalties, licensing fees, revenue sharing, payment to artists whose work trained AI models, monetary restitution, lawsuits seeking damages, profit-sharing schemes, donation/no-pay models, flat fees for training data, and the economic terms under which AI companies should compensate artists.
"""

MULTI_THEME_EXAMPLES = """Multi-theme example sentences (how to think about sentences that touch several themes):

- "AI companies should disclose training data and pay the artists whose work was used"
  → transparency, compensation

- "Artists losing income to generative AI deserve copyright protection over their style"
  → threat, ownership, compensation

- "AI is a useful tool for rapid ideation but it raises questions about who owns the output"
  → utility, ownership

Public discourse sentences often entangle multiple themes. Identify ALL that apply, not just the most prominent.
"""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_F_MULTILABEL = """You are classifying short sentences from public discourse about AI and art. Public-discourse sentences frequently entangle multiple themes simultaneously. Identify ALL themes that apply to each sentence, not just the most prominent one.

Theme definitions:

{definitions}

{multi_examples}

Artist-probe exemplars (each exemplar illustrates one theme):

{exemplars}

Now classify this sentence. List ALL themes that apply. Expect 1-3 themes for most sentences. You must return at least one theme.

Sentence: "{text}"

Respond with ONLY a comma-separated list of theme names (lowercase), no explanation. Example responses:
- "threat, compensation"
- "ownership"
- "transparency, ownership, compensation"
"""


# ---------------------------------------------------------------------------
# Few-shot exemplar construction
# ---------------------------------------------------------------------------

def build_exemplars(rng: random.Random, k: int = N_FEWSHOT_PER_THEME) -> str:
    """Pull k exemplars per theme from artist_perspectives.csv.

    Uses question_group as the theme label (survey ground truth).
    Returns a multi-line string grouped by theme.

    Side effects: reads artist_perspectives.csv.
    """
    ap = pd.read_csv(ARTIST_CSV)
    text_col = "perspective_text"
    theme_col = "question_group"
    blocks: list[str] = []
    for theme in VALID_THEMES:
        sub_df = ap[ap[theme_col] == theme]
        sub_texts = sub_df[text_col].dropna().drop_duplicates().tolist()
        rng.shuffle(sub_texts)
        chosen = sub_texts[:k]
        # If fewer unique texts than k (e.g. threat, utility, transparency
        # only have 3 unique Likert statements each), pad with duplicates
        # from the original pool so the prompt has exactly k exemplars.
        if len(chosen) < k:
            pool = sub_df[text_col].dropna().tolist()
            rng.shuffle(pool)
            chosen += pool[: k - len(chosen)]
        chosen = [c[:240] for c in chosen]
        ex_lines = "\n".join(f'  - "{c}"' for c in chosen)
        blocks.append(f"{theme}:\n{ex_lines}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_all_public_probes() -> pd.DataFrame:
    """Return the full public-probe set (all 750 probes, 150 per theme).

    Side effects: reads public_probes.csv.
    """
    df = pd.read_csv(PROBE_CSV)
    text_col = "text" if "text" in df.columns else "probe_text"
    df["__text"] = df[text_col]
    df = df[df["theme"].isin(VALID_THEMES)].copy().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def call_openai(client, prompt: str) -> str:
    """Single OpenAI chat completion. Side effects: API call."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=64,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def call_anthropic(client, prompt: str) -> str:
    """Single Anthropic message. Side effects: API call."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.content[0].text or "").strip()


def safe_call(fn, *args, **kwargs) -> tuple[str, str | None]:
    """Wrapper that catches exceptions."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return "", str(e)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_multilabel(raw: str) -> list[str]:
    """Parse a comma-separated list of themes into canonical labels.

    Ignores 'none' entirely (approach F forbids none, but be defensive).
    Returns lowercase unique theme names in first-seen order.
    """
    if not raw:
        return []
    s = raw.strip().lower()
    # Strip surrounding quotes or brackets
    s = s.strip('"').strip("'").strip("[").strip("]")
    # Take first line only (in case LLM adds explanation)
    s = s.split("\n")[0]
    parts = re.split(r"[,\|\n\t]+", s)
    out: list[str] = []
    for p in parts:
        p = p.strip().strip('"').strip("'").strip(".").strip()
        if p in VALID_THEMES and p not in out:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Run Approach F on a probe list
# ---------------------------------------------------------------------------

def run_approach_f(
    df: pd.DataFrame,
    prompt_template: str,
    label: str,
    *,
    openai_client,
    anthropic_client,
) -> pd.DataFrame:
    """Run approach F on every probe in df, storing results back into df.

    Returns the df augmented with F_openai_labels, F_anthropic_labels
    columns (each a JSON-encoded list) and F_openai_match, F_anthropic_match
    columns (booleans: did extraction theme appear in predicted list?).

    Side effects: API calls, prints progress.
    """
    openai_out: list[list[str]] = []
    anthropic_out: list[list[str]] = []

    texts = df["__text"].astype(str).tolist()

    for i, t in enumerate(texts):
        prompt = prompt_template.format(text=t)
        if i % 10 == 0:
            print(f"  [{label}] {i}/{len(texts)}")

        raw_o, err_o = safe_call(call_openai, openai_client, prompt)
        labels_o = parse_multilabel(raw_o) if not err_o else []
        openai_out.append(labels_o)
        time.sleep(OPENAI_DELAY)

        raw_a, err_a = safe_call(call_anthropic, anthropic_client, prompt)
        labels_a = parse_multilabel(raw_a) if not err_a else []
        anthropic_out.append(labels_a)
        time.sleep(ANTHROPIC_DELAY)

    df = df.copy()
    df["F_openai_labels"] = [json.dumps(x) for x in openai_out]
    df["F_anthropic_labels"] = [json.dumps(x) for x in anthropic_out]
    df["F_openai_match"] = [df.iloc[i]["theme"] in openai_out[i] for i in range(len(df))]
    df["F_anthropic_match"] = [df.iloc[i]["theme"] in anthropic_out[i] for i in range(len(df))]
    df["F_openai_n_labels"] = [len(x) for x in openai_out]
    df["F_anthropic_n_labels"] = [len(x) for x in anthropic_out]
    return df


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame, sample_label: str) -> list[dict]:
    """Build per-theme multi-label recall rows for a given sample."""
    rows = []
    for model in ("openai", "anthropic"):
        match_col = f"F_{model}_match"
        n_col = f"F_{model}_n_labels"
        overall_recall = df[match_col].mean()
        overall_avg_n = df[n_col].mean()
        row = {
            "sample": sample_label,
            "model": model,
            "mean_recall": round(float(overall_recall), 4),
            "mean_n_labels": round(float(overall_avg_n), 3),
        }
        for theme in VALID_THEMES:
            sub = df[df["theme"] == theme]
            if len(sub) == 0:
                continue
            row[f"{theme}_recall"] = round(float(sub[match_col].mean()), 4)
            row[f"{theme}_n"] = int(len(sub))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("APPROACH F — multi-label LLM validation (revised prompts)")
    print(f"Models: {OPENAI_MODEL} + {ANTHROPIC_MODEL}")
    print("=" * 70)

    load_env_file()
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: API keys not found in environment or .env")
        sys.exit(1)

    from openai import OpenAI
    from anthropic import Anthropic
    openai_client = OpenAI()
    anthropic_client = Anthropic()

    # Build prompt template with exemplars
    rng = random.Random(SEED)
    exemplars = build_exemplars(rng, k=N_FEWSHOT_PER_THEME)
    prompt_template = PROMPT_F_MULTILABEL.format(
        definitions=THEME_DEFINITIONS_REVISED,
        multi_examples=MULTI_THEME_EXAMPLES,
        exemplars=exemplars,
        text="{text}",  # placeholder to be formatted per-probe
    )
    print(f"\nBuilt prompt template with {N_FEWSHOT_PER_THEME} exemplars per theme "
          f"({N_FEWSHOT_PER_THEME * len(VALID_THEMES)} total)")
    print(f"Prompt template length: {len(prompt_template)} chars")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Run: ALL 750 public probes (150 per theme) -----
    print(f"\n--- Run: ALL public probes (150 per theme, 750 total) ---")
    df_full = load_all_public_probes()
    print(f"Loaded {len(df_full)} public probes "
          f"(per-theme counts: {dict(df_full['theme'].value_counts())})")
    df_full = run_approach_f(
        df_full, prompt_template, "F-full",
        openai_client=openai_client,
        anthropic_client=anthropic_client,
    )

    # ----- Write per-probe output -----
    full_csv = OUTPUT_DIR / "approach_f_per_probe_full.csv"
    df_full.to_csv(full_csv, index=False)
    print(f"\nSaved per-probe full: {full_csv}")

    # ----- Aggregate results -----
    # Report per-theme recall separately so the 4 clean themes and compensation
    # can be reported independently in the manuscript.
    all_rows = summarize(df_full, "full_750")
    results = pd.DataFrame(all_rows)
    out_csv = OUTPUT_DIR / "approach_f_multilabel_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")

    # ----- Print summary -----
    print("\n" + "=" * 70)
    print("SUMMARY: Approach F — multi-label recall across FULL 750 public probes")
    print("=" * 70)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
