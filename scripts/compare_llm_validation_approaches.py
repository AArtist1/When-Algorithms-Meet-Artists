"""Compare 5 LLM validation approaches for public probe theme classification.

Runs five publication-grade prompting strategies on a balanced 250-probe sample
(50 per theme) using two independent LLMs (OpenAI gpt-4o-mini and Anthropic
claude-sonnet-4.6). Each approach probes a different hypothesis about *why*
zero-shot single-label F1 is low (~0.43-0.46 in the original validation):

  A. Zero-shot baseline (current prompt) -- reproduces the published numbers.
  B. Few-shot with 4 exemplars per theme drawn from artist_perspectives.csv
     (the closest gold-standard data) -- tests whether in-context examples
     help the LLM internalize how *we* draw the boundaries.
  C. Rubric-based with extended inclusion / exclusion / distinguishing
     criteria per theme -- mirrors standard content-analysis codebooks.
  D. Chain-of-thought with multi-label top-k -- tests whether multi-thematic
     probes are the real problem (scored as "extraction theme in top-2").
  E. LLM-as-judge pairwise validation: present probe + assigned theme and ask
     YES/PARTIALLY/NO -- the most lenient defensibility metric.

Outputs:
    figures/final_pipeline/llm_validation_comparison.csv
        rows = (approach, model, theme), columns = precision, recall, f1, n,
        and (for D/E) top2_acc / pass_rate.
    Also prints a summary table to stdout.

Side effects:
    Loads .env file (best-effort) for API keys.
    Makes ~2,500 API calls (250 probes x 5 approaches x 2 models).
    Writes one CSV file. Prints progress.

Estimated cost: ~$2.50 total. Estimated runtime: ~10 minutes.
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
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures" / "final_pipeline"
PROBE_CSV = DATA_DIR / "public_probes.csv"
ARTIST_CSV = DATA_DIR / "artist_perspectives.csv"

VALID_THEMES = ["threat", "utility", "ownership", "transparency", "compensation"]
N_PER_THEME = 10  # 50 total per probe source (pilot run, 30-min budget)
SEED = 42
N_FEWSHOT_PER_THEME = 3

OPENAI_DELAY = 0.1
ANTHROPIC_DELAY = 0.15

OPENAI_MODEL_PRIMARY = "gpt-5.4-mini"
ANTHROPIC_MODEL_PRIMARY = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# .env loader (best-effort)
# ---------------------------------------------------------------------------

def load_env_file() -> None:
    """Load OPENAI_API_KEY and ANTHROPIC_API_KEY from a known .env if not set.

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
# Prompts
# ---------------------------------------------------------------------------

THEME_DEFINITIONS_SHORT = """- threat: concerns about AI threatening artists, jobs, or creative practice
- utility: AI as a useful or positive tool for art and creativity
- ownership: who should own AI-generated artwork, intellectual property rights
- transparency: disclosure of training data, consent, how AI models use artist work
- compensation: payment, revenue sharing, financial models for artists whose work trains AI"""

PROMPT_A_ZEROSHOT = """Classify the following sentence into exactly one of these five themes based on its primary topic:

{defs}

If the sentence does not clearly fit any theme, respond with "none".

Sentence: "{text}"

Respond with ONLY the theme name (one word, lowercase). Do not explain."""

PROMPT_B_FEWSHOT = """Classify the following sentence into exactly one of these five themes based on its primary topic:

{defs}

Here are example artist statements for each theme:

{exemplars}

If the sentence does not clearly fit any theme, respond with "none".

Sentence: "{text}"

Respond with ONLY the theme name (one word, lowercase). Do not explain."""

# Rubric-based prompt (Approach C) -- standard content-analysis codebook style.
THEME_RUBRICS = """**threat**
INCLUDE: Statements expressing concern, fear, anxiety, or warnings that AI is harming or could harm artists, art workers, creative jobs, the art market, the value of human creativity, the survival of art as a profession, or artists' livelihoods. Includes job displacement, undercutting of rates, devaluation of human skill, and existential / professional threat framings.
EXCLUDE: Statements that are primarily about *who owns* AI output (-> ownership), about *being paid* for training data (-> compensation), or about *what training data was used* (-> transparency), even if they imply harm.
DISTINGUISHING: The core claim is "AI hurts / endangers artists or art." If the primary frame is policy, payment, IP, or disclosure, choose another theme.

**utility**
INCLUDE: Statements framing AI as a positive tool, an enabler, an assistant, a collaborator, a productivity boost, a democratizer, a creative partner, an inspiration source, or otherwise endorsing AI's benefits for artists, designers, or creative workflows.
EXCLUDE: Statements that *only* describe what AI can technically do (without a positive evaluative stance) or that frame AI's capability as a *threat* (-> threat).
DISTINGUISHING: The core claim is "AI helps / benefits artists or creative work." Pure capability descriptions without an endorsement should be classified by the topic they actually take a stance on.

**ownership**
INCLUDE: Statements about copyright, intellectual property, authorship, who legally or morally owns AI-generated images, the legal status of AI output, derivative work rights, style ownership, the public domain status of AI art, and infringement claims regarding AI-generated artworks.
EXCLUDE: Statements about disclosing what training data was used (-> transparency) or about paying artists whose work trained models (-> compensation), unless ownership of the *output* is the primary claim.
DISTINGUISHING: The core claim is "who owns / has rights to the AI-generated artwork or its style."

**transparency**
INCLUDE: Statements about disclosure of training datasets, consent of artists whose work was scraped, opt-in / opt-out mechanisms, dataset cards, provenance tracking, audit requirements, the right to know what data was used, and obligations on AI developers to reveal sources.
EXCLUDE: Statements that focus on *paying* the artists whose work was used (-> compensation), or on *who owns the output* (-> ownership). Disclosure obligations are transparency; payment obligations are compensation.
DISTINGUISHING: The core claim is "we should know / they should reveal what data was used to train the model."

**compensation**
INCLUDE: Statements about royalties, licensing fees, revenue sharing, payment to artists whose work trains AI models, monetary models, financial restitution, lawsuits seeking damages, profit-sharing schemes, donation/no-pay models, and the economic terms under which AI companies should compensate artists.
EXCLUDE: Statements about disclosing data sources without a payment claim (-> transparency), about output ownership (-> ownership), or about general economic harm to artists' livelihoods without a specific payment mechanism (-> threat).
DISTINGUISHING: The core claim is "artists should (or should not) be paid / receive money / share profits when their work is used."
"""

PROMPT_C_RUBRIC = """You are a content analyst classifying public-discourse sentences about AI and art into one of five themes. Use this codebook:

{rubrics}

Now classify the following sentence. Apply the inclusion / exclusion criteria above. Pick the SINGLE theme whose distinguishing criterion best matches the sentence's primary claim. If no theme fits, respond with "none".

Sentence: "{text}"

Respond with ONLY the theme name (one word, lowercase). Do not explain."""

PROMPT_D_COT = """Classify the following sentence into the five themes below.

{defs}

Reason step-by-step in JSON:
1. "primary_claim": one short phrase summarizing the sentence's main claim
2. "themes_present": a list of any themes that appear (can be 1-5)
3. "ranked": the same themes ordered from MOST to LEAST relevant (top item is the single best fit)

Sentence: "{text}"

Respond with ONLY a JSON object of the form:
{{"primary_claim": "...", "themes_present": ["...", "..."], "ranked": ["...", "..."]}}
Use only theme names from the list above (lowercase). Do not include any text outside the JSON."""

PROMPT_E_JUDGE = """A sentence has been assigned to a theme by an automated extraction pipeline. Your job is to judge whether the assignment is defensible.

Theme definitions:
{defs}

Sentence: "{text}"
Assigned theme: {theme}

Does this sentence fit the assigned theme? Consider that public-discourse sentences are often multi-thematic; a "PARTIALLY" verdict means the theme is one of several legitimate readings.

Respond with EXACTLY one of: YES, PARTIALLY, NO. No explanation."""


# ---------------------------------------------------------------------------
# Few-shot exemplar construction
# ---------------------------------------------------------------------------

def build_fewshot_exemplars(rng: random.Random, k: int = N_FEWSHOT_PER_THEME) -> str:
    """Pull k exemplars per theme from artist_perspectives.csv.

    Returns a formatted multi-line string suitable for inlining in a prompt.
    Side effects: reads artist_perspectives.csv.
    """
    ap = pd.read_csv(ARTIST_CSV)
    text_col = "perspective_text"
    theme_col = "question_group"
    blocks: list[str] = []
    for theme in VALID_THEMES:
        sub = (
            ap[ap[theme_col] == theme][text_col]
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        rng.shuffle(sub)
        chosen = sub[:k]
        # Truncate very long ones for prompt budget.
        chosen = [c if len(c) <= 240 else c[:237] + "..." for c in chosen]
        ex_lines = "\n".join(f"  - \"{c}\"" for c in chosen)
        blocks.append(f"{theme}:\n{ex_lines}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_balanced_probes(n_per_theme: int = N_PER_THEME, seed: int = SEED) -> pd.DataFrame:
    """Return a balanced sample of public probes with n_per_theme per theme.

    Side effects: reads public_probes.csv.
    """
    df = pd.read_csv(PROBE_CSV)
    text_col = "text" if "text" in df.columns else "probe_text"
    df = df[[c for c in df.columns]].copy()
    df["__text"] = df[text_col]
    samples = []
    for theme in VALID_THEMES:
        sub = df[df["theme"] == theme]
        chosen = sub.sample(n=min(n_per_theme, len(sub)), random_state=seed)
        samples.append(chosen)
    out = pd.concat(samples, ignore_index=True)
    return out.reset_index(drop=True)


def sample_artist_probes(n_per_theme: int = N_PER_THEME, seed: int = SEED) -> pd.DataFrame:
    """Return a balanced sample of artist probes with n_per_theme per theme.

    Uses `question_group` as the ground-truth theme label (survey design).
    Side effects: reads artist_perspectives.csv.
    """
    df = pd.read_csv(ARTIST_CSV)
    text_col = "perspective_text"
    theme_col = "question_group"
    # Canonical theme labels
    df = df[df[theme_col].isin(VALID_THEMES)].copy()
    # Deduplicate by text to avoid sampling identical Likert statements
    df = df.drop_duplicates(subset=[text_col])
    df["__text"] = df[text_col]
    df["theme"] = df[theme_col]
    samples = []
    for theme in VALID_THEMES:
        sub = df[df["theme"] == theme]
        chosen = sub.sample(n=min(n_per_theme, len(sub)), random_state=seed)
        samples.append(chosen)
    out = pd.concat(samples, ignore_index=True)
    return out[["__text", "theme"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _normalize_label(raw: str) -> str:
    """Map an LLM string to a canonical theme label or 'none'/'error'."""
    s = (raw or "").strip().lower()
    s = s.replace('"', "").replace("'", "").strip().strip(".")
    s = s.split()[0] if s else ""
    if s in VALID_THEMES:
        return s
    return "none"


def _call_openai(client, model: str, prompt: str, max_tokens: int = 256) -> str:
    """Single OpenAI chat completion. Side effects: API call."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def _call_anthropic(client, model: str, prompt: str, max_tokens: int = 256) -> str:
    """Single Anthropic message. Side effects: API call."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.content[0].text or "").strip()


def _safe_call(call_fn, *args, **kwargs) -> tuple[str, str | None]:
    """Wrapper that catches exceptions and returns (raw, error_str)."""
    try:
        return call_fn(*args, **kwargs), None
    except Exception as e:
        return "", str(e)


# ---------------------------------------------------------------------------
# Per-approach runners
# ---------------------------------------------------------------------------

def run_single_label_approach(
    texts: list[str],
    prompt_template: str,
    *,
    openai_client,
    anthropic_client,
    extra_format: dict | None = None,
    label_name: str,
) -> dict:
    """Run a single-label approach (A, B, or C) on both LLMs.

    Returns dict with "openai" and "anthropic" -> list[str] labels.
    Side effects: API calls, prints progress.
    """
    extra = extra_format or {}
    openai_labels: list[str] = []
    anthropic_labels: list[str] = []

    for i, t in enumerate(texts):
        prompt = prompt_template.format(text=t, **extra)
        if i % 25 == 0:
            print(f"  [{label_name}] {i}/{len(texts)}")

        raw, err = _safe_call(_call_openai, openai_client, OPENAI_MODEL_PRIMARY, prompt, 32)
        openai_labels.append("error" if err else _normalize_label(raw))
        time.sleep(OPENAI_DELAY)

        raw, err = _safe_call(_call_anthropic, anthropic_client, ANTHROPIC_MODEL_PRIMARY, prompt, 32)
        anthropic_labels.append("error" if err else _normalize_label(raw))
        time.sleep(ANTHROPIC_DELAY)

    return {"openai": openai_labels, "anthropic": anthropic_labels}


def _parse_cot_json(raw: str) -> list[str]:
    """Parse chain-of-thought JSON output to a ranked theme list.

    Tolerates markdown fencing and trailing prose. Returns canonical theme
    names only.
    """
    if not raw:
        return []
    s = raw.strip()
    # Strip markdown fences if present.
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # Greedy: extract first {...} block.
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    ranked = obj.get("ranked") or obj.get("themes_present") or []
    if not isinstance(ranked, list):
        return []
    out: list[str] = []
    for x in ranked:
        if isinstance(x, str):
            t = x.strip().lower()
            if t in VALID_THEMES and t not in out:
                out.append(t)
    return out


def run_cot_approach(
    texts: list[str],
    *,
    openai_client,
    anthropic_client,
) -> dict:
    """Approach D: chain-of-thought multi-label.

    Returns dict with per-model lists of *ranked theme lists*.
    Side effects: API calls, prints progress.
    """
    openai_ranked: list[list[str]] = []
    anthropic_ranked: list[list[str]] = []
    for i, t in enumerate(texts):
        prompt = PROMPT_D_COT.format(text=t, defs=THEME_DEFINITIONS_SHORT)
        if i % 25 == 0:
            print(f"  [D-CoT] {i}/{len(texts)}")

        raw, err = _safe_call(_call_openai, openai_client, OPENAI_MODEL_PRIMARY, prompt, 256)
        openai_ranked.append([] if err else _parse_cot_json(raw))
        time.sleep(OPENAI_DELAY)

        raw, err = _safe_call(_call_anthropic, anthropic_client, ANTHROPIC_MODEL_PRIMARY, prompt, 256)
        anthropic_ranked.append([] if err else _parse_cot_json(raw))
        time.sleep(ANTHROPIC_DELAY)
    return {"openai": openai_ranked, "anthropic": anthropic_ranked}


def _parse_judge(raw: str) -> str:
    """Map a judge response to YES / PARTIALLY / NO / NONE."""
    if not raw:
        return "NONE"
    s = raw.strip().upper()
    s = re.sub(r"[^A-Z]", " ", s)
    if "PARTIALLY" in s or "PARTIAL" in s:
        return "PARTIALLY"
    if "YES" in s:
        return "YES"
    if "NO" in s:
        return "NO"
    return "NONE"


def run_judge_approach(
    texts: list[str],
    assigned_themes: list[str],
    *,
    openai_client,
    anthropic_client,
) -> dict:
    """Approach E: pairwise LLM-as-judge.

    Returns dict with per-model lists of YES/PARTIALLY/NO/NONE verdicts.
    Side effects: API calls, prints progress.
    """
    openai_verdicts: list[str] = []
    anthropic_verdicts: list[str] = []
    for i, (t, theme) in enumerate(zip(texts, assigned_themes)):
        prompt = PROMPT_E_JUDGE.format(text=t, theme=theme, defs=THEME_DEFINITIONS_SHORT)
        if i % 25 == 0:
            print(f"  [E-Judge] {i}/{len(texts)}")

        raw, err = _safe_call(_call_openai, openai_client, OPENAI_MODEL_PRIMARY, prompt, 16)
        openai_verdicts.append("ERROR" if err else _parse_judge(raw))
        time.sleep(OPENAI_DELAY)

        raw, err = _safe_call(_call_anthropic, anthropic_client, ANTHROPIC_MODEL_PRIMARY, prompt, 16)
        anthropic_verdicts.append("ERROR" if err else _parse_judge(raw))
        time.sleep(ANTHROPIC_DELAY)
    return {"openai": openai_verdicts, "anthropic": anthropic_verdicts}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_theme_prf(assigned: list[str], predicted: list[str]) -> pd.DataFrame:
    """Per-theme precision/recall/F1 for single-label predictions."""
    rows = []
    for theme in VALID_THEMES:
        tp = sum(1 for a, p in zip(assigned, predicted) if a == theme and p == theme)
        fp = sum(1 for a, p in zip(assigned, predicted) if a != theme and p == theme)
        fn = sum(1 for a, p in zip(assigned, predicted) if a == theme and p != theme)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "theme": theme,
            "n_assigned": sum(1 for a in assigned if a == theme),
            "n_predicted": sum(1 for p in predicted if p == theme),
            "tp": tp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
    return pd.DataFrame(rows)


def per_theme_topk(assigned: list[str], ranked_lists: list[list[str]], k: int = 2) -> pd.DataFrame:
    """For Approach D: per-theme top-k accuracy + per-theme P/R/F1 from rank-1."""
    rows = []
    rank1 = [r[0] if r else "none" for r in ranked_lists]
    for theme in VALID_THEMES:
        idxs = [i for i, a in enumerate(assigned) if a == theme]
        n = len(idxs)
        # Top-k accuracy: assigned theme appears in top-k of ranked list
        topk_hits = sum(1 for i in idxs if theme in (ranked_lists[i][:k] if ranked_lists[i] else []))
        topk_acc = topk_hits / n if n else 0.0
        # Rank-1 P/R/F1
        tp = sum(1 for i, a in enumerate(assigned) if a == theme and rank1[i] == theme)
        fp = sum(1 for i, a in enumerate(assigned) if a != theme and rank1[i] == theme)
        fn = sum(1 for i, a in enumerate(assigned) if a == theme and rank1[i] != theme)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "theme": theme,
            "n_assigned": n,
            "n_predicted_rank1": sum(1 for r in rank1 if r == theme),
            "tp": tp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "top2_accuracy": round(topk_acc, 4),
        })
    return pd.DataFrame(rows)


def per_theme_judge(assigned: list[str], verdicts: list[str]) -> pd.DataFrame:
    """For Approach E: per-theme YES rate and YES+PARTIALLY pass rate."""
    rows = []
    for theme in VALID_THEMES:
        idxs = [i for i, a in enumerate(assigned) if a == theme]
        n = len(idxs)
        yes_n = sum(1 for i in idxs if verdicts[i] == "YES")
        partial_n = sum(1 for i in idxs if verdicts[i] == "PARTIALLY")
        no_n = sum(1 for i in idxs if verdicts[i] == "NO")
        pass_rate = (yes_n + partial_n) / n if n else 0.0
        rows.append({
            "theme": theme,
            "n_assigned": n,
            "yes": yes_n,
            "partially": partial_n,
            "no": no_n,
            "yes_rate": round(yes_n / n, 4) if n else 0.0,
            "pass_rate": round(pass_rate, 4),
            # Map to a P/R/F1-style "f1" for table parity (yes-only treated as recall;
            # precision is undefined here so we use pass_rate as a single summary)
            "precision": round(yes_n / n, 4) if n else 0.0,
            "recall": round(pass_rate, 4),
            "f1": round(pass_rate, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all 5 approaches on a balanced 250-probe sample and write CSV."""
    print("=" * 70)
    print("LLM VALIDATION APPROACH COMPARISON  (5 approaches x 2 models)")
    print("=" * 70)

    load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set after .env load. Aborting.")
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set after .env load. Aborting.")
        sys.exit(1)

    # Initialize clients
    from openai import OpenAI
    from anthropic import Anthropic
    openai_client = OpenAI()
    anthropic_client = Anthropic()

    # Sample
    df = sample_balanced_probes()
    texts = df["__text"].astype(str).tolist()
    assigned = df["theme"].astype(str).tolist()
    print(f"\nLoaded {len(df)} probes ({N_PER_THEME} per theme)")

    # Few-shot exemplars
    rng = random.Random(SEED)
    exemplars = build_fewshot_exemplars(rng)
    print(f"Built {N_FEWSHOT_PER_THEME * len(VALID_THEMES)} few-shot exemplars from artist_perspectives.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # ----- Approach A: zero-shot baseline -----
    print("\n--- Approach A: zero-shot baseline (public probes) ---")
    A = run_single_label_approach(
        texts,
        PROMPT_A_ZEROSHOT,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        extra_format={"defs": THEME_DEFINITIONS_SHORT},
        label_name="A",
    )
    df["A_openai"] = A["openai"]
    df["A_anthropic"] = A["anthropic"]
    for model_key, preds in [("openai", A["openai"]), ("anthropic", A["anthropic"])]:
        m = per_theme_prf(assigned, preds)
        for _, r in m.iterrows():
            all_rows.append({"probe_source": "public", "approach": "A_zeroshot", "model": model_key, **r.to_dict()})

    # ----- Approach B: few-shot 4-per-theme -----
    print("\n--- Approach B: few-shot (4 exemplars per theme) ---")
    B = run_single_label_approach(
        texts,
        PROMPT_B_FEWSHOT,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        extra_format={"defs": THEME_DEFINITIONS_SHORT, "exemplars": exemplars},
        label_name="B",
    )
    df["B_openai"] = B["openai"]
    df["B_anthropic"] = B["anthropic"]
    for model_key, preds in [("openai", B["openai"]), ("anthropic", B["anthropic"])]:
        m = per_theme_prf(assigned, preds)
        for _, r in m.iterrows():
            all_rows.append({"probe_source": "public", "approach": "B_fewshot", "model": model_key, **r.to_dict()})

    # ----- Approach C: rubric-based -----
    print("\n--- Approach C: rubric-based (extended codebook) ---")
    C = run_single_label_approach(
        texts,
        PROMPT_C_RUBRIC,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        extra_format={"rubrics": THEME_RUBRICS},
        label_name="C",
    )
    df["C_openai"] = C["openai"]
    df["C_anthropic"] = C["anthropic"]
    for model_key, preds in [("openai", C["openai"]), ("anthropic", C["anthropic"])]:
        m = per_theme_prf(assigned, preds)
        for _, r in m.iterrows():
            all_rows.append({"probe_source": "public", "approach": "C_rubric", "model": model_key, **r.to_dict()})

    # ----- Approach D: chain-of-thought multi-label -----
    print("\n--- Approach D: chain-of-thought multi-label ---")
    D = run_cot_approach(
        texts,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
    )
    df["D_openai_ranked"] = [json.dumps(r) for r in D["openai"]]
    df["D_anthropic_ranked"] = [json.dumps(r) for r in D["anthropic"]]
    for model_key, ranked in [("openai", D["openai"]), ("anthropic", D["anthropic"])]:
        m = per_theme_topk(assigned, ranked, k=2)
        for _, r in m.iterrows():
            all_rows.append({"probe_source": "public", "approach": "D_cot_multilabel", "model": model_key, **r.to_dict()})

    # ----- Approach E: LLM-as-judge -----
    print("\n--- Approach E: LLM-as-judge pairwise ---")
    E = run_judge_approach(
        texts,
        assigned,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
    )
    df["E_openai_verdict"] = E["openai"]
    df["E_anthropic_verdict"] = E["anthropic"]
    for model_key, verdicts in [("openai", E["openai"]), ("anthropic", E["anthropic"])]:
        m = per_theme_judge(assigned, verdicts)
        for _, r in m.iterrows():
            all_rows.append({"probe_source": "public", "approach": "E_llm_judge", "model": model_key, **r.to_dict()})

    # ----- ARTIST PROBE CEILING BASELINE (Approach A zero-shot only) -----
    # Ground truth: the `question_group` column from the Lovato et al. 2024 survey.
    # Expected outcome: near-ceiling F1 for threat/utility/ownership/transparency
    # (template wording contains theme word); more meaningful F1 for compensation.
    print("\n--- Artist-probe ceiling baseline (Approach A on artist probes) ---")
    df_art = sample_artist_probes()
    art_texts = df_art["__text"].astype(str).tolist()
    art_assigned = df_art["theme"].astype(str).tolist()
    print(f"Loaded {len(df_art)} artist probes ({N_PER_THEME} per theme)")

    A_art = run_single_label_approach(
        art_texts,
        PROMPT_A_ZEROSHOT,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        extra_format={"defs": THEME_DEFINITIONS_SHORT},
        label_name="A-artist",
    )
    df_art["A_openai"] = A_art["openai"]
    df_art["A_anthropic"] = A_art["anthropic"]
    for model_key, preds in [("openai", A_art["openai"]), ("anthropic", A_art["anthropic"])]:
        m = per_theme_prf(art_assigned, preds)
        for _, r in m.iterrows():
            all_rows.append({"probe_source": "artist", "approach": "A_zeroshot", "model": model_key, **r.to_dict()})

    # Write outputs
    results = pd.DataFrame(all_rows)
    out_csv = OUTPUT_DIR / "llm_validation_comparison.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved per-approach metrics: {out_csv}")

    per_probe_csv = OUTPUT_DIR / "llm_validation_comparison_per_probe.csv"
    df.to_csv(per_probe_csv, index=False)
    print(f"Saved per-probe predictions: {per_probe_csv}")

    # Also write per-probe artist predictions
    art_per_probe_csv = OUTPUT_DIR / "llm_validation_comparison_per_probe_artist.csv"
    df_art.to_csv(art_per_probe_csv, index=False)
    print(f"Saved per-probe artist predictions: {art_per_probe_csv}")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SUMMARY: per-approach mean F1 (single-label) / pass-rate (E)")
    print("=" * 70)
    summary_rows = []
    summary_targets = [
        ("public", "A_zeroshot"),
        ("public", "B_fewshot"),
        ("public", "C_rubric"),
        ("public", "D_cot_multilabel"),
        ("public", "E_llm_judge"),
        ("artist", "A_zeroshot"),  # ceiling baseline
    ]
    for probe_source, approach in summary_targets:
        for model_key in ["openai", "anthropic"]:
            sub = results[
                (results["probe_source"] == probe_source)
                & (results["approach"] == approach)
                & (results["model"] == model_key)
            ]
            if sub.empty:
                continue
            mean_f1 = sub["f1"].mean()
            row = {
                "probe_source": probe_source,
                "approach": approach,
                "model": model_key,
                "mean_f1": round(mean_f1, 4),
            }
            for _, r in sub.iterrows():
                row[f"{r['theme']}_f1"] = r["f1"]
            if approach == "D_cot_multilabel" and "top2_accuracy" in sub.columns:
                row["mean_top2_acc"] = round(sub["top2_accuracy"].mean(), 4)
            if approach == "E_llm_judge" and "pass_rate" in sub.columns:
                row["mean_pass_rate"] = round(sub["pass_rate"].mean(), 4)
                row["mean_yes_rate"] = round(sub["yes_rate"].mean(), 4)
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "llm_validation_comparison_summary.csv", index=False)
    print(f"\nSaved summary: {OUTPUT_DIR / 'llm_validation_comparison_summary.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
