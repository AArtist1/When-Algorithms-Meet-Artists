"""Regenerate Figure 3 (macro-thematic distribution) with canonical numbers.

Uses the verified canonical allocation from figures/final_pipeline/all_metrics.csv
and the macro-theme cluster mapping (fixed for k=20, random_state=42):
    Philosophy of Creativity: C1, C4, C11, C13, C17  (artist probes: 551 + 0 + 0 + 643 + 0 = 1194)
    Practice & Pedagogy:      C3, C7, C14, C16       (artist probes: 8 + 56 + 0 + 0 = 64)
    Technical Genealogy:      C2, C5, C15, C19       (artist probes: 0 + 0 + 0 + 1 = 1)
    Governance & Rights:      C8, C10, C18           (artist probes: 0 + 0 + 0 = 0)
    Institutions & Markets:   C0, C6, C9, C12        (artist probes: 0 + 0 + 0 + 0 = 0)

Public corpus macro-theme shares (from final pipeline clustering of 1,736 chunks):
    Philosophy 38.1%, Practice 34.8%, Technical 14.7%, Governance 7.5%, Institutions 4.9%

Side effects:
    Overwrites figure_3_macrotheme_distributions.{png,pdf} in:
        figures/manuscript/
        jan_2026_manuscript/NMS2026/figures/
        jan_2026_manuscript/NatureCollection2026/figures/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIRS = [
    ROOT / "figures" / "manuscript",
    ROOT / "jan_2026_manuscript" / "NMS2026" / "figures",
    ROOT / "jan_2026_manuscript" / "NatureCollection2026" / "figures",
]

MACRO_ORDER = [
    "Philosophy of Creativity",
    "Practice & Pedagogy",
    "Technical Genealogy",
    "Governance & Rights",
    "Institutions & Markets",
]

# Canonical public corpus shares (n=1,736 chunks)
PUB_PCT = {
    "Philosophy of Creativity": 38.1,
    "Practice & Pedagogy": 34.8,
    "Technical Genealogy": 14.7,
    "Governance & Rights": 7.5,
    "Institutions & Markets": 4.9,
}

# Canonical artist probe shares (n=1,259 probes)
# Derived from cluster-level counts: C1=551, C13=643 (Philosophy), C3=8, C7=56 (Practice),
# C19=1 (Technical), all others 0.
ART_COUNTS = {
    "Philosophy of Creativity": 551 + 643,
    "Practice & Pedagogy": 8 + 56,
    "Technical Genealogy": 1,
    "Governance & Rights": 0,
    "Institutions & Markets": 0,
}
ART_TOTAL = sum(ART_COUNTS.values())
assert ART_TOTAL == 1259, f"expected 1259 artist probes, got {ART_TOTAL}"
ART_PCT = {k: 100.0 * v / ART_TOTAL for k, v in ART_COUNTS.items()}

PUB_TOTAL = 1736


def save_figure(fig: Figure, name: str, out_dirs: list[Path]) -> None:
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")


def main() -> None:
    x = np.arange(len(MACRO_ORDER))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars_pub = ax.bar(
        x - width / 2,
        [PUB_PCT[m] for m in MACRO_ORDER],
        width,
        color="#4C72B0",
        alpha=0.75,
        label=f"Public discourse (n={PUB_TOTAL:,})",
    )
    bars_art = ax.bar(
        x + width / 2,
        [ART_PCT[m] for m in MACRO_ORDER],
        width,
        color="#e74c3c",
        alpha=0.75,
        label=f"Artist probes (n={ART_TOTAL:,})",
    )

    for bar in bars_pub:
        h = bar.get_height()
        if h > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
            )
    for bar in bars_art:
        h = bar.get_height()
        if h > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MACRO_ORDER, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Percentage of total", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax.set_title(
        "Macro-Thematic Distribution: Public Discourse vs Artist Probes",
        fontsize=12.5, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    save_figure(fig, "figure_3_macrotheme_distributions", OUT_DIRS)
    plt.close()
    print("Saved figure_3_macrotheme_distributions to:")
    for d in OUT_DIRS:
        print(f"  {d}")


if __name__ == "__main__":
    main()
