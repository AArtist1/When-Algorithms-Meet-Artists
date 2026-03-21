"""Visualization functions for the analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
    alpha: float = 0.6,
    s: int = 15,
) -> plt.Figure:
    """Scatter plot of 2D points colored by cluster label.

    Args:
        X_2d: Array of shape (n, 2).
        labels: Cluster labels for each point.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        alpha: Point transparency.
        s: Point size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[i]], label=f"Cluster {label}",
            alpha=alpha, s=s, edgecolors="none",
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig


def plot_artist_concentration(
    public_topic_dist: np.ndarray,
    artist_topic_dist: np.ndarray,
    topic_labels: list[str],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Bar chart comparing public discourse and artist probe distributions across topics.

    Args:
        public_topic_dist: Fraction of public corpus in each topic.
        artist_topic_dist: Fraction of artist probes in each topic.
        topic_labels: Labels for each topic.
        save_path: If provided, save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    x = np.arange(len(topic_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, public_topic_dist, width, label="Public Discourse", color="#4C72B0")
    ax.bar(x + width / 2, artist_topic_dist, width, label="Artist Probes", color="#DD8452")

    ax.set_xlabel("Topic")
    ax.set_ylabel("Proportion")
    ax.set_title("Artist Probe Concentration Across Public Discourse Topics")
    ax.set_xticks(x)
    ax.set_xticklabels(topic_labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_salience_ratios(
    themes: list[str],
    ratios_vs_public: list[float],
    ratios_vs_probes: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Bar chart of salience ratios by theme.

    Args:
        themes: Theme names.
        ratios_vs_public: Salience ratios vs raw public discourse.
        ratios_vs_probes: Salience ratios vs style-matched public probes (optional).
        save_path: If provided, save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    x = np.arange(len(themes))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, ratios_vs_public, width, label="vs. Public Discourse", color="#4C72B0")

    if ratios_vs_probes is not None:
        ax.bar(x + width / 2, ratios_vs_probes, width, label="vs. Public Probes", color="#DD8452")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Theme")
    ax.set_ylabel("Salience Ratio")
    ax.set_title("Theme-Based Salience Ratios (90% Coverage Threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels(themes, fontsize=11)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def summarize_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    coord_cols: list[str],
) -> pd.DataFrame:
    """Compute cluster sizes and centroids.

    Returns:
        DataFrame with cluster_id, n_points, percentage, and centroid coordinates.
    """
    groups = df.groupby(cluster_col)
    rows = []
    for label, group in groups:
        centroid = group[coord_cols].mean().values
        rows.append({
            "cluster_id": label,
            "n_points": len(group),
            "percentage": 100.0 * len(group) / len(df),
            **{f"centroid_{col}": float(centroid[i]) for i, col in enumerate(coord_cols)},
        })
    return pd.DataFrame(rows).sort_values("n_points", ascending=False)


def add_extreme_flags(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    q: float = 0.05,
) -> pd.DataFrame:
    """Mark extreme-value points based on quantile thresholds.

    Adds boolean columns: extreme_high_x, extreme_low_x, extreme_high_y, extreme_low_y.
    """
    df = df.copy()
    x_lo, x_hi = df[x_col].quantile(q), df[x_col].quantile(1 - q)
    y_lo, y_hi = df[y_col].quantile(q), df[y_col].quantile(1 - q)

    df["extreme_high_x"] = df[x_col] >= x_hi
    df["extreme_low_x"] = df[x_col] <= x_lo
    df["extreme_high_y"] = df[y_col] >= y_hi
    df["extreme_low_y"] = df[y_col] <= y_lo
    return df
