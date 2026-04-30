"""MLP projection head sensitivity analysis.

Evaluates the projection head's reliability by computing:
    1. R-squared across multiple cross-validation folds
    2. k-NN neighborhood preservation at multiple k values
    3. Cluster assignment stability under projection noise

This addresses the reviewer concern that R-squared=0.73 means 27%
unexplained variance, and quantifies how that affects downstream results.

Output:
    figures/mlp_sensitivity/knn_preservation_table.csv
    figures/mlp_sensitivity/cv_r2_summary.csv
    figures/mlp_sensitivity/cluster_stability.csv
    Console output with formatted summary.

Side effects:
    Writes CSV files. Prints to stdout.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus_umap import (
    distance_matrix_consensus,
    run_umap_multi_seed,
    umap_from_precomputed_distances,
)
from src.clustering import run_kmeans
from src.projection import knn_preservation, train_projection_head, project_to_consensus_space


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS = [
    137, 85, 127, 59, 195, 243, 170, 77, 186, 79,
    69, 42, 240, 105, 199, 91, 151, 82, 177, 234,
    46, 101, 34, 175, 108, 81, 176, 241, 20, 53,
]

ROOT = Path(__file__).parent.parent
EDIT_DIR = ROOT / "When-Algorithms-Meet-Artists-EDIT"
OUTPUT_DIR = ROOT / "figures" / "mlp_sensitivity"

KNN_K_VALUES = [5, 10, 15, 30, 50, 72]
N_CV_FOLDS = 5
N_NOISE_TRIALS = 20


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def get_cv_r2_scores(
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, float | list[float]]:
    """Compute R-squared across k-fold cross-validation.

    Trains the same MLP architecture on each fold and reports
    per-fold and aggregate R-squared statistics.

    Args:
        X: Input embeddings, shape (n, d_in).
        Y: Target consensus coordinates, shape (n, d_out).
        n_folds: Number of CV folds.
        random_state: Random seed for fold splitting.

    Returns:
        Dict with mean, std, min, max, and per-fold R-squared values.

    Side effects:
        Prints fold-level progress.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train_s = scaler_x.fit_transform(X_train)
        X_val_s = scaler_x.transform(X_val)
        Y_train_s = scaler_y.fit_transform(Y_train)
        Y_val_s = scaler_y.transform(Y_val)

        mlp = MLPRegressor(
            hidden_layer_sizes=(512, 128),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(X_train_s, Y_train_s)
        r2 = float(mlp.score(X_val_s, Y_val_s))
        fold_scores.append(r2)
        print(f"  Fold {fold_idx + 1}/{n_folds}: R2 = {r2:.4f}")

    return {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "min": float(np.min(fold_scores)),
        "max": float(np.max(fold_scores)),
        "per_fold": fold_scores,
    }


def get_knn_preservation_multi_k(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    k_values: list[int],
) -> dict[int, float]:
    """Compute k-NN preservation at multiple k values.

    Args:
        Y_true: True consensus coordinates, shape (n, d).
        Y_pred: Predicted consensus coordinates, shape (n, d).
        k_values: List of k values to evaluate.

    Returns:
        Dict mapping k to preservation fraction.

    Side effects:
        None.
    """
    results = {}
    for k in k_values:
        k_eff = min(k, len(Y_true) - 1)
        pres = knn_preservation(Y_true, Y_pred, k=k_eff)
        results[k] = pres
    return results


def get_cluster_assignment_stability(
    X_artist: np.ndarray,
    X_public: np.ndarray,
    consensus_8d: np.ndarray,
    n_trials: int = 20,
) -> dict[str, float]:
    """Test how stable artist cluster assignments are across projection head retrains.

    Trains the MLP multiple times with different random seeds and measures
    how often each artist probe gets the same cluster assignment.

    Args:
        X_artist: Artist embeddings, shape (n_artist, d).
        X_public: Public embeddings, shape (n_public, d).
        consensus_8d: Public consensus coordinates.
        n_trials: Number of retrain trials.

    Returns:
        Dict with mean_agreement (fraction of probes with majority label),
        modal_pct (mean fraction assigned to most common label per probe),
        and n_trials.

    Side effects:
        Prints progress.
    """
    labels_pub, _ = run_kmeans(consensus_8d, n_clusters=28, metric="euclidean")
    nn = NearestNeighbors(n_neighbors=1).fit(consensus_8d)

    all_labels = []
    for trial in range(n_trials):
        proj = train_projection_head(X_public, consensus_8d, random_state=trial)
        art_8d = project_to_consensus_space(X_artist, proj['model'],
                                             proj['scaler_X'], proj['scaler_Y'])
        _, idx = nn.kneighbors(art_8d)
        trial_labels = labels_pub[idx.flatten()]
        all_labels.append(trial_labels)
        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")

    all_labels = np.array(all_labels)  # shape: (n_trials, n_artist)

    # For each artist probe, what fraction of trials agree on the most common label?
    modal_fractions = []
    for i in range(all_labels.shape[1]):
        unique, counts = np.unique(all_labels[:, i], return_counts=True)
        modal_fractions.append(counts.max() / n_trials)

    return {
        "mean_modal_fraction": float(np.mean(modal_fractions)),
        "median_modal_fraction": float(np.median(modal_fractions)),
        "min_modal_fraction": float(np.min(modal_fractions)),
        "pct_above_80": float(np.mean(np.array(modal_fractions) > 0.80)),
        "pct_above_90": float(np.mean(np.array(modal_fractions) > 0.90)),
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load and run consensus UMAP
    print("Loading embeddings...")
    X_public = np.load(EDIT_DIR / "embeddings" / "Chunks_of_Public_250words_with_25word_overlap_e5_embeddings.npy")
    X_artist = np.load(EDIT_DIR / "embeddings" / "Artists_Perspectives_e5_embeddings.npy")

    print(f"Running consensus UMAP ({len(SEEDS)} seeds)...")
    ue = run_umap_multi_seed(X_public, seeds=SEEDS, n_components=8,
                              n_neighbors=27, min_dist=0.1, metric="cosine")
    D = distance_matrix_consensus(ue, metric="euclidean")
    consensus_8d, _ = umap_from_precomputed_distances(D, n_components=8,
                                                       n_neighbors=27, min_dist=0.1)

    # --- 1. Cross-validated R-squared ---
    print(f"\n{'='*60}")
    print(f"1. CROSS-VALIDATED R-SQUARED ({N_CV_FOLDS}-fold)")
    print(f"{'='*60}")

    cv_results = get_cv_r2_scores(X_public, consensus_8d, n_folds=N_CV_FOLDS)
    print(f"\nR2 mean: {cv_results['mean']:.4f} +/- {cv_results['std']:.4f}")
    print(f"R2 range: [{cv_results['min']:.4f}, {cv_results['max']:.4f}]")

    df_cv = pd.DataFrame({
        "fold": list(range(1, N_CV_FOLDS + 1)),
        "r2": cv_results["per_fold"],
    })
    df_cv.to_csv(OUTPUT_DIR / "cv_r2_summary.csv", index=False)
    print(f"Saved: cv_r2_summary.csv")

    # --- 2. k-NN preservation at multiple k ---
    print(f"\n{'='*60}")
    print(f"2. K-NN PRESERVATION AT MULTIPLE K VALUES")
    print(f"{'='*60}")

    # Train standard projection head for k-NN analysis
    proj = train_projection_head(X_public, consensus_8d, random_state=42)
    Y_pred = proj['model'].predict(
        proj['scaler_X'].transform(X_public)
    )
    Y_pred = proj['scaler_Y'].inverse_transform(Y_pred)

    knn_results = get_knn_preservation_multi_k(consensus_8d, Y_pred, KNN_K_VALUES)
    print(f"\nk-NN preservation:")
    for k, pres in sorted(knn_results.items()):
        print(f"  k={k:3d}: {pres:.4f} ({pres*100:.1f}%)")

    df_knn = pd.DataFrame([{"k": k, "preservation": p} for k, p in sorted(knn_results.items())])
    df_knn.to_csv(OUTPUT_DIR / "knn_preservation_table.csv", index=False)
    print(f"Saved: knn_preservation_table.csv")

    # --- 3. Cluster assignment stability ---
    print(f"\n{'='*60}")
    print(f"3. CLUSTER ASSIGNMENT STABILITY ({N_NOISE_TRIALS} retrain trials)")
    print(f"{'='*60}")

    stability = get_cluster_assignment_stability(
        X_artist, X_public, consensus_8d, n_trials=N_NOISE_TRIALS
    )
    print(f"\nMean modal fraction: {stability['mean_modal_fraction']:.3f}")
    print(f"Median modal fraction: {stability['median_modal_fraction']:.3f}")
    print(f"Min modal fraction: {stability['min_modal_fraction']:.3f}")
    print(f"Probes with >80% agreement: {stability['pct_above_80']*100:.1f}%")
    print(f"Probes with >90% agreement: {stability['pct_above_90']*100:.1f}%")

    df_stab = pd.DataFrame([stability])
    df_stab.to_csv(OUTPUT_DIR / "cluster_stability.csv", index=False)
    print(f"Saved: cluster_stability.csv")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY FOR MANUSCRIPT")
    print(f"{'='*60}")
    print(f"\nProjection head validation:")
    print(f"  R2 = {cv_results['mean']:.2f} +/- {cv_results['std']:.2f} ({N_CV_FOLDS}-fold CV)")
    print(f"  k-NN preservation: {knn_results[5]*100:.1f}% (k=5), {knn_results[15]*100:.1f}% (k=15), {knn_results[72]*100:.1f}% (k=72)")
    print(f"  Cluster stability: {stability['pct_above_80']*100:.0f}% of artist probes receive the same")
    print(f"    cluster label in >80% of {N_NOISE_TRIALS} retrain trials")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
