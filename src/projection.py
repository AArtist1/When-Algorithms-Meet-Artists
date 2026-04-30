"""MLP projection head for mapping embeddings into consensus UMAP space."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def train_projection_head(
    X_embed: np.ndarray,
    Y_consensus: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (1024, 512, 256, 128, 64),
    activation: str = "relu",
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    max_iter: int = 1000,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Train an MLP to map from embedding space to consensus UMAP coordinates.

    Args:
        X_embed: High-dimensional embeddings, shape (n, embedding_dim).
        Y_consensus: Consensus UMAP coordinates, shape (n, n_components).
        hidden_layer_sizes: MLP hidden layer architecture.
        activation: Activation function.
        alpha: L2 regularization strength.
        learning_rate_init: Initial learning rate.
        max_iter: Maximum training iterations.
        early_stopping: Whether to use early stopping.
        validation_fraction: Fraction for early stopping validation.
        test_size: Fraction held out for final evaluation.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary with keys: model, scaler_X, scaler_Y, r2_train, r2_val,
        X_test, Y_test, Y_pred_test, n_train, n_val.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_embed, Y_consensus, test_size=test_size, random_state=random_state
    )

    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)

    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(Y_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    mlp.fit(X_train_s, Y_train_s)

    r2_train = float(mlp.score(X_train_s, Y_train_s))
    Y_test_s = scaler_Y.transform(Y_test)
    r2_val = float(mlp.score(X_test_s, Y_test_s))

    Y_pred_test_s = mlp.predict(X_test_s)
    Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_s)

    return {
        "model": mlp,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "r2_train": r2_train,
        "r2_val": r2_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred_test": Y_pred_test,
        "n_train": len(X_train),
        "n_val": len(X_test),
    }


def project_to_consensus_space(
    X_new: np.ndarray,
    model: MLPRegressor,
    scaler_X: StandardScaler,
    scaler_Y: StandardScaler,
) -> np.ndarray:
    """Project new embeddings into the consensus UMAP coordinate space.

    Args:
        X_new: New embeddings, shape (n_new, embedding_dim).
        model: Trained MLPRegressor.
        scaler_X: Fitted StandardScaler for input features.
        scaler_Y: Fitted StandardScaler for output coordinates.

    Returns:
        Projected coordinates, shape (n_new, n_components).
    """
    X_scaled = scaler_X.transform(X_new)
    Y_pred_scaled = model.predict(X_scaled)
    return scaler_Y.inverse_transform(Y_pred_scaled)


def knn_preservation(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    k: int = 15,
) -> float:
    """Compute k-NN neighborhood preservation between true and predicted coordinates.

    Returns:
        Fraction of k-nearest neighbors preserved (0 to 1).
    """
    from sklearn.neighbors import NearestNeighbors

    nn_true = NearestNeighbors(n_neighbors=k + 1).fit(Y_true)
    nn_pred = NearestNeighbors(n_neighbors=k + 1).fit(Y_pred)

    _, idx_true = nn_true.kneighbors(Y_true)
    _, idx_pred = nn_pred.kneighbors(Y_pred)

    # Exclude self (index 0)
    idx_true = idx_true[:, 1:]
    idx_pred = idx_pred[:, 1:]

    overlaps = 0
    for i in range(len(Y_true)):
        overlaps += len(set(idx_true[i]) & set(idx_pred[i]))

    return overlaps / (len(Y_true) * k)
