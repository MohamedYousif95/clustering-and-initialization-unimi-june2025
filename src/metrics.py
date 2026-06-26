"""Cluster evaluation metrics used across all experiments."""

import numpy as np
from .distance import nearest_sq_distances, pairwise_sq_distances


def inertia(X: np.ndarray, centroids: np.ndarray) -> float:
    """Within-cluster sum of squared distances, the k-means objective φ (Section 2)."""
    return float(nearest_sq_distances(X, centroids).sum())


def silhouette_sample(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 2000,
    seed: int = 0,
) -> float:
    """
    Estimate average silhouette width on a random subsample of X.

    Silhouette is used only for exploratory analysis alongside inertia;
    the primary comparison metric across all experiments is inertia.

    Parameters
    ----------
    X           : (n, d) feature matrix (standardized)
    labels      : (n,) cluster assignments
    sample_size : number of points to subsample (for speed on large datasets)
    seed        : random seed for subsampling

    Returns
    -------
    float in [-1, 1]; higher is better
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sub, labels_sub = X[idx], labels[idx]
    else:
        X_sub, labels_sub = X, labels

    return _silhouette_score(X_sub, labels_sub)


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute average silhouette width over all points in X."""
    n = len(X)
    # Full pairwise distance matrix, feasible for the subsample sizes used here
    dists = np.sqrt(pairwise_sq_distances(X, X))
    unique_labels = np.unique(labels)

    scores = np.empty(n)
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False

        if not own_mask.any():
            scores[i] = 0.0
            continue

        a = dists[i, own_mask].mean()

        other_means = [
            dists[i, labels == lbl].mean()
            for lbl in unique_labels
            if lbl != own
        ]

        if not other_means:
            scores[i] = 0.0
            continue

        b = min(other_means)
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(scores.mean())
