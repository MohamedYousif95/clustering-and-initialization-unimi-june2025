"""
Squared Euclidean distance primitives used throughout the clustering pipeline.

Paper ref: Section 2, the k-means objective φ = Σ_{x∈X} min_{c∈C} d(x,c)²
is computed directly from these building blocks.
"""

import numpy as np


def pairwise_sq_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distances from every point in X to every center in C.

    Uses the expansion ||x-c||² = ||x||² - 2x·c + ||c||² to avoid an explicit
    loop over centers, keeping memory at O(n·k) rather than O(n·k·d).

    Parameters
    ----------
    X : (n, d)
    C : (k, d)

    Returns
    -------
    D : (n, k)  where D[i, j] = ||X[i] - C[j]||²
    """
    X_sq = np.einsum("ij,ij->i", X, X)[:, np.newaxis]  # (n, 1)
    C_sq = np.einsum("ij,ij->i", C, C)[np.newaxis, :]  # (1, k)
    cross = X @ C.T                                      # (n, k)
    # Clamp to 0 to guard against tiny negative values from floating-point cancellation
    return np.maximum(0.0, X_sq - 2.0 * cross + C_sq)


def nearest_sq_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Squared distance from each point in X to its nearest center in C.

    Parameters
    ----------
    X : (n, d)
    C : (k, d)

    Returns
    -------
    d_min : (n,)  where d_min[i] = min_j ||X[i] - C[j]||²
    """
    return pairwise_sq_distances(X, C).min(axis=1)
