"""
Seeding strategies for k-means.

Both functions share the same signature:
    (X, k, rng) -> (k, d) array of initial centers

Paper ref:
    random_init: baseline described in Section 1 / Theorem 1.1
    plusplus_init: Algorithm 1, Section 3 (D²-weighted seeding)
"""

import numpy as np
from .distance import nearest_sq_distances


def random_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Choose k centers uniformly at random without replacement.

    This is the standard baseline; it provides no approximation guarantee
    and can yield solutions O(k) times worse than optimal (Theorem 1.1).
    """
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    D²-weighted seeding, k-means++ Algorithm 1 (Section 3, Arthur & Vassilvitskii 2007).

    Step 1: pick c₁ uniformly at random from X.
    Step 2: for i = 2 … k, sample the next center from X with probability
            proportional to D(x)², the squared distance to the nearest
            already-chosen center (the D²-sampling distribution).
    Step 3: return the k chosen centers.

    This initialization alone gives E[φ] ≤ 8(ln k + 2) · φ_OPT (Theorem 3.1)
    before any Lloyd iterations are run.
    """
    n = len(X)
    first_idx = int(rng.integers(n))
    centers = [X[first_idx]]

    for _ in range(k - 1):
        C = np.array(centers)                          # (i, d)
        sq_dists = nearest_sq_distances(X, C)          # (n,)  D(x)²
        total = sq_dists.sum()
        if total == 0.0:
            # All remaining points coincide with already-chosen centers;
            # fall back to uniform sampling for the remaining slots.
            probs = np.ones(n) / n
        else:
            probs = sq_dists / total
        next_idx = int(rng.choice(n, p=probs))
        centers.append(X[next_idx])

    return np.array(centers)                           # (k, d)
