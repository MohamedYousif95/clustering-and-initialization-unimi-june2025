"""
Lloyd's k-means algorithm with pluggable initialization.

Paper ref:
    Lloyd's algorithm, Section 2 (informal description and cost notation).
    The key property used here is that both the E-step (assign) and the
    M-step (update) are guaranteed to not increase φ, so the algorithm
    converges in finite iterations on a finite dataset.
"""

import numpy as np
from .distance import pairwise_sq_distances, nearest_sq_distances
from .init import random_init, plusplus_init

_INIT_FN = {
    "random": random_init,
    "plusplus": plusplus_init,
}

MAX_ITER_DEFAULT = 300
TOL_DEFAULT = 1e-4


class KMeans:
    """
    Lloyd's k-means with pluggable seeding and multiple restarts.

    Parameters
    ----------
    k            : number of clusters
    init         : 'random' | 'plusplus'
    max_iter     : maximum Lloyd iterations per independent run
    tol          : convergence threshold, max per-coordinate centroid shift
    n_init       : number of independent restarts; the run with lowest inertia
                   is stored as the result
    random_state : integer seed passed to numpy.random.default_rng

    Attributes (set after fit)
    --------------------------
    centroids_ : (k, d) final cluster centers
    labels_    : (n,) cluster index for each training point
    inertia_   : float, within-cluster sum of squared distances (φ)
    n_iter_    : int, number of Lloyd iterations in the best run
    """

    def __init__(
        self,
        k: int,
        init: str = "plusplus",
        max_iter: int = MAX_ITER_DEFAULT,
        tol: float = TOL_DEFAULT,
        n_init: int = 10,
        random_state: int = 0,
    ) -> None:
        if init not in _INIT_FN:
            raise ValueError(f"init must be one of {list(_INIT_FN)}, got '{init}'")
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run n_init independent restarts on X and keep the best solution.

        The 'best' run is the one with the lowest inertia, consistent with
        the multi-start strategy discussed in Section 4 of the paper.
        """
        rng = np.random.default_rng(self.random_state)
        init_fn = _INIT_FN[self.init]

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = None

        for _ in range(self.n_init):
            centroids, labels, cost, n_iter = self._single_run(X, rng, init_fn)
            if cost < best_inertia:
                best_inertia = cost
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each point in X to its nearest centroid."""
        return self._assign(X, self.centroids_)

    def fit_with_history(self, X: np.ndarray) -> list[float]:
        """
        Single run that records inertia after every Lloyd iteration.

        Used for convergence curve plots in the notebooks.
        Returns a list of inertia values, one per iteration.
        """
        rng = np.random.default_rng(self.random_state)
        init_fn = _INIT_FN[self.init]
        centroids = init_fn(X, self.k, rng)

        history: list[float] = []
        for _ in range(self.max_iter):
            labels = self._assign(X, centroids)
            new_centroids = self._update(X, labels, rng)
            history.append(float(nearest_sq_distances(X, new_centroids).sum()))
            if self._converged(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        self.centroids_ = centroids
        self.labels_ = self._assign(X, centroids)
        self.inertia_ = history[-1] if history else np.inf
        self.n_iter_ = len(history)
        return history

    def _single_run(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
        init_fn,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """One complete fit: init → assign → update → repeat until convergence."""
        centroids = init_fn(X, self.k, rng)
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            labels = self._assign(X, centroids)
            new_centroids = self._update(X, labels, rng)
            if self._converged(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        labels = self._assign(X, centroids)
        cost = float(nearest_sq_distances(X, centroids).sum())
        return centroids, labels, cost, n_iter

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """E-step: assign each point to its nearest centroid. Returns (n,) int array."""
        return pairwise_sq_distances(X, centroids).argmin(axis=1)

    def _update(
        self, X: np.ndarray, labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        M-step: recompute each centroid as the mean of its assigned points.

        If a cluster is empty (can happen with random init on pathological data),
        reinitialise its centroid to a random point drawn from X.
        """
        d = X.shape[1]
        centroids = np.empty((self.k, d))
        for c in range(self.k):
            mask = labels == c
            if mask.any():
                centroids[c] = X[mask].mean(axis=0)
            else:
                centroids[c] = X[int(rng.integers(len(X)))]
        return centroids

    def _converged(self, old: np.ndarray, new: np.ndarray) -> bool:
        return float(np.abs(old - new).max()) < self.tol
