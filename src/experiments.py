"""
Multi-trial experiment runner.

Runs k-means with n_init=1 for n_trials independent trials so each trial
reflects a single seeding attempt, isolating the effect of initialization
rather than the effect of taking the best of multiple restarts.
"""

import numpy as np
from .kmeans import KMeans


def run_trials(
    X: np.ndarray,
    k: int,
    init: str,
    n_trials: int = 50,
    base_seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Fit KMeans n_trials times, each with a distinct seed and n_init=1.

    Parameters
    ----------
    X         : feature matrix
    k         : number of clusters
    init      : 'random' | 'plusplus'
    n_trials  : number of independent runs
    base_seed : seeds used are base_seed, base_seed+1, …, base_seed+n_trials-1

    Returns
    -------
    dict with keys:
        'inertia' : (n_trials,) float, final φ for each trial
        'n_iter'  : (n_trials,) int, Lloyd iterations used
    """
    inertias = np.empty(n_trials)
    iters = np.empty(n_trials, dtype=int)

    for t in range(n_trials):
        model = KMeans(k=k, init=init, n_init=1, random_state=base_seed + t)
        model.fit(X)
        inertias[t] = model.inertia_
        iters[t] = model.n_iter_

    return {"inertia": inertias, "n_iter": iters}
