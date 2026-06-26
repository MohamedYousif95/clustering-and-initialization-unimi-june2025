"""
Data loading and synthetic dataset generation.

All functions return standardized numpy arrays so the clustering
pipeline never needs to worry about scale.
"""

import numpy as np
import pandas as pd


# Real-world datasets

def load_cc_general(path: str) -> np.ndarray:
    """
    Load CC GENERAL, drop the identifier column, median-impute missing values,
    and return the standardized feature matrix.

    Missing values (MINIMUM_PAYMENTS: ~3.5%, CREDIT_LIMIT: 1 row) are filled
    with the column median, a robust choice for right-skewed financial data.

    Returns
    -------
    X : (8950, 17) float64, zero mean and unit variance per column
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["CUST_ID"])

    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return _standardize(df.to_numpy(dtype=np.float64))


def load_cc_general_with_names(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Same as load_cc_general but also returns the feature names.
    Used in notebooks that need axis labels for heatmaps.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["CUST_ID"])
    feature_names = list(df.columns)

    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = _standardize(df.to_numpy(dtype=np.float64))
    return X, feature_names


def load_mnist_subset(
    n_samples: int = 5000,
    pca_components: int = 50,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST via sklearn, subsample, normalise, and reduce with PCA.

    sklearn is used here only as a data utility (fetch_openml) and for PCA
    dimensionality reduction, not for any clustering step.

    Parameters
    ----------
    n_samples      : number of images to keep (out of 70 000)
    pca_components : number of PCA components (~85% variance at 50)
    random_state   : seed for subsampling and PCA

    Returns
    -------
    X_pca : (n_samples, pca_components), PCA-projected features
    y     : (n_samples,) int, true digit labels 0-9 (for purity checks only)
    """
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA

    X_all, y_all = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto", return_X_y=True
    )

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_all), size=n_samples, replace=False)
    X_sub = X_all[idx].astype(np.float64) / 255.0
    y_sub = y_all[idx].astype(int)

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_sub)
    return X_pca, y_sub


# Synthetic datasets

def make_blobs(
    n_samples: int = 2000,
    n_clusters: int = 5,
    n_features: int = 2,
    cluster_std: float = 0.8,
    center_box: tuple[float, float] = (-10.0, 10.0),
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate isotropic Gaussian blobs with well-separated random centers.

    Centers are drawn uniformly from center_box; each cluster is then an
    isotropic Gaussian with standard deviation cluster_std.

    Returns
    -------
    X      : (n_samples, n_features)
    labels : (n_samples,) true cluster index
    """
    rng = np.random.default_rng(random_state)
    low, high = center_box
    centers = rng.uniform(low, high, size=(n_clusters, n_features))
    return _sample_from_centers(centers, n_samples, cluster_std, rng)


def make_overlapping_blobs(
    n_samples: int = 2000,
    n_clusters: int = 5,
    n_features: int = 2,
    cluster_std: float = 2.5,
    center_box: tuple[float, float] = (-5.0, 5.0),
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian blobs with high standard deviation so clusters overlap.

    Used for the extension experiment (Section 9 of the project plan).
    """
    return make_blobs(n_samples, n_clusters, n_features, cluster_std, center_box, random_state)


# Private helpers

def _sample_from_centers(
    centers: np.ndarray,
    n_samples: int,
    std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_samples points around the given centers and shuffle."""
    n_clusters, n_features = centers.shape
    base = n_samples // n_clusters
    remainder = n_samples - base * n_clusters

    X_parts, label_parts = [], []
    for c in range(n_clusters):
        n = base + (1 if c < remainder else 0)
        X_parts.append(rng.normal(centers[c], std, size=(n, n_features)))
        label_parts.append(np.full(n, c, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    perm = rng.permutation(len(X))
    return X[perm], labels[perm]


def _standardize(X: np.ndarray) -> np.ndarray:
    """Zero mean and unit variance per column. Constant columns get std = 1."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std
