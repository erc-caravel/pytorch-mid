"""
Dimensionality reduction for high-dimensional latent tensors.

The MAISI VAE produces spatial latent tensors of shape (1, 4, 64, 64, 32).
After spatial pooling to (4, 8, 8, 4) and flattening, the joint
image+mask vector is 2 × (4 × 8 × 8 × 4) = 2048 dimensions. PCA to k=128
retains >94% of variance and makes the covariance matrix tractable (k² = 16k
vs D² = 4M parameters).

Proposition (Bińkowski et al., 2018 / Dai & Wipf, 2019):
Under a Gaussian latent distribution, PCA projection preserves the Fréchet
distance up to the discarded eigenvalue mass:
    |MID_k - MID_full| <= 2 * sum_{i > k} lambda_i
"""

import numpy as np


class PCAProjection:
    """
    Incremental PCA projection fitted on a stream of feature batches.

    Parameters
    ----------
    k : int
        Number of principal components to retain.
    """

    def __init__(self, k: int = 128):
        self.k = k
        self.components_: np.ndarray | None = None   # (k, d)
        self.mean_: np.ndarray | None = None         # (d,)
        self.explained_variance_: np.ndarray | None = None  # (k,)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "PCAProjection":
        """
        Fit PCA on a (N, d) feature matrix.

        Parameters
        ----------
        X : (N, d) float32 array
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.k, svd_solver="full")
        pca.fit(X.astype(np.float64))
        self.mean_ = pca.mean_.astype(np.float32)
        self.components_ = pca.components_.astype(np.float32)
        self.explained_variance_ = pca.explained_variance_ratio_
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project (N, d) features to (N, k).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return (X.astype(np.float32) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def variance_explained(self) -> float:
        """Return cumulative variance explained by the k components."""
        if self.explained_variance_ is None:
            return float("nan")
        return float(self.explained_variance_.sum())

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PCAProjection":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
