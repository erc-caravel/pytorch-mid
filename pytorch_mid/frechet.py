"""
Fréchet distance between two multivariate Gaussians.

This is the core computation underlying both FID and MID:

    d_F^2(N(mu_1, Sigma_1), N(mu_2, Sigma_2))
        = ||mu_1 - mu_2||^2
          + Tr(Sigma_1 + Sigma_2 - 2 * sqrtm(Sigma_1 @ Sigma_2))

This equals the squared 2-Wasserstein distance W_2^2 between the two Gaussians
(Dowson & Landau, 1982).
"""

import numpy as np
from scipy import linalg


def compute_statistics(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical mean and covariance of a feature matrix.

    Parameters
    ----------
    features : (N, k) float32 array

    Returns
    -------
    mu    : (k,) mean vector
    sigma : (k, k) covariance matrix
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute the squared Fréchet distance between two Gaussians.

    Parameters
    ----------
    mu1, mu2       : (k,) mean vectors
    sigma1, sigma2 : (k, k) covariance matrices
    eps            : small offset added to diagonal for numerical stability

    Returns
    -------
    Squared Fréchet distance (non-negative scalar).

    References
    ----------
    Dowson & Landau (1982). "The Fréchet distance between multivariate normal
    distributions." Journal of Multivariate Analysis, 12(3), 450–455.

    Heusel et al. (2017). "GANs trained by a two time-scale update rule converge
    to a local Nash equilibrium." NeurIPS 2017. (FID paper)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product of covariances: sigma1 @ sigma2
    # Add eps * I for numerical stability before matrix square root
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2 + offset, disp=False)

    # sqrtm can produce tiny imaginary parts due to numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            raise ValueError(
                f"Matrix square root has significant imaginary component: "
                f"{np.max(np.abs(np.imag(covmean))):.4e}"
            )
        covmean = np.real(covmean)

    tr_covmean = np.trace(covmean)
    distance = (
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * tr_covmean
    )
    return float(max(distance, 0.0))  # clip numerical negatives
