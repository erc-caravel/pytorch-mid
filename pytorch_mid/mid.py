"""
Core MID computation.

Definition (Medical segmentation-Informed Divergence)
------------------------------------------------------
Given reference distribution P (GT image-mask pairs) and candidate
distribution Q (predicted image-mask pairs):

    MID(P, Q) = ||mu_P - mu_Q||^2
                + Tr(Sigma_P + Sigma_Q - 2 * sqrtm(Sigma_P @ Sigma_Q))

where (mu_P, Sigma_P) and (mu_Q, Sigma_Q) are the empirical mean and
covariance of joint embeddings phi(x, m) = pi([mu_theta(x); mu_theta(m)]).

- mu_theta(.) : MAISI VAE posterior mean encoder (deterministic embedding)
- pi          : PCA projection from the flattened joint latent space to R^k
"""

from __future__ import annotations

import numpy as np

from .frechet import compute_statistics, frechet_distance
from .projection import PCAProjection


def compute_mid_from_latents(
    ref_latents: np.ndarray,
    cand_latents: np.ndarray,
    projection: PCAProjection | None = None,
) -> float:
    """
    Compute MID from pre-extracted joint latent vectors.

    Parameters
    ----------
    ref_latents  : (N_ref,  d) float32 — joint [img; mask] latent vectors
    cand_latents : (N_cand, d) float32 — joint [img; mask] latent vectors
    projection   : fitted PCAProjection, or None to use raw features

    Returns
    -------
    MID score (lower = more similar distributions).

    Examples
    --------
    >>> import numpy as np
    >>> from pytorch_mid import compute_mid_from_latents, PCAProjection
    >>> rng = np.random.default_rng(0)
    >>> ref  = rng.standard_normal((50, 256)).astype(np.float32)
    >>> cand = rng.standard_normal((50, 256)).astype(np.float32) + 0.5
    >>> proj = PCAProjection(k=64).fit(ref)
    >>> score = compute_mid_from_latents(ref, cand, projection=proj)
    >>> score > 0
    True
    """
    if ref_latents.ndim != 2 or cand_latents.ndim != 2:
        raise ValueError(
            f"Expected 2-D arrays; got shapes {ref_latents.shape} and {cand_latents.shape}."
        )
    if ref_latents.shape[1] != cand_latents.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {ref_latents.shape[1]} vs {cand_latents.shape[1]}."
        )
    if len(ref_latents) < 2 or len(cand_latents) < 2:
        raise ValueError("Need at least 2 samples in each set to estimate covariance.")

    if projection is not None:
        ref_latents  = projection.transform(ref_latents)
        cand_latents = projection.transform(cand_latents)

    mu_ref,  sigma_ref  = compute_statistics(ref_latents)
    mu_cand, sigma_cand = compute_statistics(cand_latents)
    return frechet_distance(mu_ref, sigma_ref, mu_cand, sigma_cand)


def compute_mid(
    ref_images:   "torch.Tensor",
    ref_masks:    "torch.Tensor",
    cand_images:  "torch.Tensor",
    cand_masks:   "torch.Tensor",
    encoder=None,
    projection: PCAProjection | None = None,
    pool: tuple[int, int, int] = (8, 8, 4),
    device: str = "cuda",
) -> float:
    """
    Compute MID from raw 3D image and mask volumes.

    Requires the MAISI encoder from the `panqc` package (not included in
    this package due to model weight licensing). See the README for
    instructions on obtaining the encoder.

    Parameters
    ----------
    ref_images   : (N, 1, H, W, D) float32 tensor — reference CT/MRI volumes
    ref_masks    : (N, C, H, W, D) float32 tensor — reference segmentation masks
    cand_images  : (M, 1, H, W, D) float32 tensor — candidate volumes
    cand_masks   : (M, C, H, W, D) float32 tensor — candidate segmentation masks
    encoder      : MAISI VAE encoder instance. If None, raises an error with
                   instructions.
    projection   : fitted PCAProjection, or None to fit on the reference set.
    pool         : spatial average-pooling target (h, w, d) applied to latents
                   before concatenation. Default (8, 8, 4) gives 1024 dims
                   per modality.
    device       : torch device string.

    Returns
    -------
    MID score (float, non-negative).
    """
    if encoder is None:
        raise ValueError(
            "An encoder instance is required. "
            "See the README for instructions on loading the MAISI encoder."
        )

    ref_vecs  = _encode_pairs(ref_images,  ref_masks,  encoder, pool, device)
    cand_vecs = _encode_pairs(cand_images, cand_masks, encoder, pool, device)

    if projection is None:
        projection = PCAProjection(k=128).fit(ref_vecs)

    return compute_mid_from_latents(ref_vecs, cand_vecs, projection)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pool_spatial(latent: np.ndarray, pool: tuple[int, int, int]) -> np.ndarray:
    """
    Average-pool a spatial latent tensor (C, H, W, D) to (C, ph, pw, pd)
    and flatten to (C * ph * pw * pd,).
    """
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(latent).unsqueeze(0)  # (1, C, H, W, D)
    t = F.adaptive_avg_pool3d(t, pool)          # (1, C, ph, pw, pd)
    return t.squeeze(0).numpy().ravel()


def _encode_pairs(
    images: "torch.Tensor",
    masks:  "torch.Tensor",
    encoder,
    pool: tuple[int, int, int],
    device: str,
) -> np.ndarray:
    """Encode a batch of (image, mask) pairs to joint latent vectors."""
    import torch
    vecs = []
    for img, msk in zip(images, masks):
        img_b = img.unsqueeze(0).to(device)
        msk_b = msk.unsqueeze(0).to(device)
        with torch.no_grad():
            img_lat = encoder.encode_mean(img_b).cpu().numpy()[0]  # (C,H,W,D)
            msk_lat = encoder.encode_mean(msk_b).cpu().numpy()[0]
        img_vec = _pool_spatial(img_lat, pool)
        msk_vec = _pool_spatial(msk_lat, pool)
        vecs.append(np.concatenate([img_vec, msk_vec]))
    return np.stack(vecs).astype(np.float32)
