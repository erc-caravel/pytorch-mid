"""
Sensitivity decomposition of MID into image, mask, and cross terms.

For a joint embedding phi(x, m) = [phi_x ; phi_m] in R^{2k}, the joint
covariance has the block structure:

    Sigma = | Sigma_xx  Sigma_xm |
            | Sigma_mx  Sigma_mm |

The decomposition computes three Fréchet distances:
  MID_img   : Fréchet distance on the image subspace only
  MID_mask  : Fréchet distance on the mask subspace only
  MID_cross : total - MID_img - MID_mask (captures alignment)

Theorem (Sensitivity Decomposition):
    MID(P, Q) = MID_img + MID_mask + MID_cross

The cross term vanishes iff image and mask latents are independent, which is
not the case for well-matched image-mask pairs. A large cross term indicates
that the image-mask correspondence is disrupted (e.g., a plausible mask placed
on the wrong image, or systematic spatial misalignment).
"""

import numpy as np
from .frechet import compute_statistics, frechet_distance


def compute_mid_decomposed(
    ref_latents: np.ndarray,
    cand_latents: np.ndarray,
) -> dict[str, float]:
    """
    Decompose MID into image, mask, and cross components.

    Parameters
    ----------
    ref_latents  : (N_ref,  2k) joint embeddings; first k dims = image
    cand_latents : (N_cand, 2k) joint embeddings

    Returns
    -------
    dict with keys:
      total  : full MID on the joint space
      img    : MID on the image subspace only
      mask   : MID on the mask subspace only
      cross  : total - img - mask  (interaction term)

    Examples
    --------
    >>> import numpy as np
    >>> from pytorch_mid.decomposition import compute_mid_decomposed
    >>> rng = np.random.default_rng(42)
    >>> ref  = rng.standard_normal((40, 128)).astype(np.float32)
    >>> cand = rng.standard_normal((40, 128)).astype(np.float32)
    >>> cand[:, 64:] += 2.0   # shift mask half only
    >>> d = compute_mid_decomposed(ref, cand)
    >>> d["mask"] > d["img"]   # mask term should dominate
    True
    """
    assert ref_latents.shape[1] == cand_latents.shape[1], "Dimension mismatch."
    assert ref_latents.shape[1] % 2 == 0, "Joint dimension must be even."

    half = ref_latents.shape[1] // 2
    ref_img,  ref_mask  = ref_latents[:, :half],  ref_latents[:, half:]
    cand_img, cand_mask = cand_latents[:, :half], cand_latents[:, half:]

    mu_r_joint,  sig_r_joint  = compute_statistics(ref_latents)
    mu_c_joint,  sig_c_joint  = compute_statistics(cand_latents)
    total = frechet_distance(mu_r_joint, sig_r_joint, mu_c_joint, sig_c_joint)

    mu_r_img,   sig_r_img   = compute_statistics(ref_img)
    mu_c_img,   sig_c_img   = compute_statistics(cand_img)
    img_score = frechet_distance(mu_r_img, sig_r_img, mu_c_img, sig_c_img)

    mu_r_mask,  sig_r_mask  = compute_statistics(ref_mask)
    mu_c_mask,  sig_c_mask  = compute_statistics(cand_mask)
    mask_score = frechet_distance(mu_r_mask, sig_r_mask, mu_c_mask, sig_c_mask)

    return {
        "total": total,
        "img":   img_score,
        "mask":  mask_score,
        "cross": total - img_score - mask_score,
    }
