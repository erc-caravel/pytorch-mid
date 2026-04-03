"""
pytorch-mid: Medical segmentation-Informed Divergence

A Fréchet-type distributional metric for 3D medical image segmentation,
analogous to FID for generative models but purpose-built for the segmentation
setting. Operates in the joint latent space of a pretrained 3D medical VAE
(MAISI), capturing both appearance fidelity and anatomical shape priors in a
single scalar divergence.

Basic usage
-----------
    from pytorch_mid import compute_mid

    # ref_images, ref_masks  : (N, 1, H, W, D) torch tensors, float32
    # pred_images, pred_masks: (M, 1, H, W, D) torch tensors, float32
    score = compute_mid(ref_images, ref_masks, pred_images, pred_masks)

Or from pre-extracted latent vectors:

    from pytorch_mid import compute_mid_from_latents
    score = compute_mid_from_latents(ref_latents, pred_latents)
"""

from .mid import compute_mid, compute_mid_from_latents
from .frechet import frechet_distance
from .projection import PCAProjection

__version__ = "0.1.0"
__all__ = [
    "compute_mid",
    "compute_mid_from_latents",
    "frechet_distance",
    "PCAProjection",
]
