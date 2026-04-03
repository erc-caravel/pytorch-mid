# pytorch-mid

**MID: Medical segmentation-Informed Divergence**

A Fréchet-type distributional metric for 3D medical image segmentation, analogous to [FID](https://github.com/mseitzer/pytorch-fid) for generative models but purpose-built for the segmentation setting.

MID operates in the joint latent space of a pretrained 3D medical VAE (MAISI), capturing both appearance fidelity and anatomical shape priors in a single scalar divergence.

## Definition

Given a reference distribution P (ground-truth image-mask pairs) and a candidate distribution Q (predicted image-mask pairs):

```
MID(P, Q) = ||mu_P - mu_Q||^2
            + Tr(Sigma_P + Sigma_Q - 2 * sqrtm(Sigma_P @ Sigma_Q))
```

where (mu_P, Sigma_P) and (mu_Q, Sigma_Q) are the empirical mean and covariance of joint embeddings phi(x, m) = pi([mu_theta(x); mu_theta(m)]):

- `mu_theta(.)`: MAISI VAE posterior mean encoder
- `pi`: PCA projection from the joint latent space to R^k

This equals the squared 2-Wasserstein distance W_2^2 between two Gaussian approximations (Dowson & Landau, 1982).

## Installation

```bash
pip install pytorch-mid
```

## Quick start

### From pre-extracted latent vectors (no encoder required)

```python
import numpy as np
from pytorch_mid import compute_mid_from_latents, PCAProjection

# Joint [image_latent; mask_latent] vectors, shape (N, d)
ref_latents  = np.load("reference_latents.npy")   # ground-truth pairs
cand_latents = np.load("candidate_latents.npy")   # model predictions

# Fit PCA on reference, then compute MID
proj  = PCAProjection(k=128).fit(ref_latents)
score = compute_mid_from_latents(ref_latents, cand_latents, projection=proj)
print(f"MID = {score:.4f}")
```

### Decomposed MID (image / mask / cross terms)

```python
from pytorch_mid.decomposition import compute_mid_decomposed

ref_proj  = proj.transform(ref_latents)
cand_proj = proj.transform(cand_latents)

decomp = compute_mid_decomposed(ref_proj, cand_proj)
print(f"MID_img   = {decomp['img']:.4f}")
print(f"MID_mask  = {decomp['mask']:.4f}")
print(f"MID_cross = {decomp['cross']:.4f}   (image-mask alignment)")
```

### Command-line interface

```bash
pytorch-mid latents --ref ref_latents.npy --cand cand_latents.npy --k 128
```

## Running the examples

```bash
pip install -e .
python examples/random_vectors.py
```

## Properties

| Property | Status |
|----------|--------|
| Non-negative | yes |
| Symmetric | yes |
| Zero iff distributions equal (Gaussian case) | yes |
| Triangle inequality | no (pseudometric) |
| No per-sample GT pairing required | yes |

## Citation

```bibtex
@article{mid2026,
  title   = {MedSID: Segmentation-Informed Distributional Divergence
             for Evaluating Medical Image Segmentation via Pretrained Latent Representations},
  author  = {...},
  journal = {NeurIPS},
  year    = {2026},
}
```

## License

Apache 2.0
