"""
Example: computing MID from random latent vectors (no encoder required).

This demonstrates the API when you already have pre-extracted features,
e.g., from a pretrained MAISI encoder run offline.

Scenario
--------
Three distributions are defined:
  - ref   : reference (GT) distribution — N(0, I)
  - close : candidate close to reference — N(0.2, I)
  - far   : candidate far from reference — N(3.0, I)

We verify that MID(ref, close) < MID(ref, far).
"""

import numpy as np
from pytorch_mid import compute_mid_from_latents, PCAProjection
from pytorch_mid.decomposition import compute_mid_decomposed

RNG = np.random.default_rng(42)

# Simulate joint [image_latent; mask_latent] vectors in R^{2048}
D = 2048   # 2 x (4 x 8 x 8 x 4) after spatial pooling
N = 200    # samples per set (must be > k=128 for PCA to be well-defined)

ref   = RNG.standard_normal((N, D)).astype(np.float32)
close = RNG.standard_normal((N, D)).astype(np.float32) + 0.2
far   = RNG.standard_normal((N, D)).astype(np.float32) + 3.0

# Fit PCA on the reference set (k=128 components)
proj = PCAProjection(k=128).fit(ref)
print(f"PCA: {proj.variance_explained()*100:.1f}% variance explained at k=128")

# --- Scalar MID scores -------------------------------------------------------
mid_close = compute_mid_from_latents(ref, close, projection=proj)
mid_far   = compute_mid_from_latents(ref, far,   projection=proj)
mid_self  = compute_mid_from_latents(ref, ref,   projection=proj)

print(f"\nMID (ref vs self)  = {mid_self:.4f}   (should be ≈0)")
print(f"MID (ref vs close) = {mid_close:.4f}")
print(f"MID (ref vs far)   = {mid_far:.4f}   (should be >> MID_close)")
assert mid_close < mid_far, "MID should increase with distribution shift"
assert mid_self < 1e-3,     "MID(P, P) should be ≈0"
print("Ordering check: MID_self < MID_close < MID_far  ✓")

# --- Decomposed MID: shift mask half only ------------------------------------
print("\n--- Decomposed MID ---")
mask_only_shift = ref.copy()             # start from reference (same image half)
mask_only_shift[:, D // 2:] += 3.0     # shift only the mask half

ref_proj  = proj.transform(ref)
cand_proj = proj.transform(mask_only_shift)

decomp = compute_mid_decomposed(ref_proj, cand_proj)
print(f"total : {decomp['total']:.4f}")
print(f"img   : {decomp['img']:.4f}   (image half unshifted — should be small)")
print(f"mask  : {decomp['mask']:.4f}  (mask half shifted — should be large)")
print(f"cross : {decomp['cross']:.4f}")
assert decomp["mask"] > decomp["img"], "Mask term should dominate when only mask is shifted"
print("Decomposition check: mask >> img  ✓")

# --- Symmetry check ----------------------------------------------------------
print("\n--- Symmetry check ---")
mid_AB = compute_mid_from_latents(ref, close, projection=proj)
mid_BA = compute_mid_from_latents(close, ref, projection=proj)
print(f"MID(A, B) = {mid_AB:.6f}")
print(f"MID(B, A) = {mid_BA:.6f}")
assert abs(mid_AB - mid_BA) < 1e-4, "MID should be symmetric"
print("Symmetry check: MID(A, B) ≈ MID(B, A)  ✓")

# --- Monotonicity with shift magnitude --------------------------------------
print("\n--- Monotonicity with shift magnitude ---")
shifts = [0.0, 0.5, 1.0, 2.0, 4.0]
prev_score = -1.0
for shift in shifts:
    cand = RNG.standard_normal((N, D)).astype(np.float32) + shift
    score = compute_mid_from_latents(ref, cand, projection=proj)
    check = "✓" if score > prev_score else "✗"
    print(f"  shift={shift:.1f}  MID={score:.4f}  {check}")
    prev_score = score
print("Monotonicity check passed  ✓")
