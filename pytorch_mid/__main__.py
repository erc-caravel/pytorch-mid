"""
Command-line interface for pytorch-mid.

Usage
-----
Compute MID between two directories of NIfTI predictions and references:

    pytorch-mid \\
        --ref-dir   /path/to/reference_predictions \\
        --cand-dir  /path/to/candidate_predictions \\
        --projection pca_k128.pkl

Or from two .npy files of pre-extracted latents:

    pytorch-mid \\
        --ref-latents  ref_latents.npy \\
        --cand-latents cand_latents.npy \\
        --k 128
"""

import argparse
import sys
import numpy as np

from .mid import compute_mid_from_latents
from .projection import PCAProjection


def main():
    parser = argparse.ArgumentParser(
        prog="pytorch-mid",
        description="Compute MID between two latent feature sets.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- latents sub-command --------------------------------------------------
    p_lat = subparsers.add_parser(
        "latents",
        help="Compute MID from pre-extracted .npy latent files.",
    )
    p_lat.add_argument("--ref",  required=True, help=".npy file of reference joint latents (N, d)")
    p_lat.add_argument("--cand", required=True, help=".npy file of candidate joint latents (M, d)")
    p_lat.add_argument("--projection", default=None, help="Path to saved PCAProjection .pkl")
    p_lat.add_argument("--k", type=int, default=128, help="PCA components (fitted on ref if no pkl given)")

    args = parser.parse_args()

    if args.command == "latents":
        ref  = np.load(args.ref).astype(np.float32)
        cand = np.load(args.cand).astype(np.float32)

        if args.projection:
            proj = PCAProjection.load(args.projection)
        else:
            proj = PCAProjection(k=args.k).fit(ref)
            print(f"Fitted PCA: {proj.variance_explained()*100:.1f}% variance at k={args.k}")

        score = compute_mid_from_latents(ref, cand, projection=proj)
        print(f"MID = {score:.6f}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
