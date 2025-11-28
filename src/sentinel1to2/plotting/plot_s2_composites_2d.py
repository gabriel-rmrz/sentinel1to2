import numpy as np
import matplotlib.pyplot as plt
from ..tools.make_rgb_from_names import make_rgb_from_names
from pathlib import Path


def plot_s2_composites_2d(output_dir: Path,
                          bands: np.ndarray,      # (C, H, W)
                          band_names,
                          scene,
                          prefix: str):
    """
    Plot Sentinel-2 composites:
      - True Color (B4-B3-B2)  -> red, green, blue
      - False Color (NIR–Red–Green)
      - True Color with gamma correction
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) True Color
    rgb_true_gt = make_rgb_from_names(bands, band_names, "red", "green", "blue", gamma=1.0)
    if rgb_true_gt is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb_true_gt)
        ax.set_title(f"{scene} – True Color (B4-B3-B2)", fontsize=14)
        ax.axis("off")

        fname = output_dir / f"{prefix}_{scene}_true_color.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 2) False Color NIR–Red–Green
    rgb_false = make_rgb_from_names(bands, band_names, "nir", "red", "green", gamma=1.0)
    if rgb_false is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb_false)
        ax.set_title(f"{scene} – False Color (NIR–Red–Green)", fontsize=14)
        ax.axis("off")

        fname = output_dir / f"{prefix}_{scene}_false_color_nir_rg.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3) True Color with gamma correction
    rgb_true_gamma = make_rgb_from_names(bands, band_names, "red", "green", "blue", gamma=2.2)
    if rgb_true_gamma is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb_true_gamma)
        ax.set_title(f"{scene} – True Color (gamma 2.2)", fontsize=14)
        ax.axis("off")

        fname = output_dir / f"{prefix}_{scene}_true_color_gamma2p2.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
