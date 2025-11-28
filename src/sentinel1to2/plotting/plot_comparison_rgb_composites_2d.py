import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ..tools.make_rgb_from_names import make_rgb_from_names

def plot_comparison_rgb_composites_2d(output_dir: Path,
                                      bands1: np.ndarray,      # (C, H, W) GT
                                      bands2: np.ndarray,      # (C, H, W) INF
                                      band_names,
                                      scene,
                                      prefix: str):
    """
    Compare RGB composites (True Color, False Color, gamma-corrected):
      [ GT | INF | DIFF ] for each composite.

    DIFF is per-channel difference (INF - GT) shown with a diverging colormap
    on luminance (mean over channels).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert bands1.shape == bands2.shape
    C, H, W = bands1.shape
    assert len(band_names) == C

    composites = [
        ("true_color",
         "True Color (B4-B3-B2)",
         ("red", "green", "blue"),
         1.0),
        ("false_color_nir_rg",
         "False Color (NIR–Red–Green)",
         ("nir", "red", "green"),
         1.0),
        ("true_color_gamma2p2",
         "True Color (gamma 2.2)",
         ("red", "green", "blue"),
         2.2),
    ]

    for key, title, (r_name, g_name, b_name), gamma in composites:
        rgb1 = make_rgb_from_names(bands1, band_names, r_name, g_name, b_name, gamma=gamma)
        rgb2 = make_rgb_from_names(bands2, band_names, r_name, g_name, b_name, gamma=gamma)

        if rgb1 is None or rgb2 is None:
            continue  # missing bands, skip

        # DIFF per pixel as mean difference over channels
        diff_rgb = rgb2.astype(np.float32) - rgb1.astype(np.float32)
        # luminance-like scalar to visualize diff
        diff_lum = diff_rgb.mean(axis=-1)
        valid = np.isfinite(diff_lum)
        if np.any(valid):
            max_abs = np.nanpercentile(np.abs(diff_lum[valid]), 98)
            if max_abs == 0:
                max_abs = 1.0
        else:
            max_abs = 1.0
        vmin_diff, vmax_diff = -max_abs, max_abs

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # GT
        axes[0].imshow(rgb1)
        axes[0].set_title(f"{scene} – {title} – GT")
        axes[0].axis("off")

        # INF
        axes[1].imshow(rgb2)
        axes[1].set_title(f"{scene} – {title} – INF")
        axes[1].axis("off")

        # DIFF (luminance diff)
        im3 = axes[2].imshow(diff_lum, cmap="coolwarm",
                             vmin=vmin_diff, vmax=vmax_diff)
        axes[2].set_title(f"{scene} – {title} – DIFF (INF − GT)")
        axes[2].axis("off")

        cbar = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("INF − GT (mean over RGB)", rotation=90)

        fig.suptitle(f"{scene} – {title} (GT vs INF vs DIFF)", fontsize=14)
        plt.tight_layout()

        fname = output_dir / f"{prefix}_{scene}_{key}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
