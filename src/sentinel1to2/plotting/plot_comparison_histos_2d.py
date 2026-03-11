import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..tools.stretch_2d import stretch_2d

# Optional: map band/index names to nice colormaps
INDEX_CMAPS = {
    # Sentinel-2 bands
    "b1": "Greys",
    "blue": "Blues",
    "green": "Greens",
    "red": "Reds",
    "b5": "YlOrBr",
    "rededge": "YlOrRd",
    "b7": "YlGnBu",
    "nir": "Greys",
    "b8a": "Greys",
    "b9": "Greys",
    "b10": "PuBuGn",
    "swir": "cubehelix",
    "b12": "cubehelix",

    # Common indices (adapt as you wish)
    "ndvi":  "RdYlGn",
    "gndvi": "RdYlGn",
    "ndre":  "RdYlGn",

    "reci":  "YlGnBu",
    "cig":   "YlGnBu",
    "cire":  "YlGnBu",

    "msi":   "magma",
    "ndwi":  "BrBG",
    "bsi":   "magma",
    "ndsi":  "Blues",

    "evi":   "RdYlGn",
    "savi":  "YlGn",
    "arvi":  "RdYlGn",

    "mcari": "YlGn",
    "msavi": "YlGn",
}

def plot_comparison_histos_2d(output_dir: Path,
                              indices1: np.ndarray,   # (C, H, W) e.g. GT
                              indices2: np.ndarray,   # (C, H, W) e.g. INF
                              names,
                              scene,
                              prefix: str):
    """
    Compare two stacks of indices (GT vs INF) with a third panel showing
    the difference (INF - GT):

        [ GT | INF | DIFF ]

    Styling is consistent with plot_histo_2d:
    - same stretch_2d() helper
    - same INDEX_CMAPS for colormaps
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = indices1.shape[0]
    assert indices2.shape[0] == C, "indices1 and indices2 must have same C"
    assert len(names) == C, "names length must match number of channels"

    for i in range(C):
        raw_name = names[i]
        name = raw_name.lower()

        # Stretch both like in single-plot version
        #img1 = stretch_2d(indices1[i])  # GT
        #img2 = stretch_2d(indices2[i])  # INF
        img1 = indices1[i]  # GT
        img2 = indices2[i]  # INF
        cmap = INDEX_CMAPS.get(name, "viridis")

        # Difference in original value space
        diff = indices2[i].astype(np.float32) - indices1[i].astype(np.float32)
        valid = np.isfinite(diff)
        if np.any(valid):
            max_abs = np.nanpercentile(np.abs(diff[valid]), 98)
            if max_abs == 0:
                max_abs = 1.0
        else:
            max_abs = 1.0
        vmin_diff, vmax_diff = -max_abs, max_abs

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # 1) GT
        im1 = axes[0].imshow(img1, cmap=cmap)
        axes[0].set_title(f"{scene} – {raw_name.upper()} – GT", fontsize=12)
        axes[0].axis("off")

        # 2) INF
        im2 = axes[1].imshow(img2, cmap=cmap)
        axes[1].set_title(f"{scene} – {raw_name.upper()} – INF", fontsize=12)
        axes[1].axis("off")

        # 3) DIFF = INF - GT
        im3 = axes[2].imshow(diff, cmap="coolwarm",
                             vmin=vmin_diff, vmax=vmax_diff)
        axes[2].set_title(f"{scene} – {raw_name.upper()} – DIFF (INF − GT)",
                          fontsize=12)
        axes[2].axis("off")

        # Shared colorbar for GT + INF (stretched 0–1)
        fig.colorbar(im2,
                     ax=axes[0:2].ravel().tolist(),
                     fraction=0.046,
                     pad=0.04)

        # Separate colorbar for DIFF
        cbar_diff = fig.colorbar(im3,
                                 ax=axes[2],
                                 fraction=0.046,
                                 pad=0.04)
        cbar_diff.ax.set_ylabel("INF − GT", rotation=90)

        fig.suptitle(f"{scene} – {raw_name.upper()} (GT vs INF vs DIFF)",
                     fontsize=14)
        #plt.tight_layout()

        fname = output_dir / f"{prefix}_{scene}_{raw_name}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

