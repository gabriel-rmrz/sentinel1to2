import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..tools.stretch_2d import stretch_2d

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

def plot_histo_2d(output_dir: Path,
                  indices: np.ndarray,   # (C, H, W)
                  names,
                  scene,
                  prefix: str):
    """
    Plot each index/band as a 2D image with colorbar.

    Parameters
    ----------
    output_dir : Path
        Directory where PNGs are saved.
    indices : np.ndarray
        Array of shape (C, H, W).
    names : list[str]
        Names of each channel, length C.
    scene : str
        Scene identifier (used in title and filename).
    prefix : str
        Prefix for output file names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = indices.shape[0]
    assert len(names) == C, "names length must match indices.shape[0]"

    for i in range(C):
        name = names[i]
        img = stretch_2d(indices[i])

        cmap = INDEX_CMAPS.get(name.lower(), "viridis")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(f"{scene} – {name}", fontsize=12)
        ax.axis("off")

        # simple side colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(name, rotation=90)

        fname = output_dir / f"{prefix}_{scene}_{name}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
