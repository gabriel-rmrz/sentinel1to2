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

    # Common indices
    "ndvi": "RdYlGn",
    "gndvi": "RdYlGn",
    "ndre": "RdYlGn",
    "reci": "YlGnBu",
    "cig": "YlGnBu",
    "cire": "YlGnBu",
    "msi": "magma",
    "ndwi": "BrBG",
    "bsi": "magma",
    "ndsi": "Blues",
    "evi": "RdYlGn",
    "savi": "YlGn",
    "arvi": "RdYlGn",
    "mcari": "YlGn",
    "msavi": "YlGn",
}


def plot_histo_2d(
    output_dir: Path,
    indices: np.ndarray,   # (C, H, W)
    names,
    scene,
    prefix: str,
    p_low: float = 1.0,
    p_high: float = 99.0,
):
    """
    Plot each index/band as a 2D image with colorbar.

    Display logic:
    - For each image independently, compute low/high percentiles.
    - Values below/above those thresholds are clipped to the threshold values.
    - The color scale is set to that clipped range.

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
    p_low, p_high : float
        Percentiles used for display clipping.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = indices.shape[0]
    assert len(names) == C, "names length must match indices.shape[0]"

    for i in range(C):
        name = names[i]
        img_raw = indices[i].astype(np.float32, copy=False)

        cmap = INDEX_CMAPS.get(name.lower(), "viridis")

        valid = np.isfinite(img_raw)

        if np.any(valid):
            lo = np.nanpercentile(img_raw[valid], p_low)
            hi = np.nanpercentile(img_raw[valid], p_high)

            if np.isclose(lo, hi):
                hi = lo + 1.0

            img_disp = img_raw.copy()
            img_disp[valid] = np.clip(img_raw[valid], lo, hi)

            vmin, vmax = lo, hi
        else:
            img_disp = img_raw.copy()
            vmin, vmax = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img_disp, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{scene} – {name}", fontsize=12)
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(name, rotation=90)

        fname = output_dir / f"{prefix}_{scene}_{name}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
