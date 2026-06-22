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

# ------------------------------------------------------------------
# Typical fixed display ranges
# Assumption for bands: reflectance in [0, 1]
# Adjust if your data uses another scaling.
# ------------------------------------------------------------------
TYPICAL_RANGES = {
    # Sentinel-2 bands
    "b1":       (0.0, 0.3),
    "blue":     (0.0, 0.3),
    "green":    (0.0, 0.4),
    "red":      (0.0, 0.4),
    "b5":       (0.0, 0.5),
    "rededge":  (0.0, 0.6),
    "b7":       (0.0, 0.6),
    "nir":      (0.0, 0.8),
    "b8a":      (0.0, 0.8),
    "b9":       (0.0, 0.3),
    "b10":      (0.0, 0.1),
    "swir":     (0.0, 0.6),
    "b12":      (0.0, 0.6),

    # Vegetation / spectral indices
    "ndvi":     (0.0, 1.0),
    "gndvi":    (0.0, 1.0),
    "ndre":     (0.0, 0.3),
    "ndwi":     (-1.0, 1.0),
    "bsi":      (-0.5, 0.5),
    "ndsi":     (-1.0, 0.0),
    "arvi":     (0.0, 1.0),
    "savi":     (0.0, 0.7),
    "msavi":    (-1.0, 1.0),

    # Indices that are not naturally bounded in [-1, 1]
    "evi":      (0.0, 1.0),
    "reci":     (0.0, 0.6),
    "cig":      (-2.0, 10.0),
    "cire":     (0.0, 0.7),
    "msi":      (0.0, 1.6),
    "mcari":    (0.0, 0.4),
}

# Optional fallback if a name is missing from TYPICAL_RANGES
DEFAULT_RANGE = (0.0, 1.0)


def plot_histo_2d(
    output_dir: Path,
    indices: np.ndarray,   # (C, H, W)
    names,
    scene,
    prefix: str,
    use_percentile_fallback: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.0,
):
    """
    Plot each index/band as a 2D image with colorbar using fixed typical
    ranges so plots are consistent across scenes.

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
    use_percentile_fallback : bool
        If True, names not found in TYPICAL_RANGES use percentile limits.
        If False, they use DEFAULT_RANGE.
    p_low, p_high : float
        Percentiles used only for fallback.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = indices.shape[0]
    assert len(names) == C, "names length must match indices.shape[0]"

    for i in range(C):
        raw_name = names[i]
        name = raw_name.lower()
        img_raw = indices[i].astype(np.float32, copy=False)

        cmap = INDEX_CMAPS.get(name, "viridis")
        valid = np.isfinite(img_raw)

        if name in TYPICAL_RANGES:
            vmin, vmax = TYPICAL_RANGES[name]
        else:
            if use_percentile_fallback and np.any(valid):
                vmin = np.nanpercentile(img_raw[valid], p_low)
                vmax = np.nanpercentile(img_raw[valid], p_high)
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1.0
            else:
                vmin, vmax = DEFAULT_RANGE

        img_disp = img_raw.copy()
        if np.any(valid):
            img_disp[valid] = np.clip(img_raw[valid], vmin, vmax)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img_disp, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{scene} – {raw_name}", fontsize=12)
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(raw_name, rotation=90)

        fname = output_dir / f"{prefix}_{scene}_{raw_name}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
