import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


def plot_comparison_histos_2d(
    output_dir: Path,
    indices1: np.ndarray,   # (C, H, W), e.g. GT
    indices2: np.ndarray,   # (C, H, W), e.g. INF
    names,
    scene,
    prefix: str,
    p_low: float = 1.0,
    p_high: float = 99.0,
    diff_percentile: float | None = 98.0,
):
    """
    Compare two stacks of images/indices with:

        [ GT | INF | DIFF ]

    Display logic:
    - GT and INF are clipped for visualization using a COMMON percentile range
      computed from both images together.
    - The difference panel is computed from RAW values: diff = indices2 - indices1
    - The diff display can optionally use percentile-based symmetric scaling.

    Parameters
    ----------
    output_dir : Path
        Folder where PNGs will be saved.
    indices1, indices2 : np.ndarray
        Arrays of shape (C, H, W).
    names : sequence[str]
        Channel/index names, length C.
    scene : str
        Scene identifier for titles and filenames.
    prefix : str
        Prefix for output filenames.
    p_low, p_high : float
        Percentiles used to clip GT/INF for display, computed jointly.
    diff_percentile : float | None
        If not None, use symmetric scaling based on this percentile of |diff|.
        If None, use the true max absolute value of diff.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = indices1.shape[0]
    assert indices2.shape[0] == C, "indices1 and indices2 must have same number of channels"
    assert len(names) == C, "names length must match number of channels"

    for i in range(C):
        raw_name = names[i]
        name = raw_name.lower()
        cmap = INDEX_CMAPS.get(name, "viridis")

        # Raw arrays
        img1_raw = indices1[i].astype(np.float32, copy=False)
        img2_raw = indices2[i].astype(np.float32, copy=False)

        valid1 = np.isfinite(img1_raw)
        valid2 = np.isfinite(img2_raw)

        # ---------------------------------------------------------
        # COMMON percentile clipping for GT and INF display
        # ---------------------------------------------------------
        all_raw = np.concatenate([
            img1_raw[valid1].ravel(),
            img2_raw[valid2].ravel()
        ])

        if all_raw.size > 0:
            lo = np.nanpercentile(all_raw, p_low)
            hi = np.nanpercentile(all_raw, p_high)
            if np.isclose(lo, hi):
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0

        img1_disp = img1_raw.copy()
        img2_disp = img2_raw.copy()

        if np.any(valid1):
            img1_disp[valid1] = np.clip(img1_raw[valid1], lo, hi)
        if np.any(valid2):
            img2_disp[valid2] = np.clip(img2_raw[valid2], lo, hi)

        vmin_shared, vmax_shared = lo, hi

        # ---------------------------------------------------------
        # RAW difference
        # ---------------------------------------------------------
        diff = img2_raw - img1_raw
        valid_diff = np.isfinite(diff)

        if np.any(valid_diff):
            abs_diff = np.abs(diff[valid_diff])

            if diff_percentile is None:
                max_abs = np.nanmax(abs_diff)
            else:
                max_abs = np.nanpercentile(abs_diff, diff_percentile)

            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1.0
        else:
            max_abs = 1.0

        vmin_diff, vmax_diff = -max_abs, max_abs

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # GT
        im1 = axes[0].imshow(
            img1_disp,
            cmap=cmap,
            vmin=vmin_shared,
            vmax=vmax_shared
        )
        axes[0].set_title(f"{scene} – {raw_name.upper()} – GT", fontsize=12)
        axes[0].axis("off")

        # INF
        im2 = axes[1].imshow(
            img2_disp,
            cmap=cmap,
            vmin=vmin_shared,
            vmax=vmax_shared
        )
        axes[1].set_title(f"{scene} – {raw_name.upper()} – INF", fontsize=12)
        axes[1].axis("off")

        # DIFF
        im3 = axes[2].imshow(
            diff,
            cmap="coolwarm",
            vmin=vmin_diff,
            vmax=vmax_diff
        )
        axes[2].set_title(f"{scene} – {raw_name.upper()} – DIFF (INF − GT)", fontsize=12)
        axes[2].axis("off")

        # Shared colorbar for GT + INF
        fig.colorbar(
            im1,
            ax=axes[0:2],
            fraction=0.046,
            pad=0.04
        )

        # Separate colorbar for DIFF
        cbar_diff = fig.colorbar(
            im3,
            ax=axes[2],
            fraction=0.046,
            pad=0.04
        )
        cbar_diff.ax.set_ylabel("INF − GT", rotation=90)

        fig.suptitle(f"{scene} – {raw_name.upper()} (GT vs INF vs DIFF)", fontsize=14)

        fname = output_dir / f"{prefix}_{scene}_{raw_name}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
