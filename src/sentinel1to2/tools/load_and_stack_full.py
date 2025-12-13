from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio

from .compute_vegetation_indices import compute_vegetation_indices


def _read_single_band(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        profile = src.profile
    return arr[np.newaxis, ...], profile


def _read_multiband(path: Path, bands: list[int]) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(bands).astype(np.float32)
    return arr


def load_and_stack_full(
    config: dict,
    folder: str,
    data_dir: Path,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
):
    """
    Load and normalize all inputs and targets for a full scene.

    Returns
    -------
    dsm : (1, H, W)
    s1  : (C1, H, W)
    worldcover : (1, H, W)
    s2_selected : (C2, H, W)
    indices : (Nidx, H, W)
    ind_names : list[str]
    profile : rasterio profile (from DSM)
    """

    scene_dir = Path(data_dir) / folder

    # Infer base name robustly (everything before _dsm/_s1/_s2)
    # Example: 1001_2192_dsm.tif → 1001_2192
    def _stem(name: str) -> str:
        return name.replace("_dsm.tif", "").replace("_s1.tif", "").replace("_s2.tif", "").replace("_worldcover.tif", "")

    # Build paths safely
    dsm_path = next(scene_dir.glob("*_dsm.tif"))
    base_name = _stem(dsm_path.name)

    paths = {
        "dsm": scene_dir / f"{base_name}_dsm.tif",
        "s1": scene_dir / f"{base_name}_s1.tif",
        "s2": scene_dir / f"{base_name}_s2.tif",
        "worldcover": scene_dir / f"{base_name}_worldcover.tif",
    }

    # ------------------------------------------------------------
    # DSM
    # ------------------------------------------------------------
    dsm, profile = _read_single_band(paths["dsm"], 1)
    if mean is not None and std is not None:
        dsm = (dsm - mean[0, None, None]) / std[0, None, None]

    # ------------------------------------------------------------
    # Sentinel-1 (VV/VH assumed bands 3,4 — configurable later)
    # ------------------------------------------------------------
    s1 = _read_multiband(paths["s1"], bands=[3, 4])
    s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
    if mean is not None and std is not None:
        s1 = (s1 - mean[1:3, None, None]) / std[1:3, None, None]

    # ------------------------------------------------------------
    # WorldCover
    # ------------------------------------------------------------
    worldcover, _ = _read_single_band(paths["worldcover"], 1)
    if mean is not None and std is not None:
        worldcover = (worldcover - mean[3, None, None]) / std[3, None, None]

    # ------------------------------------------------------------
    # Sentinel-2 (all bands, scaled)
    # ------------------------------------------------------------
    with rasterio.open(paths["s2"]) as src:
        s2 = src.read().astype(np.float32)

    s2 = s2 / 10000.0

    # Select bands according to config
    selected_bands = config["target"]["selected_bands"]
    s2_selected = s2[selected_bands, :, :]

    # ------------------------------------------------------------
    # Vegetation indices (computed once, selection later)
    # ------------------------------------------------------------
    if config["target"]["type"] == "indices":
        indices, ind_names = compute_vegetation_indices(config, s2_selected, indices_to_compute=config["target"]["selected_indices"])
    else:
        indices, ind_names = compute_vegetation_indices(config, s2_selected)


    return dsm, s1, worldcover, s2_selected, indices, ind_names, profile

