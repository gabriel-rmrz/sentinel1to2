from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple, Callable


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float) -> np.ndarray:
    return num / (den + eps)

def _sanitize(
    x: np.ndarray,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
    max_abs: float | None = None,
) -> np.ndarray:
    """
    Replace NaN / Inf and optionally clamp extreme magnitudes.
    AMP-safe.
    """
    x = np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if max_abs is not None:
        x = np.clip(x, -max_abs, max_abs)
    return x



def compute_vegetation_indices(
    config: Dict[str, Any],
    s2_selected: np.ndarray,
    indices_to_compute: List[str] | None = None,
    eps: float = 1e-6,
    clip: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute vegetation indices from Sentinel-2 selected bands.

    Parameters
    ----------
    config:
      Must contain:
        - config["target"]["all_bands"]: list[str]
        - config["target"]["selected_bands"]: list[int]
      Optionally:
        - config["target"]["selected_indices"]: list[str]  (used if indices_to_compute is None and target.type=="indices")
        - config["params"]["vegetation_indices"] is NOT used anymore.

    s2_selected:
      Array of shape (C, H, W) containing only the selected S2 bands,
      in the SAME order as config["target"]["selected_bands"].

    indices_to_compute:
      List of index names to compute, e.g. ["ndvi","savi"].
      If None:
        - if target.type == "indices": uses config["target"]["selected_indices"]
        - else: computes the full supported set (ordered)

    eps:
      small number to avoid division by zero

    clip:
      If True, clips indices to typical ranges (conservative defaults).

    Returns
    -------
    indices: np.ndarray
      Array (K, H, W) with computed indices in the same order as returned names.
    names: list[str]
      Names of indices in the same order.
    """

    target_cfg = config.get("target", {})
    all_bands = target_cfg.get("all_bands", [])
    selected_bands = target_cfg.get("selected_bands", [])

    if not all_bands or not selected_bands:
        raise KeyError("config['target']['all_bands'] and config['target']['selected_bands'] must be defined.")

    # Build mapping: band_name -> channel index inside s2_selected
    # selected_bands are indices into all_bands
    selected_band_names = [all_bands[i] for i in selected_bands]
    band_pos = {name: j for j, name in enumerate(selected_band_names)}

    def B(name: str) -> np.ndarray:
        if name not in band_pos:
            raise KeyError(
                f"Band '{name}' not available in s2_selected. "
                f"Available selected bands: {selected_band_names}"
            )
        return s2_selected[band_pos[name]]

    # Required bands (common names based on your all_bands list)
    blue = B("blue")
    green = B("green")
    red = B("red")
    b5 = B("b5")
    rededge = B("rededge")
    nir = B("nir")
    swir = B("swir")

    # ------------------------------------------------------------
    # Define index functions (name -> callable)
    # ------------------------------------------------------------
    def ndvi():
        return _safe_div(nir - red, nir + red, eps)

    def gndvi():
        return _safe_div(nir - green, nir + green, eps)

    def ndre():
        return _safe_div(nir - rededge, nir + rededge, eps)

    def reci():
        return (nir / (rededge + eps)) - 1.0

    def msi():
        return swir / (nir + eps)

    def ndwi():
        return _safe_div(green - nir, green + nir, eps)

    def evi():
        return 2.5 * _safe_div(nir - red, (nir + 6.0 * red - 7.5 * blue + 1.0), eps)

    def savi():
        # L=0.5, factor=1.5
        return 1.5 * _safe_div(nir - red, nir + red + 0.5, eps)

    def arvi():
        return _safe_div(nir - (2.0 * red - blue), nir + (2.0 * red - blue), eps)

    def cig():
        return _safe_div(nir - green, green, eps)

    def cire():
        return _safe_div(nir - rededge, rededge, eps)

    def bsi():
        return _safe_div((red + swir) - (nir + blue), (red + swir) + (nir + blue), eps)

    def ndsi():
        return _safe_div(green - swir, green + swir, eps)

    def mcari():
        return (((b5 - red) - 0.2 * (b5 - green)) * b5) / (red + eps)

    INDEX_FUNCS: Dict[str, Callable[[], np.ndarray]] = {
        "ndvi": ndvi,
        "gndvi": gndvi,
        "ndre": ndre,
        "reci": reci,
        "msi": msi,
        "ndwi": ndwi,
        "evi": evi,
        "savi": savi,
        "arvi": arvi,
        "cig": cig,
        "cire": cire,
        "bsi": bsi,
        "ndsi": ndsi,
        "mcari": mcari,
    }

    # Default order (stable)
    DEFAULT_ORDER = [
        "ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi",
        "savi", "arvi", "cig", "cire", "bsi", "ndsi", "mcari",
    ]

    # Determine which indices to compute
    if indices_to_compute is None:
        if target_cfg.get("type", "").lower() == "indices":
            indices_to_compute = list(target_cfg.get("selected_indices", []))
            if not indices_to_compute:
                raise KeyError("target.type='indices' but target.selected_indices is empty.")
        else:
            indices_to_compute = DEFAULT_ORDER

    # Validate
    missing = [name for name in indices_to_compute if name not in INDEX_FUNCS]
    if missing:
        raise ValueError(f"Unknown vegetation indices requested: {missing}. Supported: {sorted(INDEX_FUNCS.keys())}")

    # Compute in requested order
    out_list = []
    out_names = []

    for name in indices_to_compute:
        arr = INDEX_FUNCS[name]().astype(np.float32, copy=False)
    
        # --- NEW: sanitize NaN / Inf early ---
        arr = _sanitize(arr, max_abs=50.0)
    
        if clip:
            # semantic / physical clipping
            if name in ("ndvi", "gndvi", "ndre", "ndwi", "arvi", "bsi", "ndsi", "savi"):
                arr = np.clip(arr, -1.0, 1.0)
            elif name in ("evi",):
                arr = np.clip(arr, -1.0, 2.0)
            elif name in ("msi", "reci", "cire", "mcari", "cig"):
                arr = np.clip(arr, -10.0, 10.0)
    
        out_list.append(arr)
        out_names.append(name)


    indices = np.stack(out_list, axis=0).astype(np.float32, copy=False)
    return indices, out_names

