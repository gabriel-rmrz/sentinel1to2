import numpy as np
from .stretch_2d import stretch_2d
def make_rgb_from_names(patch, band_names, r_name, g_name, b_name, gamma=1.0):
    """
    patch: (C, H, W)
    band_names: list of length C
    r_name/g_name/b_name: names (e.g., 'red','green','blue','nir')
    returns: (H, W, 3) or None if any band is missing.
    """
    band_names_lower = [b.lower() for b in band_names]
    try:
        ir = band_names_lower.index(r_name.lower())
        ig = band_names_lower.index(g_name.lower())
        ib = band_names_lower.index(b_name.lower())
    except ValueError:
        return None

    r = stretch_2d(patch[ir])
    g = stretch_2d(patch[ig])
    b = stretch_2d(patch[ib])

    rgb = np.stack([r, g, b], axis=-1)
    if gamma != 1.0:
        rgb = np.power(rgb, 1.0 / gamma)
        rgb = np.clip(rgb, 0, 1)

    return rgb
