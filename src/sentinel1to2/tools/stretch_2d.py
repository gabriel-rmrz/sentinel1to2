import numpy as np
def stretch_2d(img, p_low=2, p_high=98):
    """Percentile stretch to [0,1], with optional /10000 normalization."""
    img = img.astype(np.float32)
    if np.nanmax(img) > 25:           # typical S2 scale 0–10000
        img = img / 10000.0

    valid = np.isfinite(img)
    if not np.any(valid):
        return np.zeros_like(img, dtype=np.float32)

    p2, p98 = np.percentile(img[valid], (p_low, p_high))
    img = (img - p2) / (p98 - p2 + 1e-6)
    return np.clip(img, 0, 1)
