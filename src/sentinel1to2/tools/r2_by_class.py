import numpy as np
from sklearn.metrics import r2_score
from .load_worldcover import load_worldcover
from .compute_diff import compute_diff
# === R² per classe ESA ===
def r2_by_class(scene, day1, day2, real_dir, pred_dir):
    real_diff, pred_diff = compute_diff(scene, day1, day2, real_dir, pred_dir)
    wc = load_worldcover(scene, day2, real_dir)
    if wc is None:
        return []

    wc_flat = wc.flatten()
    mask = np.isfinite(real_diff) & np.isfinite(pred_diff)
    real_diff, pred_diff, wc_flat = real_diff[mask], pred_diff[mask], wc_flat[mask]

    r2_global = r2_score(real_diff, pred_diff)
    if r2_global < -1:
        r2_global = -1
    classes = np.unique(wc_flat)

    results = []
    for cls in classes:
        mask_cls = wc_flat == cls
        if np.sum(mask_cls) < 100:
            continue
        try:
            r2_cls = r2_score(real_diff[mask_cls], pred_diff[mask_cls])
            if r2_cls < -1:
                r2_cls = -1
            results.append({
                "scene": scene,
                "day1": day1,
                "day2": day2,
                "esa_class": int(cls),
                "count": int(np.sum(mask_cls)),
                "r2": float(r2_cls)
            })
        except Exception:
            continue

    return results
