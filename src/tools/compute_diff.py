import os
import numpy as np
from tools.load_image import load_image

def compute_diff(scene, day1, day2, real_dir, pred_dir):
  real1 = os.path.join(real_dir, f"{scene}_{day1}", f"{day1}_s2.tif")
  real2 = os.path.join(real_dir, f"{scene}_{day2}", f"{day2}_s2.tif")
  pred1 = os.path.join(pred_dir, f"{scene}_{day1}_pred.tif")
  pred2 = os.path.join(pred_dir, f"{scene}_{day2}_pred.tif")

  r1 = load_image(real1, bands=[1,2,3,4,5,6,7,10,11], scale=10000)
  r2 = load_image(real2, bands=[1,2,3,4,5,6,7,10,11], scale=10000)
  p1 = load_image(pred1)
  p2 = load_image(pred2)

  real_diff = (r2 - r1)[2].flatten()  # banda 8a (7 in zero-based)
  pred_diff = (p2 - p1)[2].flatten()

  print(f"len(real_diff): {len(real_diff)}")
  #mask = np.isfinite(real_diff) & np.isfinite(pred_diff)
  #return real_diff[mask], pred_diff[mask]
  return real_diff, pred_diff
"""
# === Differenze tra due date ===
def compute_diff(scene, day1, day2, real_dir, pred_dir):
    # File reali
    real1 = os.path.join(real_dir, f"{scene}_{day1}", f"{day1}_s2.tif")
    real2 = os.path.join(real_dir, f"{scene}_{day2}", f"{day2}_s2.tif")

    # File predetti
    pred1 = os.path.join(pred_dir, f"{scene}_{day1}_pred.tif")
    pred2 = os.path.join(pred_dir, f"{scene}_{day2}_pred.tif")

    # Caricamento reali (Sentinel-2 bands selezionate)
    r1 = load_image(real1, bands=[1,2,3,4,5,6,7,10,11], scale=10000)
    r2 = load_image(real2, bands=[1,2,3,4,5,6,7,10,11], scale=10000)

    # Caricamento predetti
    p1 = load_image(pred1)
    p2 = load_image(pred2)


    # Differenze
    #KKK = 7
    real_diff = r2.flatten()#(r2 - r1)[KKK].flatten()
    pred_diff = p2.flatten() #(p2 - p1)[KKK].flatten()

    # Maschera validi
    mask = np.isfinite(real_diff) & np.isfinite(pred_diff)
    return real_diff[mask], pred_diff[mask]
"""
