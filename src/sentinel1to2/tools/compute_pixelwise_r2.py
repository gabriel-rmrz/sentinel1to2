import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from .load_image import load_image
from .save_r2_map_as_geotiff import save_r2_map_as_geotiff


# === Calcolo R² pixelwise ===
def compute_pixelwise_r2(scene, fast=False, n_samples=1000, seed=42):
  real_days = sorted(real_scenes.get(scene, []))
  pred_days = sorted(pred_scenes.get(scene, []))
  common_days = sorted(set(real_days).intersection(pred_days))
  if len(common_days) < 5:
    print(f"⚠️ Scena {scene} ha solo {len(common_days)} acquisizioni comuni (serve ≥5)")
    return None, None

  print(f"📦 Scena {scene}: uso {len(common_days)} date comuni: {common_days}")

  # Carico stack temporali
  real_stack, pred_stack = [], []
  for day in common_days:
    real_path = os.path.join(real_dir, f"{scene}_{day}", f"{day}_s2.tif")
    pred_path = os.path.join(pred_dir, f"{scene}_{day}_pred.tif")
    real_img = load_image(real_path, bands=bands_s2, scale=scale)[band_idx]
    pred_img = load_image(pred_path)[band_idx]
    real_stack.append(real_img)
    pred_stack.append(pred_img)

  real_stack = np.stack(real_stack, axis=0)
  pred_stack = np.stack(pred_stack, axis=0)
  mask = np.all(np.isfinite(real_stack), axis=0) & np.all(np.isfinite(pred_stack), axis=0)
  H, W = mask.shape

  if fast:
    np.random.seed(seed)
    valid_idx = np.argwhere(mask)
    n_valid = len(valid_idx)
    sample_idx = valid_idx[np.random.choice(n_valid, min(n_samples, n_valid), replace=False)]
    r2_values = []
    for (i, j) in sample_idx:
      y_true = real_stack[:, i, j]
      y_pred = pred_stack[:, i, j]
      if np.std(y_true) > 0:
        r2_values.append(r2_score(y_true, y_pred))
    mean_r2 = np.nanmean(r2_values)
    print(f"⚡ FAST mode ({len(r2_values)} pixel): R² medio = {mean_r2:.3f}")
    return None, mean_r2

  r2_map = np.full((H, W), np.nan, dtype=np.float32)
  for i in range(H):
    for j in range(W):
      if mask[i, j]:
        y_true = real_stack[:, i, j]
        y_pred = pred_stack[:, i, j]
        if np.std(y_true) > 0:
          r2_map[i, j] = r2_score(y_true, y_pred)

  mean_r2 = np.nanmean(r2_map)
  print(f"✅ R² medio per scena {scene} = {mean_r2:.3f}")

  # Salvo GeoTIFF
  ref_path = os.path.join(real_dir, f"{scene}_{common_days[0]}", f"{common_days[0]}_s2.tif")
  save_r2_map_as_geotiff(scene, r2_map, ref_path)

  # Plot mappa R²
  plt.figure(figsize=(7,6))
  plt.imshow(r2_map, cmap="coolwarm", vmin=-1, vmax=1)
  plt.colorbar(label="R² pixel-wise")
  plt.title(f"Scene {scene} - R² medio = {mean_r2:.3f}")
  plt.axis("off")
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f"{scene}_r2_map.png"), dpi=200)
  plt.close()

  return r2_map, mean_r2
