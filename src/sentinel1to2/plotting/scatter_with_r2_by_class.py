import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from ..tools.compute_diff import compute_diff
from ..tools.load_worldcover import load_worldcover

# === Scatter globale + R² per classe ===
def scatter_with_r2_by_class(scene, day1, day2, real_dir, pred_dir):
  output_dir = "data/output_scatter_by_class"
  real_diff, pred_diff = compute_diff(scene, day1, day2, real_dir, pred_dir)
  wc = load_worldcover(scene, day2, real_dir)  # worldcover allineato all’immagine “più recente”
  if wc is None:
    return

  wc_flat = wc.flatten()
  mask = np.isfinite(real_diff) & np.isfinite(pred_diff)
  print(mask)
  print(f"len(wc_flat): {len(wc_flat)}")
  real_diff, pred_diff, wc_flat = real_diff[mask], pred_diff[mask], wc_flat[mask]

  n_pixels = len(real_diff)
  r2_global = r2_score(real_diff, pred_diff)
  if r2_global < -1:
    r2_global = -1

  # Hexbin globale
  plt.figure(figsize=(7,6))
  hb = plt.hexbin(real_diff, pred_diff, gridsize=200, cmap="viridis", bins="log")
  plt.colorbar(hb, label="log10(N pixel)")
  lims = [min(real_diff.min(), pred_diff.min()), max(real_diff.max(), pred_diff.max())]
  plt.plot(lims, lims, 'r--', lw=2, label="y = x")
  plt.xlabel("Differenze reali (S2)")
  plt.ylabel("Differenze predette")
  plt.title(f"Scene {scene} ({day1}-{day2}) | R² globale = {r2_global:.3f}")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f"{scene}_{day1}_{day2}_hexbin.png"), dpi=300)
  plt.close()

  # R² per classe ESA
  classes = np.unique(wc_flat)
  stats = []
  for cls in classes:
    mask_cls = wc_flat == cls
    if np.sum(mask_cls) < 100:  # ignora classi rare
      continue
    try:
      r2_cls = r2_score(real_diff[mask_cls], pred_diff[mask_cls])
      if r2_cls < -1:
        r2_cls = -1
      stats.append((int(cls), np.sum(mask_cls), r2_cls))
    except Exception:
      continue

  # CSV per classe
  csv_path = os.path.join(output_dir, f"r2_by_class_{scene}_{day1}_{day2}.csv")
  with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ESA_Class", "Count", "R2"])
    writer.writerows(stats)
  print(f"📄 CSV per classe salvato: {csv_path}")

  # Barplot R² per classe
  classes, counts, r2_vals = zip(*stats)
  plt.figure(figsize=(10,6))
  plt.bar(classes, r2_vals, color="steelblue")
  plt.xlabel("ESA WorldCover class")
  plt.ylabel("R²")
  plt.title(f"R² per classe ESA — Scene {scene} ({day1}-{day2})")
  plt.grid(True, axis="y", linestyle="--", alpha=0.6)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f"r2_by_class_{scene}_{day1}_{day2}.png"), dpi=300)
  plt.close()

  print(f"✅ Hexbin e barplot completati per {scene} ({day1}-{day2}) — R² globale = {r2_global:.3f}")
  return r2_global, stats

