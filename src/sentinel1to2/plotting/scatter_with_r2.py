import os
import matplotlib.pyplot as plt
from sklearn.metrics  import r2_score
from ..tools.compute_diff import compute_diff

def scatter_with_r2(scene, day1, day2, real_dir, pred_dir):
  output_dir = "data/output_scatter"
  real_diff, pred_diff = compute_diff(scene, day1, day2, real_dir, pred_dir)
  #idx = np.random.choice(len(real_diff), 10000, replace=False)
  #real_diff, pred_diff = real_diff[idx], pred_diff[idx]

  n_pixels = len(real_diff.flatten())
  # Calcolo R²
  r2 = r2_score(real_diff, pred_diff)

  # Hexbin plot
  plt.figure(figsize=(7,6))
  hb = plt.hexbin(real_diff, pred_diff, gridsize=200, cmap="viridis", bins="log")
  plt.colorbar(hb, label="log10(N pixel)")

  # Linea y=x
  lims = [
    min(real_diff.min(), pred_diff.min()),
    max(real_diff.max(), pred_diff.max())
  ]
  plt.plot(lims, lims, 'r--', lw=2, label="y = x")

  plt.xlabel("Differenze reali (S2)")
  plt.ylabel("Differenze predette")
  plt.title(f"Scene {scene} ({day1} vs {day2})\nR² = {r2:.3f}")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()

  save_path = os.path.join(output_dir, f"{scene}_{day1}_{day2}_hexbin.png")
  plt.savefig(save_path, dpi=300)
  plt.close()

  print(f"✅ Hexbin salvato: {save_path}, R² = {r2:.3f}")

  return r2, n_pixels
