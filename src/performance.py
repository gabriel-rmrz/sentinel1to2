import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from tools.get_pred_scenes import get_pred_scenes
from tools.get_real_scenes import get_real_scenes
from tools.r2_by_class import r2_by_class
from plotting.scatter_with_r2 import scatter_with_r2
from plotting.scatter_with_r2_by_class import scatter_with_r2_by_class


def performance():
  # === CONFIG ===
  pred_dir = "data/output_combined/"
  real_dir = "data/inference"
  output_dir = "data/output_scatter/"
  os.makedirs(output_dir, exist_ok=True)
  
  # === MAIN LOOP SU TUTTE LE SCENE ===
  pred_scenes = get_pred_scenes(pred_dir)
  real_scenes = get_real_scenes(real_dir)
  common_scenes = sorted(set(pred_scenes) & set(real_scenes))

  '''
  scene = "1"
  if scene in pred_scenes and scene in real_scenes:
    # prendo la prima coppia disponibile
    day1, day2 = pred_scenes[scene][:2]
    scatter_with_r2(scene, day1, day2,real_dir, pred_dir)
    r2_map, mean_r2 = compute_pixelwise_r2(scene, fast=False)
    print(f"\n📊 Riepilogo salvato in: {csv_path}")
  else:
    print(f"Nessuna coppia trovata per scena {scene}")  
  
  '''

  csv_path = os.path.join(output_dir, "r2_summary.csv")
  with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["scene", "day1", "day2", "r2", "n_pixels"])
  
    for scene in sorted(set(pred_scenes) & set(real_scenes)):
      for (day1, day2) in combinations(pred_scenes[scene], 2):
        if day1 in real_scenes[scene] and day2 in real_scenes[scene]:
          try:
            r2, n_pixels = scatter_with_r2(scene, day1, day2, real_dir, pred_dir)
            writer.writerow([scene, day1, day2, r2, n_pixels])
            print(f"✅ Scene {scene}, {day1} vs {day2} → R²={r2:.3f}")
          except Exception as e:
            print(f"⚠️ Errore scena {scene} {day1}-{day2}: {e}")
      days = sorted(set(pred_scenes[scene]) & set(real_scenes[scene]))
      if len(days) < 2:
        continue
      day1, day2 = days[:2]  # puoi anche fare loop su tutte le coppie
      scatter_with_r2_by_class(scene, day1, day2, real_dir, pred_dir)

  all_results = []

  for scene in tqdm(common_scenes, desc="Processing scenes"):
    days = sorted(set(pred_scenes[scene]) & set(real_scenes[scene]))
    if len(days) < 2:
      continue
    # Per semplicità: prima e seconda data
    day1, day2 = days[0], days[1]
    scene_results = r2_by_class(scene, day1, day2, real_dir, pred_dir)
    all_results.extend(scene_results)

  # === Salvataggio CSV complessivo ===
  df = pd.DataFrame(all_results)
  csv_path = os.path.join(output_dir, "r2_by_class_all_scenes.csv")
  df.to_csv(csv_path, index=False)
  print(f"📄 CSV complessivo salvato: {csv_path}")
  
  # === Plot complessivi per classe ===
  plt.figure(figsize=(12,6))
  df.boxplot(column="r2", by="esa_class", grid=False, showmeans=True)
  plt.title("Distribuzione R² per classe ESA (tutte le scene)")
  plt.suptitle("")
  plt.xlabel("ESA WorldCover class")
  plt.ylabel("R²")
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "r2_boxplot_all_scenes.png"), dpi=300)
  plt.close()
  
  # Media e deviazione standard per classe
  agg = df.groupby("esa_class")["r2"].agg(["mean", "std", "count"]).reset_index()
  plt.figure(figsize=(10,6))
  plt.bar(agg["esa_class"], agg["mean"], yerr=agg["std"], color="steelblue", alpha=0.8)
  plt.xlabel("ESA WorldCover class")
  plt.ylabel("R² medio ± std")
  plt.title("R² medio per classe ESA (tutte le scene)")
  plt.grid(True, axis="y", linestyle="--", alpha=0.6)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "r2_mean_std_all_scenes.png"), dpi=300)
  plt.close()
