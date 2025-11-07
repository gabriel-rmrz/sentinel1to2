import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity  
from sklearn.metrics import r2_score
from itertools import combinations
from tqdm import tqdm
from .tools.load_image import load_image
from .tools.load_predicted_ndvi import load_predicted_ndvi
from .tools.get_pred_scenes import get_pred_scenes
from .tools.get_real_scenes import get_real_scenes
from .tools.r2_by_class import r2_by_class
from .plotting.scatter_with_r2 import scatter_with_r2
from .plotting.scatter_with_r2_by_class import scatter_with_r2_by_class

def compute_vegetation_indices(s2):
  """
  inputs:
    s2: We are considering only the bands relevants for our pipeline. If we have all sentinel bands, we are suposed to filter
        leaving only the 'selected' band defined below:
          band_names = ["b1","blue", "green", "red", "b5", "rededge", "b7", "nir","b8a","b9", "b10", "swir", "b12"]
          selected_bands = [1,2,3,4,5,6,7,10,11]
          
  """

  # Bande Sentinel-2 standardizzate
  blue   = s2[0]  # B2
  green  = s2[1]  # B3
  red    = s2[2]  # B4
  b5 = s2[3]  # B5
  rededge = s2[4] # B6
  nir    = s2[6]  # B8
  swir   = s2[8] # B12

  eps = 1e-6
  # === Indici Spettrali ===
  ndvi = (nir - red) / (nir + red + eps)
  gndvi = (nir - green) / (nir + green + eps)
  ndre = (nir - rededge) / (nir + rededge + eps)
  reci = (nir / (rededge + eps)) - 1
  msi = swir / (nir + eps)
  ndwi = (green - nir) / (green + nir + eps)
  evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
  savi = ((nir - red) / (nir + red + 0.5)) * (1.5)
  arvi = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + eps)
  cire = nir / (rededge + eps)
  bsi = ((red + swir) - (nir + blue)) / ((red + swir) + (nir + blue) + eps)
  ndsi = (green - swir) / (green + swir + eps)
  mcari = ((b5 - red) - 0.2*(b5 - green)) * b5 / (red + eps)
  ind_names = ["ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi", "savi", "arvi", "cire", "bsi", "ndsi", "mcari"]

  #s2_selected = s2[ np.r_[1,2,3,4,5,6,7,10,11] ]/10000
  #print(np.clip(mcari, 0, 10).shape)
  indices = np.stack([
      np.clip(ndvi, -1, 1), #np.clip(ndvi, -1, 1), #In teoria mi interessano i soli valori tra 0 e 1
      np.clip(gndvi, -1, 1),
      np.clip(ndre, -1, 1),
      np.clip(reci, -1, 10),
      np.clip(msi, 0, 10),
      np.clip(ndwi, -1, 1),
      np.clip(evi, 0, 2),
      np.clip(savi, -1, 1),
      np.clip(arvi, -1, 1),
      np.clip(cire, 0, 10),
      np.clip(bsi, -1, 1),
      np.clip(ndsi, -1, 1),
      np.clip(mcari, 0, 10)
  ], axis=0).astype(np.float32)

  return indices, ind_names 

def compute_metrics(img_gt, img_inf):
  mae = np.abs(img_inf - img_gt).mean()
  psnr = peak_signal_noise_ratio(img_gt, img_inf, data_range=1.0)
  ssim = structural_similarity(img_gt, img_inf, data_range=1.0)
  r2 = r2_score(img_gt, img_inf)
  #print(f"mae: {mae}, psnr: {psnr}, ssim: {ssim}, r2: {r2}")
  metrics =  [mae, psnr, ssim, r2]
  return metrics

def compute_all_metrics(df, scene_name, g_truth, inference, names):
   for i in range(g_truth.shape[0]):
     metrics = compute_metrics(g_truth[i,:,:], inference[i,:,:])
     df.loc[-1] = [scene_name, names[i]]+metrics
     df.index = df.index+1
     df = df.sort_index()
   return df

def plot_histos_from_df(df, metric_names, prefix):
  if "veg_index" in df.keys():
    hist_vars = df["veg_index"].unique()  
    var_type = "veg_index"
  elif "band" in df.keys():
    hist_vars = df["band"].unique()
    var_type = "band"

  for hv in hist_vars:
    for mn in metric_names:
      fig, ax = plt.subplots()
      df[df[var_type] ==hv].hist(mn, ax=ax)
      fig.savefig(f"plots/metrics/histos/{prefix}_{hv}_{mn}.png")
      plt.close(fig)

def plot_histo_2d(indices, names, scene, prefix):
  for i in range(indices.shape[0]):
    fig, ax = plt.subplots()
    ax.imshow(indices[i])
    fig.savefig(f"plots/indices/histos2d/{prefix}_{scene}_{names[i]}.png")
    plt.close(fig)

def plot_scatter_gt_vs_inf(indices_gt, indices_inf, names, scene, prefix):
  # TODO: Add x=y line
  # TODO: Add to the config file the indices selected and the their limits
  for i in range(indices_gt.shape[0]):
    fig, ax = plt.subplots()
    ax.scatter(indices_gt[i].flatten(), indices_inf[i].flatten(), s=1, alpha=0.3)
    fig.savefig(f"plots/indices/scatter_gt_vs_inf/{prefix}_{scene}_{names[i]}.png")
    plt.close(fig)

def plot_abs_error(indices_gt, indices_inf, names, scene, prefix):
  # TODO: Add x=y line
  # TODO: Add to the config file the indices selected and the their limits
  for i in range(indices_gt.shape[0]):
    fig, ax = plt.subplots()
    plt.hist(np.abs(indices_gt[i].flatten()-indices_inf[i].flatten()))
    fig.savefig(f"plots/indices/histo_abs_error/{prefix}_{scene}_{names[i]}.png")
    plt.close(fig)
    

def performance(real_dir, pred_dir):
  """
  Inputs:
    s2 bands grand truth s2b(GT)
    s2 bands inferred s2b(I)
    NDVI inferred NDVI(I)
  compute_indices() method, using :
    s2b(GT)
    s2b(I)
  compute_metrics() method, from:
    s2b (GT) vs s2b (I)
    NDVI (GT) vs NDVI (I)
    indices (GT) vs indices (I)
  """
  # === CONFIG ===
  #pred_dir = "data/output_combined/"
  #pred_ndvi_dir = "data/output"
  #real_dir = "data/test"

  pred_dir = "data/output_combined_bkup/"
  pred_ndvi_dir = "data/output_bkup"
  real_dir = "data/test_bkup"

  output_dir = "data/output_performance/"
  os.makedirs(output_dir, exist_ok=True)
  
  #TODO: Give the option to save the indice so they don't need to be recactulated every time. 
  # Although, it might take the same time to only produce the plots.

  band_names = ["b1","blue", "green", "red", "b5", "rededge", "b7", "nir","b8a","b9", "b10", "swir", "b12"]
  selected_bands = [1,2,3,4,5,6,7,10,11]
  band_names = [band_names[j] for j in selected_bands]
  metric_names =  ["mae", "psnr", "ssim", "r2"]
  gt_vs_inf_df = pd.DataFrame(columns = ['scene','band']+metric_names)
  gt_vs_comp_df = pd.DataFrame(columns = ['scene','veg_index']+metric_names)
  gt_vs_inf_ndvi_df = pd.DataFrame(columns = ['scene','veg_index']+metric_names)
  for dname in next(os.walk(real_dir))[1]:
    scene, day = dname.split('_')
    file_paths = [f"{real_dir}/{dname}/{day}_s2.tif",
                  f"{pred_dir}/{dname}_pred.tif",
                  f"{pred_ndvi_dir}/{dname}_pred_ndvi.tif"]

    missing_files = [] 
    for fp in file_paths:
      if not os.path.isfile(fp):
        missing_files.append(fp)
    if len(missing_files)>0:
      print(f"[SKIP] The following files are missing: {missing_files}")
      continue
    try:
      """
      s2 bands grand truth s2b(GT)
      s2 bands inferred s2b(I)
      NDVI inferred NDVI(I)
      """
      print("Loading images")
      s2_gt_test = load_image(file_paths[0]) 
      s2_gt = load_image(file_paths[0], selected_bands) 
      s2_inf = load_image(file_paths[1])
      scale_ref = 10000
      s2_gt = s2_gt/scale_ref
      #s2_inf = s2_inf/scale_ref
      ndvi_inf = load_predicted_ndvi(file_paths[2])
      ndvi_inf = ndvi_inf.reshape(ndvi_inf.shape[1],ndvi_inf.shape[2])
      """
      compute_indices() method, using :
        s2b(GT)
        s2b(I)
      """
      # put the indices in the config file
      ind_from_gt, ind_names_from_gt = compute_vegetation_indices(s2_gt)
      ind_from_inf, ind_names_from_inf = compute_vegetation_indices(s2_inf)
      """
      Plot 2D hitograms for the indices
        ind_from_gt
        ind_from_inf
        ndvi_inf
      """
      plot_histo_2d(ind_from_gt, ind_names_from_gt, dname, prefix="gt")
      plot_histo_2d(ind_from_inf, ind_names_from_inf, dname, prefix="inf")
      plot_histo_2d(np.array([ndvi_inf]), ['ndvi'], dname, prefix="ndvi_inf")

      """
      Scatter plots:
        s2b (GT) vs s2b (I)
        NDVI (GT) vs NDVI (I)
        indices (GT) vs indices (C)

      """

      plot_scatter_gt_vs_inf(ind_from_gt, ind_from_inf, ind_names_from_gt, dname, prefix="ind_gt_vs_inf")
      plot_scatter_gt_vs_inf(s2_gt, s2_inf, band_names, dname, prefix="s2_gt_vs_inf")
      plot_scatter_gt_vs_inf(np.array([ind_from_gt[0,:,:]]), np.array([ndvi_inf]), ['ndvi'], dname, prefix="ndvi_gt_vs_inf")

      plot_abs_error(ind_from_gt, ind_from_inf, ind_names_from_gt, dname, prefix="ind_gt_vs_inf")
      plot_abs_error(s2_gt, s2_inf, band_names, dname, prefix="s2_gt_vs_inf")
      plot_abs_error(np.array([ind_from_gt[0,:,:]]), np.array([ndvi_inf]), ['ndvi'], dname, prefix="ndvi_gt_vs_inf")



      """
      compute_metrics() method, from:
        s2b (GT) vs s2b (I)
        NDVI (GT) vs NDVI (I)
        indices (GT) vs indices (C)
      """
      # TODO: Change order of the input parameters
      gt_vs_comp_df = compute_all_metrics(gt_vs_comp_df, dname, ind_from_gt, ind_from_inf, ind_names_from_gt)
      gt_vs_inf_df = compute_all_metrics(gt_vs_inf_df, dname, s2_gt, s2_inf, band_names)
      gt_vs_inf_ndvi_df = compute_all_metrics(gt_vs_inf_ndvi_df, dname, np.array([ind_from_gt[0,:,:]]), np.array([ndvi_inf]), ["ndvi"])

    except Exception as e:
      print(f"[ERROR] Error for the scene {dname}: {e}")
  
  '''
  print(gt_vs_inf_df)
  prefix = "gt_vs_inf"
  plot_histos_from_df(gt_vs_inf_df, metric_names,prefix)
  print(gt_vs_comp_df)
  prefix = "gt_vs_comp"
  plot_histos_from_df(gt_vs_comp_df, metric_names,prefix)
  print(gt_vs_inf_ndvi_df)
  prefix = "gt_vs_inf_ndvi"
  plot_histos_from_df(gt_vs_inf_ndvi_df, metric_names,prefix)
  '''


  same_day_comparison = False
  
  # === MAIN LOOP SU TUTTE LE SCENE ===

  if same_day_comparison:
    pred_scenes = get_pred_scenes(pred_dir)
    real_scenes = get_real_scenes(real_dir)
    common_scenes = sorted(set(pred_scenes) & set(real_scenes))

    print(pred_scenes.keys())
    print(real_scenes.keys())

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
