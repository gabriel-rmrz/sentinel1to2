import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity  
from sklearn.metrics import r2_score
from itertools import combinations
from tqdm import tqdm
from .tools.produce_outputs_from_df import produce_outputs_from_df
from .tools.compute_vegetation_indices import compute_vegetation_indices
from .tools.load_image import load_image
from .tools.load_predicted_ndvi import load_predicted_ndvi
from .tools.get_pred_scenes import get_pred_scenes
from .tools.get_real_scenes import get_real_scenes
from .tools.compute_metrics import compute_all_metrics
from .tools.r2_by_class import r2_by_class
from .plotting.scatter_with_r2 import scatter_with_r2
from .plotting.scatter_with_r2_by_class import scatter_with_r2_by_class
from .plotting.plot_comparison_rgb_composites_2d import plot_comparison_rgb_composites_2d
from .plotting.plot_comparison_histos_2d import plot_comparison_histos_2d
from .plotting.plot_s2_composites_2d import plot_s2_composites_2d
from .plotting.plot_scatter_gt_vs_inf import plot_scatter_gt_vs_inf
from .plotting.plot_abs_error import plot_abs_error
from .plotting.plot_histo_2d import plot_histo_2d
from .plotting.plot_group_metric_histograms import plot_group_metric_histograms


def read_csv_to_list(path):
  rows = []
  with open(path, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row[0])
  return rows


    

def performance(config, real_dir, pred_dir, sample_type='test'):
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


  job_dir = Path(config["job"]["dir"])
  job_data_dir = job_dir / 'data'
  job_outputs_dir = job_dir / 'outputs'
  job_tables_dir = job_outputs_dir / 'tables'
  job_plots_dir = job_outputs_dir / 'plots'
  job_lists_dir = job_data_dir / 'lists'
  #real_dir = "/lustrehome/garamire/share/agri2intesa/s1_to_s2/test"
  real_dir = config["inference"]["input_dir"]
  pred_dir = job_data_dir / config["inference"]["output_dir"] 

  job_tables_dir.mkdir(parents=True, exist_ok=True)
  job_plots_dir.mkdir(parents=True, exist_ok=True)

  
  #TODO: Give the option to save the indice so they don't need to be recactulated every time. 
  # Although, it might take the same time to only produce the plots.

  list_of_scenes = read_csv_to_list(job_lists_dir / f'{sample_type}_scenes_inferred_list.csv')
  target_type = config["target"]["type"] # "indices"
  tile_type = "scenes"
  metric_names =  config["performance"][f"{target_type}_metric_names"] 
  if target_type == "bands":
    channel_names = config["target"]["all_bands"]#["b1","blue", "green", "red", "b5", "rededge", "b7", "nir","b8a","b9", "b10", "swir", "b12"]
    selected_channels = config["target"]["selected_bands"] #[1,2,3,4,5,6,7,10,11]
    channel_names = [channel_names[j] for j in selected_channels]

    indices_metric_names =  config["performance"]["indices_metric_names"] 
    prefix1 = f"{sample_type}_{tile_type}_{target_type}_gt_vs_comp"
    table1_path = job_tables_dir / (prefix1 + ".csv")
    gt_vs_comp_file = open(table1_path, 'w')
    gt_vs_comp_file.write(','.join(['scene','indices']+ indices_metric_names))
    gt_vs_comp_file.write('\n')
  else:
    channel_name = ['ndvi']

  prefix2 = "{sample_type}_{tile_type}_{target_type}_gt_vs_inf"
  table2_path = job_tables_dir / (prefix2 + ".csv")
  gt_vs_inf_file = open(table2_path, 'w')
  gt_vs_inf_file.write(','.join(['scene',target_type]+metric_names))
  gt_vs_inf_file.write('\n')
  ### gt_vs_inf_ndvi_df = pd.DataFrame(columns = ['scene','veg_index']+metric_names)
  scene_count = 0
  #for dname in tqdm(next(os.walk(real_dir))[1], "Perfomance on {sample_type} sample"):
  for dname in list_of_scenes:
    scene, day = dname.split('_')
    file_paths = [f"{real_dir}/{dname}/{day}_s2.tif",
                  f"{pred_dir}/{sample_type}_{dname}_pred.tif"]

    missing_files = [] 
    for fp in file_paths:
      if not os.path.isfile(fp):
        missing_files.append(fp)
    if len(missing_files)>0:
      continue
    try:
      """
      channels grand truth channels(GT)
      channels  inferred clases(I)
      """
      channels_gt = load_image(file_paths[0], selected_channels) 
      channels_inf = load_image(file_paths[1])
      scale_ref = 10000
      channels_gt = channels_gt/scale_ref

      """
      compute_indices() method, using :
        s2b(GT)
        s2b(I)
      """
      compute_all_metrics(gt_vs_inf_file, 
          dname, 
          channels_gt, 
          channels_inf, 
          channel_names,
          metric_names)
      # put the indices in the config file
      if target_type == 'bands':
        ind_from_gt, ind_names_from_gt = compute_vegetation_indices(config, channels_gt)
        ind_from_inf, ind_names_from_inf = compute_vegetation_indices(config, channels_inf)
        compute_all_metrics(gt_vs_comp_file, 
            dname, 
            ind_from_gt, 
            ind_from_inf, 
            ind_names_from_gt,
            indices_metric_names)

      if scene_count < config["evaluation"]["scenes_to_plot"]:
        if target_type == "bands":
          job_plots_comp_ind_dir = job_plots_dir / f'{tile_type}/{sample_type}'

          plot_histo_2d( f'{job_plots_comp_ind_dir}/indices/histos2d',
              ind_from_gt,
              ind_names_from_gt,
              dname,
              prefix=f"computed_from_gt")
          plot_histo_2d(jf'{job_plots_comp_ind_dir}/indices/histos2d',
              ind_from_inf,
              ind_names_from_inf,
              dname,
              prefix=f"computed_from_inf")
          plot_comparison_histos_2d(
              f'{job_plots_comp_ind_dir}/indices/histos2d_comparison',
              ind_from_gt,
              ind_from_inf,
              ind_names_from_gt,
              dname,
              prefix=f"computed_from_gt_inf")
          plot_scatter_gt_vs_inf(
              f'{job_plots_comp_ind_dir}/indices/scatter_gt_vs_inf',
              ind_from_gt,
              ind_from_inf,
              ind_names_from_gt,
              dname,
              prefix=f"computed_from_gt_vs_inf")
          plot_abs_error(f'{job_plots_comp_ind_dir}/indices/histos_abs_error',
              ind_from_gt,
              ind_from_inf,
              ind_names_from_gt,
              dname,
              prefix=f"computed_from_gt_vs_inf")

        plot_comparison_rgb_composites_2d(
            job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos2d_comparison',
            channels_gt,        # (C, H, W)
            channels_inf,       # (C, H, W)
            channel_names,   # e.g. ["b1","blue","green","red",...,"nir",...]
            dname,
            prefix=f"gt_inf",
        )

        plot_s2_composites_2d(
            job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos2d',
            channels_gt,        # (C, H, W)
            channel_names,   # e.g. ["b1","blue","green","red","b5",...]
            dname,
            prefix=f"gt",
        )
        plot_s2_composites_2d(
            job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos2d',
            channels_inf,        # (C, H, W)
            channel_names,   # e.g. ["b1","blue","green","red","b5",...]
            dname,
            prefix=f"inf",
        )
        plot_histo_2d(job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos2d',
            channels_gt,
            channel_names,
            dname,
            prefix=f"gt")
        plot_histo_2d(job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos2d',
            channels_inf,
            channel_names,
            dname,
            prefix=f"inf")


        plot_comparison_rgb_composites_2d(
            job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos2d_comparison',
            channels_gt,        # (C, H, W)
            channels_inf,       # (C, H, W)
            channel_names,   # e.g. ["b1","blue","green","red",...,"nir",...]
            dname,
            prefix=f"gt_inf",
        )
        plot_comparison_histos_2d(job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos2d_comparison',
            channels_gt,
            channels_inf,
            channel_names,
            dname,
            prefix=f"gt_inf")

        plot_scatter_gt_vs_inf(job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/scatter_gt_vs_inf',
            channels_gt,
            channels_inf,
            channel_names,
            dname,
            prefix=f"gt_vs_inf")

        plot_abs_error(job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/histos_abs_error',
            channels_gt,
            channels_inf,
            channel_names,
            dname,
            prefix=f"gt_vs_inf")
        scene_count += 1

    except Exception as e:
      print(f"[ERROR] Error for the scene {dname}: {e}")
  gt_vs_inf_file.close()
  gt_vs_comp_file.close()
  if target_type == "bands":
    print(table1_path)
    gt_vs_comp_df = pd.read_csv(table1_path)
    print(gt_vs_comp_df)
    produce_outputs_from_df(gt_vs_comp_df, config, indices_metric_names,prefix1)
    plot_group_metric_histograms(
        output_dir=Path(job_plots_dir / f'{tile_type}/{sample_type}/indices/metrics'),
        df=gt_vs_comp_df,              # scene, veg_index, mae, psnr, ssim, r2
        group_col="indices",
        metrics=indices_metric_names,
        prefix="gt_vs_comp",
    )
  gt_vs_inf_df = pd.read_csv(table2_path)
  print(gt_vs_inf_df)
  produce_outputs_from_df(gt_vs_inf_df, config, metric_names, prefix2)
  plot_group_metric_histograms(
      output_dir=Path(job_plots_dir / f'{tile_type}/{sample_type}/{target_type}/metrics'),
      df= gt_vs_inf_df,              # scene, veg_index, mae, psnr, ssim, r2
      group_col=target_type,
      metrics=metric_names,
      prefix="gt_vs_inf",
  )

