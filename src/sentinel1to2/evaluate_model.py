import logging
import csv
import math
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import skimage.metrics
from tqdm import tqdm
from pathlib import Path
from .tools.compute_metrics import compute_all_metrics
from .tools.compute_vegetation_indices import compute_vegetation_indices
from .tools.produce_outputs_from_df import produce_outputs_from_df
from .plotting.plot_comparison_rgb_composites_2d import plot_comparison_rgb_composites_2d
from .plotting.plot_comparison_histos_2d import plot_comparison_histos_2d
from .plotting.plot_s2_composites_2d import plot_s2_composites_2d
from .plotting.plot_scatter_gt_vs_inf import plot_scatter_gt_vs_inf
from .plotting.plot_abs_error import plot_abs_error
from .plotting.plot_histo_2d import plot_histo_2d
from .plotting.plot_group_metric_histograms import plot_group_metric_histograms

# TODO: Add evaluation for the test sample as well.
# Add input bands to the config.


def evaluate_model(model, config, device, val_loader, num_samples=5):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
  )
  job_dir = Path(config["job"]["dir"])
  job_data_dir = job_dir / 'data'
  job_outputs_dir = job_dir / 'outputs'
  job_tables_dir = job_outputs_dir / 'tables'
  job_plots_dir = job_outputs_dir / 'plots'

  job_tables_dir.mkdir(parents=True, exist_ok=True)
  job_plots_dir.mkdir(parents=True, exist_ok=True)

  target_type = config["target"]["type"]
  model.eval()
  sampled_preds = []
  sampled_targets = []
  with torch.no_grad():
    scene_count = 0
    band_names = ["b1","blue", "green", "red", "b5", "rededge", "b7", "nir","b8a","b9", "b10", "swir", "b12"]
    selected_bands = [1,2,3,4,5,6,7,10,11]
    channel_names = [band_names[j] for j in selected_bands]
    #metric_names =  ["mae", "psnr", "ssim", "r2"]
    bands_metric_names = config["performance"]["bands_metric_names"]
    indices_metric_names = config["performance"]["indices_metric_names"]
    if target_type == "bands":
      metric_names = bands_metric_names
      #gt_vs_comp_df = pd.DataFrame(columns = ['scene','indices']+bands_metric_names)
      prefix1 = "val_patches_indices_gt_vs_comp"
      table1_path = job_tables_dir / (prefix1 + ".csv")
      gt_vs_comp_file = open(table1_path, 'w')
      gt_vs_comp_file.write(','.join(['scene','indices']+indices_metric_names))
      gt_vs_comp_file.write('\n')
    elif target_type == "indices":
      metric_names = indices_metric_names
      channel_names = config["target"]["selected_indices"]

    #gt_vs_inf_df = pd.DataFrame(columns = ['scene','band']+metric_names)
    prefix2 = f"val_patches_{target_type}_gt_vs_inf"
    table2_path = job_tables_dir / (prefix2 + ".csv")
    gt_vs_inf_file = open(table2_path, 'w')
    gt_vs_inf_file.write(','.join(['scene',target_type]+metric_names))
    gt_vs_inf_file.write('\n')
    i = 0
    for inputs, targets, scenes, patch_idx in tqdm(val_loader, "Evaluating batches"):
      inputs, targets = inputs.to(device), targets.to(device)

      outputs = model(inputs)
      target_scene = [] 
      output_scene = [] 
      for j in range(min(num_samples, inputs.size(0))):
        target_patch = targets[j].cpu().squeeze().numpy()
        output_patch = outputs[j].cpu().squeeze().numpy()
        if target_type == "bands":
          ind_from_gt, ind_names_from_gt = compute_vegetation_indices(config, target_patch)
          ind_from_inf, ind_names_from_inf = compute_vegetation_indices(config, output_patch)
          compute_all_metrics(gt_vs_comp_file, scenes[j], ind_from_gt, ind_from_inf, ind_names_from_gt, indices_metric_names)
        compute_all_metrics(gt_vs_inf_file, scenes[j], target_patch, output_patch, channel_names, metric_names)

        if scene_count < config["evaluation"]["scenes_to_plot"]:

          if target_type == "bands":
            plot_comparison_histos_2d(
                job_plots_dir / 'patches/val/indices/histos2d_comparison', 
                ind_from_gt, 
                ind_from_inf, 
                ind_names_from_gt, 
                scenes[j], 
                prefix=f"{scene_count}_computed_from_gt_inf")

            plot_scatter_gt_vs_inf(job_plots_dir / 'patches/val/indices/scatter_gt_vs_inf', 
                ind_from_gt, 
                ind_from_inf, 
                ind_names_from_gt, 
                scenes[j], 
                prefix=f"{scene_count}_computed_from_gt_vs_inf")

            plot_abs_error(job_plots_dir / 'patches/val/indices/histos_abs_error', 
                ind_from_gt, 
                ind_from_inf, 
                ind_names_from_gt, 
                scenes[j], 
                prefix=f"{scene_count}_computed_from_gt_vs_inf")
            plot_histo_2d(job_plots_dir / 'patches/val/indices/histos2d', 
                ind_from_gt, 
                ind_names_from_gt, 
                scenes[j], 
                prefix=f"{scene_count}_computed_from_gt")
            plot_histo_2d(job_plots_dir / 'patches/val/indices/histos2d', 
                ind_from_inf, 
                ind_names_from_inf, 
                scenes[j], 
                prefix=f"{scene_count}_computed_from_inf")

          plot_scatter_gt_vs_inf(job_plots_dir / f'patches/val/{target_type}/scatter_gt_vs_inf', 
              target_patch, 
              output_patch, 
              channel_names, 
              scenes[j], 
              prefix=f"{scene_count}_gt_vs_inf")

          plot_comparison_histos_2d(job_plots_dir / f'patches/val/{target_type}/histos2d_comparison', 
             target_patch, 
             output_patch, 
             channel_names, 
             scenes[j], 
             prefix=f"{scene_count}_gt_inf")
          plot_s2_composites_2d(
              job_plots_dir / f'patches/val/{target_type}/histos2d',
              target_patch,        # (C, H, W)
              channel_names,   # e.g. ["b1","blue","green","red","b5",...]
              scenes[j],
              prefix=f"{scene_count}_gt",
          )
          plot_s2_composites_2d(
              job_plots_dir / f'patches/val/{target_type}/histos2d',
              output_patch,        # (C, H, W)
              channel_names,   # e.g. ["b1","blue","green","red","b5",...]
              scenes[j],
              prefix=f"{scene_count}_inf",
          )
          plot_histo_2d(job_plots_dir / f'patches/val/{target_type}/histos2d', 
              target_patch, 
              channel_names, 
              scenes[j], 
              prefix=f"{scene_count}_gt")
          plot_histo_2d(job_plots_dir / f'patches/val/{target_type}/histos2d', 
              output_patch, 
              channel_names, 
              scenes[j], 
              prefix=f"{scene_count}_inf")



          plot_comparison_rgb_composites_2d(
              job_plots_dir / f'patches/val/{target_type}/histos2d_comparison',
              target_patch,        # (C, H, W)
              output_patch,       # (C, H, W)
              channel_names,   # e.g. ["b1","blue","green","red",...,"nir",...]
              scenes[j],
              prefix=f"{scene_count}_gt_inf",
          )

          plot_abs_error(job_plots_dir / f'patches/val/{target_type}/histos_abs_error', 
              target_patch, 
              output_patch, 
              channel_names, 
              scenes[j], 
              prefix=f"{scene_count}_gt_vs_inf")

        scene_count += 1
      
        if i * val_loader.batch_size >= num_samples:
          break
      i+=1
    gt_vs_inf_file.close()
 
    if target_type == "bands":
      gt_vs_comp_file.close()
      gt_vs_comp_df = pd.read_csv(table1_path)
      print(gt_vs_comp_df)
      produce_outputs_from_df(gt_vs_comp_df, config, indices_metric_names,prefix1)
      plot_group_metric_histograms(
          output_dir=Path(job_plots_dir / f'patches/val/indices/metrics'),
          df=gt_vs_comp_df,              # scene, veg_index, mae, psnr, ssim, r2
          group_col="indices",
          metrics=indices_metric_names,
          prefix="gt_vs_comp",
      )
    gt_vs_inf_df = pd.read_csv(table2_path)
    print(gt_vs_inf_df)
    produce_outputs_from_df(gt_vs_inf_df, config, metric_names,prefix2)
    plot_group_metric_histograms(
        output_dir=Path(job_plots_dir / f'patches/val/{target_type}/metrics'),
        df= gt_vs_inf_df,              # scene, veg_index, mae, psnr, ssim, r2
        group_col=target_type,
        metrics=metric_names,
        prefix="gt_vs_inf",
    )

    """
    produce_outputs_from_df(gt_vs_comp_df, config, metric_names,prefix1)

    plot_group_metric_histograms(
        output_dir=Path(job_plots_dir / 'patches/val/indices/metrics'),
        df=gt_vs_comp_df,              # scene, veg_index, mae, psnr, ssim, r2
        group_col="veg_index",
        metrics=["mae", "psnr", "ssim", "r2"],
        prefix="gt_vs_comp",
    )

    produce_outputs_from_df(gt_vs_inf_df, config, metric_names,prefix2)

    plot_group_metric_histograms(
        output_dir=Path(job_plots_dir / 'patches/val/bands/metrics'),
        df= gt_vs_inf_df,              # scene, veg_index, mae, psnr, ssim, r2
        group_col="bands",
        metrics=["mae", "psnr", "ssim", "r2"],
        prefix="gt_vs_inf",
    )
    """
