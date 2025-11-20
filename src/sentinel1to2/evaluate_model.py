import logging
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import skimage.metrics
from pathlib import Path
from .tools.compute_metrics import compute_all_metrics
from .tools.compute_vegetation_indices import compute_vegetation_indices
from .tools.produce_outputs_from_df import produce_outputs_from_df


# TODO: Add evaluation for the test sample as well.
# Add input bands to the config.

'''
def  plot_learning_curves(metric_vals, metric_name):
  fig, ax = plt.subplots()
  for i in range(metric_vals.shape[1]):
    ax.plot(metric_vals[:,i])
  fig.savefig(f"plots/learning_curves/{metric_name}.png")
  plt.close(fig)

'''
def plot_scatter_gt_vs_inf(output_dir, indices_gt, indices_inf, names, scene, prefix):
  output_dir.mkdir(parents=True, exist_ok=True)
  # TODO: Add x=y line
  # TODO: Add to the config file the indices selected and the their limits
  for i in range(indices_gt.shape[0]):
    fig, ax = plt.subplots()
    ax.scatter(indices_gt[i].flatten(), indices_inf[i].flatten(), s=1, alpha=0.3)
    fig.savefig(output_dir / f"{prefix}_{scene}_{names[i]}.png")
    plt.close(fig)

def plot_abs_error(output_dir, indices_gt, indices_inf, names, scene, prefix):
  output_dir.mkdir(parents=True, exist_ok=True)
  for i in range(indices_gt.shape[0]):
    fig, ax = plt.subplots()
    plt.hist(np.abs(indices_gt[i].flatten()-indices_inf[i].flatten()))
    fig.savefig(output_dir / f"{prefix}_{scene}_{names[i]}.png")
    plt.close(fig)

def plot_comparison_histos_2d(output_dir, indices1, indices2, names, scene, prefix):
  output_dir.mkdir(parents=True, exist_ok=True)
  for i in range(indices1.shape[0]):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(6)
    fig.set_figwidth(14)
    # find minimum of minima & maximum of maxima
    minmin = np.min([np.min(indices1[i]), np.min(indices2[i])])
    maxmax = np.max([np.max(indices1[i]), np.max(indices2[i])])
    
    im1 = axes[0].imshow(indices1[i], vmin=minmin, vmax=maxmax,
                         extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
    im2 = axes[1].imshow(indices2[i], vmin=minmin, vmax=maxmax,
                         extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
    
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)

    fig.savefig(output_dir / f"{prefix}_{scene}_{names[i]}.png")
    plt.close(fig)

def plot_histo_2d(output_dir, indices, names, scene, prefix):
  for i in range(indices.shape[0]):
    fig, ax = plt.subplots()
    im = ax.imshow(indices[i])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    fig.savefig(output_dir / f"{prefix}_{scene}_{names[i]}.png")
    plt.close(fig)

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

  model.eval()
  sampled_preds = []
  sampled_targets = []
  with torch.no_grad():
    band_names = ["b1","blue", "green", "red", "b5", "rededge", "b7", "nir","b8a","b9", "b10", "swir", "b12"]
    selected_bands = [1,2,3,4,5,6,7,10,11]
    band_names = [band_names[j] for j in selected_bands]
    metric_names =  ["mae", "psnr", "ssim", "r2"]
    gt_vs_inf_df = pd.DataFrame(columns = ['scene','band']+metric_names)
    gt_vs_comp_df = pd.DataFrame(columns = ['scene','veg_index']+metric_names)
    '''
    scene_gt_vs_inf_df = pd.DataFrame(columns = ['scene','band']+metric_names)
    scene_gt_vs_comp_df = pd.DataFrame(columns = ['scene','veg_index']+metric_names)
    gt_vs_inf_sample_df = pd.DataFrame(columns = ['scene','band']+metric_names)
    gt_vs_comp_sample_df = pd.DataFrame(columns = ['scene','veg_index']+metric_names)
    '''

    '''
    scenes_ind = {}
    print(val_loader[0])
    # getting the indices for the patches of each scene
    for i, (inputs, targets, scenes, patch_idx) in enumerate(val_loader):
      for j in len(scenes):
        print(scenes[j])
        if scene[j] in scenes_ind.keys():
          scenes_ind[scene[j]].append((i,j)) 
        else:
          scenes_ind = [(i,j)]
    '''

    scene_count = 0
    for i, (inputs, targets, scenes, patch_idx) in enumerate(val_loader):
      inputs, targets = inputs.to(device), targets.to(device)

      outputs = model(inputs)
      target_scene = [] 
      output_scene = [] 
      for j in range(min(num_samples, inputs.size(0))):
        target_patch = targets[j].cpu().squeeze().numpy()
        output_patch = outputs[j].cpu().squeeze().numpy()
        ind_from_gt, ind_names_from_gt = compute_vegetation_indices(target_patch)
        ind_from_inf, ind_names_from_inf = compute_vegetation_indices(output_patch)
        gt_vs_comp_df = compute_all_metrics(gt_vs_comp_df, scenes[j], ind_from_gt, ind_from_inf, ind_names_from_gt)
        gt_vs_inf_df = compute_all_metrics(gt_vs_inf_df, scenes[j], target_patch, output_patch, band_names)

        if scene_count < config["evaluation"]["scenes_to_plot"]:
          print(f"output_patch.shape: {output_patch.shape}")

          #plot_histo_2d(job_plots_ind_h2d, ind_from_gt, ind_names_from_gt, scenes[j], prefix=f"val_patch_{scene_count}_computed_from_gt")
          #plot_histo_2d(job_plots_ind_h2d, ind_from_inf, ind_names_from_inf, scenes[j], prefix=f"val_patch_{scene_count}_computed_from_inf")
          plot_comparison_histos_2d(job_plots_dir / 'patches/val/indices/histos2d_comparison', 
              ind_from_gt, 
              ind_from_inf, 
              ind_names_from_gt, 
              scenes[j], 
              prefix=f"{scene_count}_computed_from_gt_vs_inf")
         plot_comparison_histos_2d(job_plots_dir / 'patches/val/bands/histos2d_comparison', 
             target_patch, 
             output_patch, 
             band_names, 
             scenes[j], 
             prefix=f"{scene_count}_gt_vs_inf")

          plot_scatter_gt_vs_inf(job_plots_dir / 'patches/val/indices/scatter_gt_vs_inf', 
              ind_from_gt, 
              ind_from_inf, 
              ind_names_from_gt, 
              scenes[j], 
              prefix=f"{scene_count}_computed_from_gt_vs_inf")
          plot_scatter_gt_vs_inf(job_plots_dir / 'patches/val/bands/scatter_gt_vs_inf', 
              target_patch, 
              output_patch, 
              band_names, 
              scenes[j], 
              prefix=f"{scene_count}_computed_from_gt_vs_inf")

          plot_abs_error(job_plots_dir / 'patches/val/indices/histos_abs_error', 
              ind_from_gt, 
              ind_from_inf, 
              ind_names_from_gt, 
              scenes[j], 
              prefix=f"{scene_count}_computed_from_gt_vs_inf")

          plot_abs_error(job_plots_dir / 'patches/val/bands/histos_abs_error', 
              target_patch, 
              output_patch, 
              band_names, 
              scenes[j], 
              prefix=f"{scene_count}_computed_from_gt_vs_inf")

        ''' 

        # Numero di pixel da campionare per patch
        num_pix = 512
        # We can't compute spatial structured metrics flattening the arrays
        # We may can have to compute a reduced number of metrics for the subsample.
        if target_patch.shape[1]*target_patch.shape[2] > num_pix:
          indices_x = [random.randint(0, target_patch.shape[1]-1) for i in range(num_pix)]
          indices_y = [random.randint(0, target_patch.shape[2]-1) for i in range(num_pix)]
          target_patch_sample = target_patch[:, indices_x, indices_y]
          output_patch_sample = output_patch[:, indices_x, indices_y]
          ind_from_gt_sample, ind_names = compute_vegetation_indices(target_patch_sample)
          ind_from_inf_sample, _ind_names = compute_vegetation_indices(output_patch_sample)
          gt_vs_comp_sample_df = compute_all_metrics(gt_vs_comp_sample_df, scenes[j], ind_from_gt_sample, ind_from_inf_sample, ind_names)
          gt_vs_inf_sample_df = compute_all_metrics(gt_vs_inf_sample_df, scenes[j], target_patch_sample, output_patch_sample, band_names)
        '''
        scene_count += 1
      
 
      if i * val_loader.batch_size >= num_samples:
          break

    prefix1 = "val_patches_indices_gt_vs_comp"
    table1_path = job_tables_dir / (prefix1 + ".csv")
    logging.info(f"Saving output: {table1_path}")
    gt_vs_comp_df.to_csv(table1_path, index=False)
    produce_outputs_from_df(gt_vs_comp_df, config, metric_names,prefix1)

    prefix2 = "val_patches_bands_gt_vs_inf"
    table2_path = job_tables_dir / (prefix2 + ".csv")
    logging.info(f"Saving output: {table2_path}")
    gt_vs_inf_df.to_csv(table2_path, index=False)
    produce_outputs_from_df(gt_vs_inf_df, config, metric_names,prefix2)
    '''
    prefix = "val_patches_sample_gt_vs_comp"
    gt_vs_comp_sample_df.to_csv(f"tables/{prefix}.csv", index=False)
    produce_outputs_from_df(gt_vs_comp_sample_df, config, metric_names,prefix)
    prefix = "val_scenes_sample_gt_vs_comp"
    scene_gt_vs_comp_df.to_csv(f"tables/{prefix}.csv", index=False)
    produce_outputs_from_df(scene_gt_vs_comp_df, config, metric_names,prefix)
    '''

  '''  
  model.eval()
  mae_list = []
  psnr_list = []
  ssim_list = []
  sampled_preds = []
  sampled_targets = []
  for KKK in range(9):
    with torch.no_grad():
      for i, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        print(f"inputs.cpu().numpy().shape: {inputs.cpu().numpy().shape}")
        print(f"targets.cpu().numpy().shape: {targets.cpu().numpy().shape}")
        print(f"outputs.cpu().numpy().shape: {outputs.cpu().numpy().shape}")
        exit()
        for j in range(min(num_samples, inputs.size(0))):
          input_patch = inputs[j].cpu()
          target_patch = targets[j].cpu().squeeze(0)[KKK]
          output_patch = outputs[j].cpu().squeeze(0)[KKK]

  
          # Calcola metriche
          mae = torch.abs(output_patch - target_patch).mean().item()
          psnr = skimage.metrics.peak_signal_noise_ratio(
              target_patch.numpy(), output_patch.numpy(), data_range=1.0
          )
          ssim = skimage.metrics.structural_similarity(
              target_patch.numpy(), output_patch.numpy(), data_range=1.0
          )
  
          mae_list.append(mae)
          psnr_list.append(psnr)
          ssim_list.append(ssim)
  
          pred_flat = output_patch.flatten().numpy()
          target_flat = target_patch.flatten().numpy()
        
          # Numero di pixel da campionare per patch
          num_pix = 512
          if len(pred_flat) > num_pix:
              indices = random.sample(range(len(pred_flat)), num_pix)
              sampled_preds.extend(pred_flat[indices])
              sampled_targets.extend(target_flat[indices])
          else:
              sampled_preds.extend(pred_flat)
              sampled_targets.extend(target_flat)
  
          """
          # Visualizza
          fig, axs = plt.subplots(1, 3, figsize=(14, 4))
          
          # NDVI range: -1 a 1
          vmin_ndvi, vmax_ndvi = 0, 0.5
          error = torch.abs(target_patch - output_patch)
          vmin_err, vmax_err = 0, 1  # NDVI unità di errore massimo possibile
          
          im0 = axs[0].imshow(target_patch, cmap='viridis', vmin=vmin_ndvi, vmax=vmax_ndvi)
          axs[0].set_title('Target NDVI')
          axs[0].axis('off')
          cbar0 = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
          cbar0.set_label("NDVI")
          
          im1 = axs[1].imshow(output_patch, cmap='viridis', vmin=vmin_ndvi, vmax=vmax_ndvi)
          axs[1].set_title('Predicted NDVI')
          axs[1].axis('off')
          cbar1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
          cbar1.set_label("NDVI")
          
          im2 = axs[2].imshow(error, cmap='magma', vmin=vmin_err, vmax=vmax_err)
          axs[2].set_title('Absolute Error')
          axs[2].axis('off')
          cbar2 = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
          cbar2.set_label("NDVI units")
          
          plt.tight_layout()
          plt.show()
          """
          
          
        if i * val_loader.batch_size >= num_samples:
            break

    print(f"BAND:  {KKK:.1f}")
    print(f"MAE:  {np.mean(mae_list):.4f}")
    print(f"PSNR: {np.mean(psnr_list):.2f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    sampled_preds_ar = np.array(sampled_preds)
    sampled_targets_ar = np.array(sampled_targets)
    sampled_errors_ar = np.abs(sampled_preds_ar - sampled_targets_ar)
    from sklearn.metrics import r2_score
    r2 = r2_score(sampled_targets_ar, sampled_preds_ar)
    print("R^{2}:", r2)
    np.mean(sampled_errors_ar)
  '''
