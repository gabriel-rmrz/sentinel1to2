import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import skimage.metrics


KKK = 0 
# TODO: Add evaluation for the test sample as well.
# Add input bands to the config.

def  plot_learning_curves(metric_vals, metric_name):
  fig, ax = plt.subplots()
  for i in range(metric_vals.shape[1]):
    ax.plot(metric_vals[:,i])
  fig.savefig(f"plots/learning_curves/{metric_name}.png")
  plt.close(fig)


def evaluate_model(model, device, val_loader, num_samples=5):
  model.eval()
  mae_list = []
  psnr_list = []
  ssim_list = []
  mae_list_all_bands = []
  psnr_list_all_bands = []
  ssim_list_all_bands = []
  sampled_preds = []
  sampled_targets = []
  with torch.no_grad():
    for i, (inputs, targets) in enumerate(val_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      mae_list_per_epoch = []
      psnr_list_per_epoch = []
      ssim_list_per_epoch = []
      sampled_preds_per_epoch = []
      sampled_targets_per_epoch = []

      target_patch_all_bands = targets.cpu().flatten()
      output_patch_all_bands = outputs.cpu().flatten()
      # Calcola metriche
      mae_all_bands = torch.abs(output_patch_all_bands - target_patch_all_bands).mean().item()
      psnr_all_bands = skimage.metrics.peak_signal_noise_ratio(
          target_patch_all_bands.numpy(), output_patch_all_bands.numpy(), data_range=1.0
      )
      ssim_all_bands = skimage.metrics.structural_similarity(
          target_patch_all_bands.numpy(), output_patch_all_bands.numpy(), data_range=1.0
      )
 
      mae_list_all_bands.append(mae_all_bands)
      psnr_list_all_bands.append(psnr_all_bands)
      ssim_list_all_bands.append(ssim_all_bands)

      for j in range(outputs.cpu().numpy().shape[1]): # Loop over the bands
        #input_patch = inputs[j].cpu()
        target_patch = targets[:,j,:,:].cpu().flatten()
        output_patch = outputs[:,j,:,:].cpu().flatten()

 
        # Calcola metriche
        mae = torch.abs(output_patch - target_patch).mean().item()
        psnr = skimage.metrics.peak_signal_noise_ratio(
            target_patch.numpy(), output_patch.numpy(), data_range=1.0
        )
        ssim = skimage.metrics.structural_similarity(
            target_patch.numpy(), output_patch.numpy(), data_range=1.0
        )
 
        mae_list_per_epoch.append(mae)
        psnr_list_per_epoch.append(psnr)
        ssim_list_per_epoch.append(ssim)
 
        pred_flat = output_patch.flatten().numpy()
        target_flat = target_patch.flatten().numpy()
      
        # Numero di pixel da campionare per patch
        num_pix = 512
        if len(pred_flat) > num_pix:
          indices = random.sample(range(len(pred_flat)), num_pix)
          sampled_preds_per_epoch.extend(pred_flat[indices])
          sampled_targets_per_epoch.extend(target_flat[indices])
        else:
          sampled_preds_per_epoch.extend(pred_flat)
          sampled_targets_per_epoch.extend(target_flat)
      mae_list.append(mae_list_per_epoch)
      psnr_list.append(psnr_list_per_epoch)
      ssim_list.append(ssim_list_per_epoch)
      sampled_preds.append(sampled_preds_per_epoch)
      sampled_preds.append(sampled_targets_per_epoch)
    mae_list = np.array(mae_list)
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)

  plot_learning_curves(mae_list, metric_name = 'mae')
  plot_learning_curves(psnr_list, metric_name = 'psnr')
  plot_learning_curves(ssim_list, metric_name = 'ssim')
  print(f"mae_list.shape: {mae_list.shape}")

  mae_list_all_bands = np.array([mae_list_all_bands]).transpose()
  psnr_list_all_bands = np.array([psnr_list_all_bands]).transpose()
  ssim_list_all_bands = np.array([ssim_list_all_bands]).transpose()
  print(f"mae_list_all_bands.shape: {mae_list_all_bands.shape}")
  plot_learning_curves(mae_list_all_bands, metric_name = 'mae_all_bands')
  plot_learning_curves(psnr_list_all_bands, metric_name = 'psnr_all_bands')
  plot_learning_curves(ssim_list_all_bands, metric_name = 'ssim_all_bands')


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
