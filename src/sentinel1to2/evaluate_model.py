import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import skimage.metrics


"""
0 np.clip(ndvi, -1, 1),
1 np.clip(gndvi, -1, 1),
2 np.clip(ndre, -1, 1),
3 np.clip(reci, -1, 10),   # oppure anche (0, 10) se RECI < 0 non ha senso
4 np.clip(msi, 0, 10),
5 np.clip(ndwi, -1, 1),
6 np.clip(evi, 0, 2),
"""

KKK = 0 
# TODO: Add evaluation for the test sample as well.
# Add input bands to the config.


def evaluate_model(model, device, val_loader, num_samples=5):
    
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
