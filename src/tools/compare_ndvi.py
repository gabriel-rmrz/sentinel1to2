import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compare_ndvi(true_ndvi, pred_ndvi):
  mask = ~np.isnan(true_ndvi) & ~np.isnan(pred_ndvi)
  true = true_ndvi[mask]
  pred = pred_ndvi[mask]

  mae = np.mean(np.abs(true - pred))
  ssim_val = ssim(true, pred, data_range=1.0)
  psnr_val = psnr(true, pred, data_range=1.0)
  return mae, ssim_val, psnr_val
