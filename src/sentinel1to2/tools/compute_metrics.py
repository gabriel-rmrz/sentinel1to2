import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from sklearn.metrics import r2_score

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
     metrics = compute_metrics(g_truth[i], inference[i])
     df.loc[-1] = [scene_name, names[i]]+metrics
     df.index = df.index+1
     df = df.sort_index()
   return df
