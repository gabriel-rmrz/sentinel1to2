import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from sklearn.metrics import r2_score

def compute_metrics(img_gt, img_inf, metric_names):
  metrics = []
  if 'mae' in metric_names:
    mae = np.abs(img_inf - img_gt).mean()
    metrics.append(mae)
  if 'psnr' in metric_names:
    psnr = peak_signal_noise_ratio(img_gt, img_inf, data_range=1.0)
    metrics.append(psnr)
  if 'ssim' in metric_names:
    ssim = structural_similarity(img_gt, img_inf, data_range=1.0)
    metrics.append(ssim)
  if 'r2' in metric_names:
    r2 = r2_score(img_gt, img_inf)
    metrics.append(r2)
  #print(f"mae: {mae}, psnr: {psnr}, ssim: {ssim}, r2: {r2}")
  #metrics =  [mae, psnr, ssim, r2]
  return metrics

def compute_all_metrics(file, scene_name, g_truth, inference, names, metric_names):
   for i in range(g_truth.shape[0]):
     metrics = compute_metrics(g_truth[i], inference[i], metric_names)
     metrics = [f"{float(m):4f}" for m in metrics]
     file.write(','.join([scene_name, names[i]]+metrics))
     file.write('\n')

     '''
     df.loc[-1] = [scene_name, names[i]]+metrics
     df.index = df.index+1
     df = df.sort_index()
     '''
   #return df
