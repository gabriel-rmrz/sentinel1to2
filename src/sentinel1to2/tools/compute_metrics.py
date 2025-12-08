import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from sklearn.metrics import r2_score

def spectral_angle_mapper(img_gt, img_inf, eps=1e-8):
    """
    Mean Spectral Angle Mapper (SAM) for channel-first images.

    Supports:
      - (C, H, W)
      - (H, W)  -> treated as 1-band

    Returns:
      Mean SAM in radians.
    """
    # Ensure (C, H, W)
    if img_gt.ndim == 2:
        img_gt = img_gt[None, ...]
        img_inf = img_inf[None, ...]

    C, H, W = img_gt.shape

    gt = img_gt.reshape(C, -1).T     # (N, C)
    inf = img_inf.reshape(C, -1).T   # (N, C)

    dot = np.sum(gt * inf, axis=1)
    norm_gt = np.linalg.norm(gt, axis=1)
    norm_inf = np.linalg.norm(inf, axis=1)

    cos_theta = dot / (norm_gt * norm_inf + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angles = np.arccos(cos_theta)
    return np.mean(angles)

def compute_metrics(img_gt, img_inf, metric_names):
    metrics = []

    if 'mae' in metric_names:
        mae = np.abs(img_inf - img_gt).mean()
        metrics.append(mae)

    if 'psnr' in metric_names:
        psnr = peak_signal_noise_ratio(img_gt, img_inf, data_range=1.0)
        metrics.append(psnr)

    if 'ssim' in metric_names:
        if img_gt.ndim == 2:  # single-band
            ssim = structural_similarity(img_gt, img_inf, data_range=1.0)
        else:  # channel-first → convert to channel-last
            ssim = structural_similarity(
                np.moveaxis(img_gt, 0, -1),
                np.moveaxis(img_inf, 0, -1),
                data_range=1.0,
                channel_axis=-1
            )
        metrics.append(ssim)

    if 'r2' in metric_names:
        r2 = r2_score(
            img_gt.reshape(-1),
            img_inf.reshape(-1)
        )
        metrics.append(r2)

    if 'sam' in metric_names:
        sam = spectral_angle_mapper(img_gt, img_inf)
        metrics.append(sam)

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
