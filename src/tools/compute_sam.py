import numpy as np

def compute_sam(gt, pred):
  """
  Calcola lo Spectral Angle Mapper (SAM) tra due immagini spettrali 3D (C, H, W).
  Restituisce la media degli angoli in radianti.
  """
  # Reshape a (N, C) dove N = H*W
  gt_ = gt.reshape(gt.shape[0], -1).T  # (N, C)
  pred_ = pred.reshape(pred.shape[0], -1).T

  # Rimuovi i pixel con NaN
  valid_mask = np.all(~np.isnan(gt_), axis=1) & np.all(~np.isnan(pred_), axis=1)
  gt_ = gt_[valid_mask]
  pred_ = pred_[valid_mask]

  # Evita divisioni per zero
  dot_product = np.sum(gt_ * pred_, axis=1)
  norm_gt = np.linalg.norm(gt_, axis=1)
  norm_pred = np.linalg.norm(pred_, axis=1)
  denom = norm_gt * norm_pred + 1e-8

  cos_theta = np.clip(dot_product / denom, -1, 1)
  sam = np.arccos(cos_theta)

  return np.mean(sam)
