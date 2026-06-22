from __future__ import annotations

import logging
from typing import IO, List, Sequence

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


# -----------------------------
# Shape helpers
# -----------------------------
def _as_channel_first(x: np.ndarray) -> np.ndarray:
    """Ensure channel-first array (C, H, W). Accepts (H, W) -> (1, H, W)."""
    x = np.asarray(x)
    if x.ndim == 2:
        return x[None, ...]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")


def _nan_safe(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf to avoid metric explosions."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _fmt(x: float, ndigits: int = 4) -> str:
    """Format metric value for CSV."""
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return ""
    return f"{float(x):.{ndigits}f}"


# -----------------------------
# SAM
# -----------------------------
def spectral_angle_mapper(img_gt: np.ndarray, img_inf: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Spectral Angle Mapper (SAM) for channel-first images.

    Inputs:
      img_gt, img_inf: (C,H,W) or (H,W) (treated as 1-band, but not recommended)

    Returns:
      mean SAM in radians
    """
    gt = _as_channel_first(img_gt)
    inf = _as_channel_first(img_inf)

    if gt.shape != inf.shape:
        raise ValueError(f"SAM shape mismatch: gt {gt.shape}, inf {inf.shape}")

    gt = _nan_safe(gt)
    inf = _nan_safe(inf)

    C, H, W = gt.shape
    gt2 = gt.reshape(C, -1).T   # (N, C)
    inf2 = inf.reshape(C, -1).T # (N, C)

    dot = np.sum(gt2 * inf2, axis=1)
    norm_gt = np.linalg.norm(gt2, axis=1)
    norm_inf = np.linalg.norm(inf2, axis=1)

    cos_theta = dot / (norm_gt * norm_inf + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angles = np.arccos(cos_theta)
    return float(np.mean(angles))

# -----------------------------
# ERGAS
# -----------------------------
def ergas(img_gt: np.ndarray, img_inf: np.ndarray, ratio: float = 1.0, eps: float = 1e-8) -> float:
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) for
    channel-first images. Lower is better; 0 is a perfect reconstruction.

    Inputs:
      img_gt, img_inf: (C,H,W) or (H,W) (treated as 1-band; see note below)
      ratio: h/l spatial-resolution ratio; 1.0 for same-resolution translation.

    Returns:
      ERGAS value. Per-band RMSE normalized by each band's reference mean,
      aggregated across bands.
    """
    gt = _as_channel_first(img_gt)
    inf = _as_channel_first(img_inf)

    if gt.shape != inf.shape:
        raise ValueError(f"ERGAS shape mismatch: gt {gt.shape}, inf {inf.shape}")

    gt = _nan_safe(gt).astype(np.float64)
    inf = _nan_safe(inf).astype(np.float64)

    C = gt.shape[0]
    gt2 = gt.reshape(C, -1)
    inf2 = inf.reshape(C, -1)

    mse = np.mean((gt2 - inf2) ** 2, axis=1)   # (C,) per-band MSE
    mu = np.mean(gt2, axis=1)                   # (C,) per-band reference mean

    return float(100.0 * ratio * np.sqrt(np.mean(mse / (mu ** 2 + eps))))

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(img_gt: np.ndarray, img_inf: np.ndarray, metric_names: Sequence[str]) -> List[float]:
    """
    Compute metrics for a single-channel (H,W) or multi-channel (C,H,W) image.
    """
    img_gt = _nan_safe(np.asarray(img_gt))
    img_inf = _nan_safe(np.asarray(img_inf))

    out: List[float] = []

    for m in metric_names:
        ml = str(m).lower()

        if ml == "ergas":
            out.append(float(ergas(img_gt, img_inf)))
            continue

        if ml == "mae":
            out.append(float(np.mean(np.abs(img_inf - img_gt))))
            continue

        if ml == "psnr":
            out.append(float(peak_signal_noise_ratio(img_gt, img_inf, data_range=2.0)))
            continue

        if ml == "ssim":
            if img_gt.ndim == 2:
                ssim = structural_similarity(img_gt, img_inf, data_range=2.0)
            else:
                # channel-first -> channel-last
                ssim = structural_similarity(
                    np.moveaxis(img_gt, 0, -1),
                    np.moveaxis(img_inf, 0, -1),
                    data_range=2.0,
                    channel_axis=-1,
                )
            out.append(float(ssim))
            continue

        if ml == "r2":
            out.append(float(r2_score(img_gt.reshape(-1), img_inf.reshape(-1))))
            continue

        if ml == "sam":
            out.append(float(spectral_angle_mapper(img_gt, img_inf)))
            continue

        raise ValueError(f"Unknown metric '{m}'")

    return out


# -----------------------------
# CSV Writers
# -----------------------------
def write_metrics_header(file: IO[str], name_col: str, metric_names: Sequence[str]) -> None:
    file.write(",".join(["scene", name_col, *metric_names]) + "\n")


def write_per_channel_metrics(
    file: IO[str],
    scene_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    channel_names: Sequence[str],
    metric_names: Sequence[str],
    ndigits: int = 4,
) -> None:
    """
    Writes one row per channel: scene, channel_name, metrics...
    This is what your old compute_all_metrics() did (but cleaner).
    """
    gt_cf = _as_channel_first(gt)
    pred_cf = _as_channel_first(pred)

    if gt_cf.shape != pred_cf.shape:
        raise ValueError(f"Shape mismatch: gt {gt_cf.shape}, pred {pred_cf.shape}")

    C = gt_cf.shape[0]
    if len(channel_names) != C:
        raise ValueError(f"channel_names length {len(channel_names)} != number of channels {C}")

    for i in range(C):
        vals = compute_metrics(gt_cf[i], pred_cf[i], metric_names)
        file.write(",".join([scene_name, str(channel_names[i]), *[_fmt(v, ndigits) for v in vals]]) + "\n")


def write_sam_header(file: IO[str]) -> None:
    file.write("scene,sam\n")


def write_scene_sam(
    file: IO[str],
    scene_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    eps: float = 1e-8,
    ndigits: int = 6,
) -> None:
    """
    Writes one row per scene: scene,sam
    Only writes if gt/pred are multi-channel (C>1).
    """
    gt_cf = _as_channel_first(gt)
    pred_cf = _as_channel_first(pred)

    if gt_cf.shape != pred_cf.shape:
        raise ValueError(f"Shape mismatch: gt {gt_cf.shape}, pred {pred_cf.shape}")

    if gt_cf.shape[0] <= 1:
        return

    sam_val = spectral_angle_mapper(gt_cf, pred_cf, eps=eps)
    file.write(f"{scene_name},{_fmt(sam_val, ndigits)}\n")

def write_ergas_header(file: IO[str]) -> None:
    file.write("scene,ergas\n")


def write_scene_ergas(
    file: IO[str],
    scene_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    ratio: float = 1.0,
    eps: float = 1e-8,
    ndigits: int = 6,
) -> None:
    """
    Writes one row per scene: scene,ergas
    Only writes if gt/pred are multi-channel (C>1).
    """
    gt_cf = _as_channel_first(gt)
    pred_cf = _as_channel_first(pred)

    if gt_cf.shape != pred_cf.shape:
        raise ValueError(f"Shape mismatch: gt {gt_cf.shape}, pred {pred_cf.shape}")

    if gt_cf.shape[0] <= 1:
        return

    erg = ergas(gt_cf, pred_cf, ratio=ratio, eps=eps)
    file.write(f"{scene_name},{_fmt(erg, ndigits)}\n")
