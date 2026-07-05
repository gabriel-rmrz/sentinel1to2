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

def spectral_angle_mapper_map(img_gt: np.ndarray, img_inf: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Igual que spectral_angle_mapper pero devuelve el mapa (H, W) en lugar del mean.
    """
    gt  = _as_channel_first(img_gt)
    inf = _as_channel_first(img_inf)

    gt  = _nan_safe(gt)
    inf = _nan_safe(inf)

    C, H, W = gt.shape
    gt2  = gt.reshape(C, -1).T    # (N, C)
    inf2 = inf.reshape(C, -1).T   # (N, C)

    dot      = np.sum(gt2 * inf2, axis=1)
    norm_gt  = np.linalg.norm(gt2, axis=1)
    norm_inf = np.linalg.norm(inf2, axis=1)

    cos_theta = dot / (norm_gt * norm_inf + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta).reshape(H, W)   # (H, W)

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
# ── Map versions (per-pixel, same input format as compute_metrics) ────────────

def compute_metrics_map(img_gt: np.ndarray, img_inf: np.ndarray, metric_names: Sequence[str]) -> dict[str, np.ndarray]:
    """
    Compute per-pixel metric maps for (C,H,W) or (H,W) inputs.
    Returns a dict {metric_name: map (H,W)} for each supported metric.
    Unsupported metrics (ergas) return None.
    """
    gt  = _as_channel_first(_nan_safe(np.asarray(img_gt)))   # (C, H, W)
    inf = _as_channel_first(_nan_safe(np.asarray(img_inf)))  # (C, H, W)
    C, H, W = gt.shape

    maps: dict[str, np.ndarray | None] = {}

    for m in metric_names:
        ml = m.lower()

        if ml == "mae":
            # mean over channels → (H, W)
            maps[m] = np.mean(np.abs(inf - gt), axis=0)

        elif ml == "psnr":
            # per-pixel MSE → PSNR map (H, W)
            mse_map = np.mean((inf - gt) ** 2, axis=0)          # (H, W)
            data_range = 2.0
            with np.errstate(divide="ignore"):
                maps[m] = 10.0 * np.log10(data_range ** 2 / (mse_map + 1e-12))

        elif ml == "r2":
            # per-pixel R2 across channels (H, W)
            gt2  = gt.reshape(C, -1).T    # (N, C)
            inf2 = inf.reshape(C, -1).T
            ss_res = np.sum((gt2 - inf2) ** 2, axis=1)
            ss_tot = np.sum((gt2 - gt2.mean(axis=1, keepdims=True)) ** 2, axis=1)
            r2_flat = 1.0 - ss_res / (ss_tot + 1e-12)
            maps[m] = r2_flat.reshape(H, W)

        elif ml == "ssim":
            # one SSIM map per channel, then average over channels
            ssim_channels = []
            for c in range(C):
                _, ssim_map = structural_similarity(
                    gt[c], inf[c], data_range=2.0, full=True
                )
                ssim_channels.append(ssim_map)
            maps[m] = np.mean(ssim_channels, axis=0)            # (H, W)

        elif ml == "sam":
            maps[m] = spectral_angle_mapper_map(gt, inf)        # (H, W)

        elif ml == "ergas":
            maps[m] = None

        else:
            raise ValueError(f"Unknown metric '{m}'")

    return maps

def compute_metrics(target_type: str, img_gt: np.ndarray, img_inf: np.ndarray, metric_names: Sequence[str]) -> List[float]:
    """
    Compute metrics for a single-channel (H,W) or multi-channel (C,H,W) image.
    """
    data_range = 2.0
    if target_type == "bands":
        data_range = 1.0
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
            out.append(float(peak_signal_noise_ratio(img_gt, img_inf, data_range=data_range)))
            continue

        if ml == "ssim":
            if img_gt.ndim == 2:
                ssim = structural_similarity(img_gt, img_inf, data_range=data_range)
            else:
                # channel-first -> channel-last
                ssim = structural_similarity(
                    np.moveaxis(img_gt, 0, -1),
                    np.moveaxis(img_inf, 0, -1),
                    data_range=data_range,
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
    target_type: str,
    file: IO[str],
    scene_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    channel_names: Sequence[str],
    metric_names: Sequence[str],
    ndigits: int = 4,
) -> None:
    gt_cf   = _as_channel_first(gt)
    pred_cf = _as_channel_first(pred)

    if gt_cf.shape != pred_cf.shape:
        raise ValueError(f"Shape mismatch: gt {gt_cf.shape}, pred {pred_cf.shape}")

    C = gt_cf.shape[0]
    if len(channel_names) != C:
        raise ValueError(f"channel_names length {len(channel_names)} != number of channels {C}")

    per_channel   = [m for m in metric_names if m.lower() not in ("ergas", "sam")]
    ergas_enabled = any(m.lower() == "ergas" for m in metric_names)
    sam_enabled_  = any(m.lower() == "sam"   for m in metric_names)

    # ── per-channel rows: ergas and sam cells are empty ───────────────────────
    for i in range(C):
        computed = compute_metrics(target_type, gt_cf[i], pred_cf[i], per_channel)
        val_iter = iter(computed)
        ordered  = []
        for m in metric_names:
            if m.lower() in ("ergas", "sam"):
                ordered.append("")
            else:
                ordered.append(_fmt(next(val_iter), ndigits))
        file.write(",".join([scene_name, str(channel_names[i]), *ordered]) + "\n")

    # ── all_bands row: only ergas and sam filled, other cells empty ───────────
    if ergas_enabled or sam_enabled_:
        ordered = []
        for m in metric_names:
            ml = m.lower()
            if ml == "ergas":
                ordered.append(_fmt(float(ergas(gt_cf, pred_cf)), ndigits))
            elif ml == "sam":
                ordered.append(_fmt(float(spectral_angle_mapper(gt_cf, pred_cf)), ndigits))
            else:
                ordered.append("")
        file.write(",".join([scene_name, "all_bands", *ordered]) + "\n")

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
