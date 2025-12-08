import torch
import numpy as np
from tqdm import tqdm
from .pad_image import pad_image


def gaussian_kernel2d(size: int, std: float, minval: float = 1e-6) -> np.ndarray:
    """
    Simple 2D Gaussian kernel, channel-agnostic.

    Returns:
        kernel: (size, size) float32
    """
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * std**2))
    kernel = np.maximum(kernel, minval)
    return kernel.astype(np.float32)


def infer_on_scene(
    model,
    input_stack: np.ndarray,
    device,
    patch_size: int = 128,
    stride: int = 32,
    batch_size: int = 4,
    use_gaussian: bool = True,
    gaussian_std: float | None = None,
):
    """
    Patch-wise inference with optional Gaussian blending of overlapping patches.

    Args:
        model:       PyTorch model (UNet, etc.), (B, C_in, H, W) -> (B, C_out, H, W)
        input_stack: np.ndarray (C_in, H, W) pre-normalized
        device:      torch.device or str
        patch_size:  size of square patches
        stride:      stride between patch top-left corners
        batch_size:  number of patches per forward pass
        use_gaussian: if True, use Gaussian weighting on overlaps
        gaussian_std: std of Gaussian window; if None, defaults to patch_size / 4

    Returns:
        output_map: np.ndarray (C_out, orig_H, orig_W)
    """
    model.eval()
    device = torch.device(device)

    # Pad to ensure full coverage with given patch_size/stride
    input_stack, orig_h, orig_w = pad_image(input_stack, patch_size, stride)
    c_in, h, w = input_stack.shape

    # Determine output channels C_out
    with torch.no_grad():
        dummy_input = torch.zeros(
            (1, c_in, patch_size, patch_size), dtype=torch.float32, device=device
        )
        c_out = model(dummy_input).shape[1]

    # Accumulators
    output_map = np.zeros((c_out, h, w), dtype=np.float32)
    weight_map = np.zeros((1, h, w), dtype=np.float32)

    # Build patch coordinates (ensure coverage up to the borders)
    h_coords = list(range(0, h - patch_size + 1, stride))
    w_coords = list(range(0, w - patch_size + 1, stride))
    if h_coords[-1] + patch_size < h:
        h_coords.append(h - patch_size)
    if w_coords[-1] + patch_size < w:
        w_coords.append(w - patch_size)

    coords = [(top, left) for top in h_coords for left in w_coords]

    # Gaussian window
    if use_gaussian:
        if gaussian_std is None:
            gaussian_std = patch_size / 4.0
        g2d = gaussian_kernel2d(patch_size, gaussian_std)        # (ps, ps)
    else:
        g2d = None

    # Patch-wise inference in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(coords), batch_size), desc="Infer patches"):
            batch_coords = coords[i : i + batch_size]

            # Build batch of patches: (B, C_in, ps, ps)
            patches = []
            for (top, left) in batch_coords:
                patch = input_stack[:, top : top + patch_size, left : left + patch_size]
                patches.append(patch)
            patches = np.stack(patches, axis=0).astype(np.float32)

            patch_tensor = torch.from_numpy(patches).to(device)  # (B, C_in, ps, ps)
            preds = model(patch_tensor).cpu().numpy()            # (B, C_out, ps, ps)

            # Accumulate predictions
            for b, (top, left) in enumerate(batch_coords):
                pred = preds[b]  # (C_out, ps, ps)

                if use_gaussian:
                    # Apply Gaussian weights per patch
                    output_map[:, top : top + patch_size, left : left + patch_size] += (
                        pred * g2d[None, :, :]
                    )
                    weight_map[:, top : top + patch_size, left : left + patch_size] += g2d[
                        None, :, :
                    ]
                else:
                    output_map[:, top : top + patch_size, left : left + patch_size] += pred
                    weight_map[:, top : top + patch_size, left : left + patch_size] += 1.0

    # Avoid division by zero, normalize by weights
    weight_map[weight_map == 0.0] = 1.0
    output_map /= weight_map

    # Crop back to original size
    return output_map[:, :orig_h, :orig_w]

