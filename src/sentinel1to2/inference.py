from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import torch

from .tools.infer_on_scene import infer_on_scene
from .tools.get_model import get_model
from .tools.save_geotiff import save_geotiff
from .tools.load_and_stack_full import load_and_stack_full


def _load_norm_params_from_cache(config: dict):
    """
    Load normalization params (mean/std) from dataset cache.
    Expected path:
      <dataset_cache_dir>/norm/<norm_params_file>
    """
    dataset_cache_dir = Path(config["paths"]["dataset_cache_dir"])
    norm_file = config["preprocessing"].get("norm_params_file", "normalization_params.npz")
    params_path = dataset_cache_dir / "norm" / norm_file
    if not params_path.exists():
        raise FileNotFoundError(
            f"Normalization params file not found: {params_path}\n"
            f"Run preprocessing first or set preprocessing.norm_params_file correctly."
        )
    norm_params = np.load(params_path)
    return norm_params["mean"], norm_params["std"]


def _get_inference_input_dir(config: dict, sample_type: str) -> Path:
    """
    For 'val' we usually infer over the training input dir,
    for 'test' over inference.input_dir.
    """
    if sample_type == "val":
        return Path(config["preprocessing"]["input_dir"])
    return Path(config["inference"]["input_dir"])


def _get_inference_output_dir(config: dict, sample_type: str) -> Path:
    """
    All inference outputs go under run_dir.
      <run_dir>/inference/<sample_type>/
    """
    run_dir = Path(config["paths"]["run_dir"])
    out_subdir = config["inference"].get("output_subdir", f"inference/{sample_type}")
    # allow users to pass "inference/test" etc.
    return run_dir / out_subdir


def load_model_for_inference(config: dict, device: torch.device) -> torch.nn.Module:
    """
    Load generator model from run_dir checkpoints.
    Default: <run_dir>/checkpoints/best.pth
    """
    run_dir = Path(config["paths"]["run_dir"])
    ckpt_path = Path(config["inference"].get("checkpoint_path", run_dir / "checkpoints" / "best.pth"))

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first or set inference.checkpoint_path."
        )

    model = get_model(config)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def inference_one_scene(
    config: dict,
    scene_folder: str,
    model: torch.nn.Module,
    device: torch.device,
    sample_type: str = "test",
) -> Path:
    """
    Run inference on a single scene using a pre-loaded model.
    Returns the output GeoTIFF path.
    """
    logger = logging.getLogger(__name__)

    data_dir = _get_inference_input_dir(config, sample_type)
    out_dir = _get_inference_output_dir(config, sample_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = _load_norm_params_from_cache(config)

    # Load stack
    dsm, s1, wc, _s2_selected, _ind, ind_names, profile = load_and_stack_full(
        config, scene_folder, data_dir, mean, std
    )

    input_stack = np.concatenate([dsm, s1, wc], axis=0).astype(np.float32)
    logger.info(f"[{sample_type}] Scene={scene_folder} input_stack shape={input_stack.shape}")

    # Patch-wise inference (Gaussian blending supported)
    ps = int(config["inference"].get("patch_dimension", config["preprocessing"]["patch_dimension"])[0])
    stride = int(config["inference"].get("stride", 32))

    output = infer_on_scene(
        model,
        input_stack,
        device,
        patch_size=ps,
        stride=stride,
        batch_size=int(config["inference"].get("batch_size", 4)),
        use_gaussian=bool(config["inference"].get("use_gaussian", True)),
        gaussian_std=config["inference"].get("gaussian_std", None),
    )

    out_path = out_dir / f"{sample_type}__{scene_folder}__pred.tif"
    save_geotiff(output, profile, out_path)
    logger.info(f"✅ Saved: {out_path}")
    return out_path


def inference(
    config: dict,
    scene_folder: str,
    device: str | torch.device = "cuda",
    sample_type: str = "test",
) -> Path:
    """
    Backwards-friendly wrapper (keeps the original call style)
    but uses the new cache/run dirs.
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    device = torch.device(device if (isinstance(device, str) and torch.cuda.is_available()) else device)

    model = load_model_for_inference(config, device)
    return inference_one_scene(config, scene_folder, model, device, sample_type=sample_type)

