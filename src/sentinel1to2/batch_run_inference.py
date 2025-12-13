from __future__ import annotations

import logging
import csv
from pathlib import Path
import torch

from .inference import load_model_for_inference, inference_one_scene


def write_list_to_csv(path: Path, list_out):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([[x] for x in list_out])


def _read_scene_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Scene list not found: {path}")
    scenes = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                # your current file is a 1-column CSV, but this handles both:
                # "scene" or "scene,..." formats
                scenes.append(s.split(",")[0].strip())
    return scenes


def batch_run_inference(config: dict, device: str | torch.device = "cuda", sample_type: str = "test"):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    logger = logging.getLogger(__name__)

    device = torch.device(device if (isinstance(device, str) and torch.cuda.is_available()) else device)

    dataset_cache_dir = Path(config["paths"]["dataset_cache_dir"])
    run_dir = Path(config["paths"]["run_dir"])

    # Where to read scenes from
    if sample_type == "val":
        data_dir = Path(config["preprocessing"]["input_dir"])
        scene_list_path = dataset_cache_dir / "lists" / "validation_scene_list.csv"
        all_scenes = _read_scene_list(scene_list_path)
    else:
        data_dir = Path(config["inference"]["input_dir"])
        all_scenes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    sample_size = int(config["inference"].get("sample_size", 0))
    if sample_size == 0 or sample_size > len(all_scenes):
        sample_scenes = all_scenes
        sample_size = len(all_scenes)
    else:
        sample_scenes = all_scenes[:sample_size]

    logger.info(f"Sampling {sample_size} out of {len(all_scenes)} scenes available in {data_dir}")

    # Save inferred scenes list into run outputs (not dataset cache)
    lists_out_dir = run_dir / "inference" / "lists"
    lists_out_dir.mkdir(parents=True, exist_ok=True)
    inferred_list_path = lists_out_dir / f"{sample_type}_scenes_inferred_list.csv"
    write_list_to_csv(inferred_list_path, sample_scenes)
    logger.info(f"Saved inferred scene list: {inferred_list_path}")

    # Load model once
    model = load_model_for_inference(config, device)

    # Run inference for each scene
    for scene_folder in sample_scenes:
        logger.info(f"🔍 Inference on scene: {scene_folder}")
        inference_one_scene(
            config=config,
            scene_folder=scene_folder,
            model=model,
            device=device,
            sample_type=sample_type,
        )

