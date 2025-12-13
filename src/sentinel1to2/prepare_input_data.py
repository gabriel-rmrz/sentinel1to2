import logging
import csv
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
import h5py

from sklearn.model_selection import train_test_split

from .tools.process_scene import process_scene
from .tools.compute_hdf5_mean_std import compute_hdf5_mean_std
from .tools.config_utils import save_yaml


def write_list_to_csv(path: Path, list_out):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([[x] for x in list_out])


def _maybe_prepare_cache_dirs(dataset_cache_dir: Path, policy: str):
    """
    policy:
      - "reuse": keep existing files if they exist (skip creation if present)
      - "overwrite": delete and recreate the dataset cache dir
      - "error": fail if directory exists and is non-empty
    """
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    # If dir exists, decide what to do
    has_any = any(dataset_cache_dir.iterdir())
    if has_any:
        if policy == "reuse":
            return
        if policy == "overwrite":
            shutil.rmtree(dataset_cache_dir)
            dataset_cache_dir.mkdir(parents=True, exist_ok=True)
            return
        if policy == "error":
            raise FileExistsError(
                f"Dataset cache dir already exists and is not empty: {dataset_cache_dir}"
            )

    # Create expected subdirs
    (dataset_cache_dir / "h5").mkdir(parents=True, exist_ok=True)
    (dataset_cache_dir / "lists").mkdir(parents=True, exist_ok=True)
    (dataset_cache_dir / "norm").mkdir(parents=True, exist_ok=True)


def prepare_input_data(config: dict):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Preparation of the input samples")

    # -----------------------------
    # Paths: dataset cache (reusable)
    # -----------------------------
    if "paths" not in config or "dataset_cache_dir" not in config["paths"]:
        raise KeyError(
            "config['paths']['dataset_cache_dir'] not found. "
            "Make sure __main__.py resolves paths via resolve_paths(config)."
        )

    dataset_cache_dir = Path(config["paths"]["dataset_cache_dir"])
    cache_h5_dir = dataset_cache_dir / "h5"
    cache_lists_dir = dataset_cache_dir / "lists"
    cache_norm_dir = dataset_cache_dir / "norm"

    # Policy for cache usage (Condor-safe, no prompts)
    prep_cfg = config.get("preprocessing", {})
    policy = prep_cfg.get("cache_policy", "reuse")  # reuse|overwrite|error
    _maybe_prepare_cache_dirs(dataset_cache_dir, policy)

    # -----------------------------
    # Input data directory
    # -----------------------------
    data_dir = Path(prep_cfg["input_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"preprocessing.input_dir not found: {data_dir}")

    # -----------------------------
    # Output file paths in cache
    # -----------------------------
    train_hdf5_path = cache_h5_dir / config["training"]["data"]["train_dataset"]
    val_hdf5_path = cache_h5_dir / config["training"]["data"]["val_dataset"]

    # Norm params go into norm/
    norm_params_filename = prep_cfg.get("norm_params_file", "normalization_params.npz")
    norm_params_out_path = cache_norm_dir / norm_params_filename

    # -----------------------------
    # Scene sampling
    # -----------------------------
    all_scenes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if len(all_scenes) == 0:
        raise RuntimeError(f"No scene folders found under: {data_dir}")

    sample_size = int(prep_cfg.get("sample_size", 0))
    if sample_size == 0 or sample_size > len(all_scenes):
        sample_scenes = all_scenes
        sample_size = len(all_scenes)
    else:
        sample_scenes = all_scenes[:sample_size]

    logger.info(f"Sampling {sample_size} out of {len(all_scenes)} scenes available in {data_dir}")

    # -----------------------------
    # Train/val split (deterministic)
    # -----------------------------
    split_cfg = prep_cfg.get("split", {})
    test_size = float(split_cfg.get("val_fraction", 0.2))
    seed = int(config.get("run", {}).get("seed", 42))

    train_folders, val_folders = train_test_split(
        sample_scenes, test_size=test_size, random_state=seed
    )

    logger.info("Saving scene lists")
    write_list_to_csv(cache_lists_dir / "training_scene_list.csv", train_folders)
    write_list_to_csv(cache_lists_dir / "validation_scene_list.csv", val_folders)

    # -----------------------------
    # Normalization parameters
    # -----------------------------
    do_norm_params = bool(prep_cfg.get("do_norm_params", True))

    # If policy == reuse and everything exists, we can skip recomputation
    # (optional but very useful for Condor)
    already_have_all = (
        train_hdf5_path.exists()
        and val_hdf5_path.exists()
        and norm_params_out_path.exists()
        and (cache_lists_dir / "training_scene_list.csv").exists()
        and (cache_lists_dir / "validation_scene_list.csv").exists()
    )
    if policy == "reuse" and already_have_all:
        logger.info("Dataset cache already exists. Reusing cached preprocessing artifacts.")
        return

    if do_norm_params:
        # 1) Build a temporary HDF5 from training scenes (unnormalized)
        tmp_train_path = cache_h5_dir / (train_hdf5_path.stem + "_tmp.h5")

        logger.info(f"Producing temporary training dataset for mean/std: {tmp_train_path}")
        with h5py.File(tmp_train_path, "w") as hf:
            metadata_grp = hf.create_group("metadata")
            metadata_grp.create_dataset("scene_list", data=np.array(train_folders, dtype="S"))
            for folder in tqdm(train_folders, desc="Processing training scenes (tmp)"):
                process_scene(config, folder, data_dir, hf)

        # 2) Compute mean/std
        mean, std = compute_hdf5_mean_std(tmp_train_path)
        logger.info(f"Saving normalization parameters into {norm_params_out_path}")
        np.savez(norm_params_out_path, mean=mean, std=std)

        # 3) Remove tmp file (optional)
        try:
            tmp_train_path.unlink()
        except Exception:
            logger.warning(f"Could not remove tmp file: {tmp_train_path}")

    else:
        # Use precalculated normalization parameters
        precalc_path = Path(prep_cfg["precalc_norm_params_path"])
        if not precalc_path.exists():
            raise FileNotFoundError(f"preprocessing.precalc_norm_params_path not found: {precalc_path}")

        params = np.load(precalc_path)
        mean = params["mean"]
        std = params["std"]

        logger.info(f"Saving normalization parameters into {norm_params_out_path}")
        np.savez(norm_params_out_path, mean=mean, std=std)

    # -----------------------------
    # Create final normalized training HDF5
    # -----------------------------
    logger.info(f"Producing training dataset: {train_hdf5_path}")
    with h5py.File(train_hdf5_path, "w") as hf:
        metadata_grp = hf.create_group("metadata")
        metadata_grp.create_dataset("scene_list", data=np.array(train_folders, dtype="S"))
        for folder in tqdm(train_folders, desc="Writing normalized training scenes"):
            process_scene(config, folder, data_dir, hf, mean=mean, std=std)

    # -----------------------------
    # Create validation HDF5 (normalized with same mean/std)
    # -----------------------------
    logger.info(f"Producing validation dataset: {val_hdf5_path}")
    with h5py.File(val_hdf5_path, "w") as hf:
        metadata_grp = hf.create_group("metadata")
        metadata_grp.create_dataset("scene_list", data=np.array(val_folders, dtype="S"))
        for folder in tqdm(val_folders, desc="Writing normalized validation scenes"):
            process_scene(config, folder, data_dir, hf, mean=mean, std=std)

    # -----------------------------
    # Write dataset metadata (reproducibility)
    # -----------------------------
    meta = {
        "dataset_id": config["paths"].get("dataset_id"),
        "dataset_cache_dir": str(dataset_cache_dir),
        "raw_input_dir": str(data_dir),
        "target": config.get("target", {}),
        "preprocessing": config.get("preprocessing", {}),
        "training_data_files": {
            "train_h5": str(train_hdf5_path),
            "val_h5": str(val_hdf5_path),
        },
        "lists": {
            "train_list": str(cache_lists_dir / "training_scene_list.csv"),
            "val_list": str(cache_lists_dir / "validation_scene_list.csv"),
        },
        "norm_params": str(norm_params_out_path),
    }
    save_yaml(meta, dataset_cache_dir / "meta.yaml")

    logger.info("Preparation of the samples performed successfully.")

