# src/sentinel1to2/tools/config_utils.py

from __future__ import annotations
from pathlib import Path
import yaml
import hashlib
from datetime import datetime

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_merge(a: dict, b: dict) -> dict:
    """Merge b into a (recursively)."""
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def compute_dataset_id(cfg: dict) -> str:
    ds = cfg.get("dataset", {})
    tgt = cfg.get("target", {})
    prep = cfg.get("preprocessing", {})

    dataset_name = ds.get("name", "dataset")
    version = ds.get("version", 1)

    ttype = tgt.get("type", "bands")
    if ttype == "indices":
        items = "_".join(tgt.get("selected_indices", [])) or "indices"
    else:
        # store either indices list or selected bands list for uniqueness
        sel = tgt.get("selected_bands", [])
        items = "bands" if not sel else "b" + "_".join(map(str, sel))

    ps = prep.get("patch_dimension", [128, 128])[0]
    stride = prep.get("stride", 32)
    norm_scheme = "precalc" if not prep.get("do_norm_params", True) else "compute"

    return f"{slug(dataset_name)}__target-{ttype}-{slug(items)}__ps{ps}_s{stride}__norm-{norm_scheme}__v{version}"

def compute_run_name(cfg: dict, dataset_id: str) -> str:
    run = cfg.get("run", {})
    model = cfg.get("model", {})
    tr = cfg.get("training", {})
    tgt = cfg.get("target", {})

    seed = run.get("seed", 0)
    ttype = tgt.get("type", "bands")

    if ttype == "indices":
        tgt_items = "_".join(tgt.get("selected_indices", [])) or "indices"
    else:
        sel = tgt.get("selected_bands", [])
        tgt_items = "bands" if not sel else "b" + "_".join(map(str, sel))

    model_name = model.get("name", "model")
    enc = model.get("parameters", {}).get("encoder_name", "")
    model_id = f"{model_name}-{enc}" if enc else model_name

    loss_name = tr.get("loss", {}).get("name", "loss")
    gan_mode = tr.get("gan", {}).get("mode", "none")

    date = datetime.now().strftime("%Y-%m-%d")
    dataset_short = dataset_id.split("__")[0]  # e.g. puglia_s1_to_s2

    return slug(f"{date}__{dataset_short}__{model_id}__tgt-{ttype}-{tgt_items}__loss-{loss_name}__gan-{gan_mode}__seed{seed}")

def resolve_paths(cfg: dict) -> dict:
    """
    Adds:
      cfg['paths']['dataset_cache_dir']
      cfg['paths']['run_dir']
    """
    paths = cfg.setdefault("paths", {})
    dataset_cache_root = Path(paths.get("dataset_cache_root", "data/datasets"))
    runs_root = Path(paths.get("runs_root", "runs"))

    dataset_id = compute_dataset_id(cfg)
    paths["dataset_id"] = dataset_id
    paths["dataset_cache_dir"] = str(dataset_cache_root / dataset_id)

    run = cfg.setdefault("run", {})
    run_name = run.get("name") or compute_run_name(cfg, dataset_id)
    run["name"] = run_name
    paths["run_dir"] = str(runs_root / run_name)

    return cfg

def save_yaml(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

