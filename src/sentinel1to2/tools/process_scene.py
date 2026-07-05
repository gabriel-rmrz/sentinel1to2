from __future__ import annotations

import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Optional

from .load_and_stack_full import load_and_stack_full


def _select_indices(indices: np.ndarray, ind_names: list[str], selected: list[str]) -> np.ndarray:
    """
    indices: (N, H, W)
    returns: (K, H, W) where K=len(selected)
    """
    name_to_idx = {name: i for i, name in enumerate(ind_names)}
    missing = [name for name in selected if name not in name_to_idx]
    if missing:
        raise ValueError(f"Requested indices not found in computed indices: {missing}")

    idxs = [name_to_idx[name] for name in selected]
    out = indices[idxs, :, :]  # (K, H, W)
    # ensure 3D even for K=1
    if out.ndim == 2:
        out = out[None, :, :]
    return out


def _iter_patch_coords(H: int, W: int, patch: int, stride: int):
    """
    Generates top-left coords covering the full image.
    Includes the last patch to cover borders.
    """
    if H < patch or W < patch:
        return

    tops = list(range(0, H - patch + 1, stride))
    lefts = list(range(0, W - patch + 1, stride))

    # include last patch to cover borders
    if tops and tops[-1] != H - patch:
        tops.append(H - patch)
    if lefts and lefts[-1] != W - patch:
        lefts.append(W - patch)

    for top in tops:
        for left in lefts:
            yield top, left


def process_scene(
    config: dict,
    folder: str,
    data_dir: Path,
    hdf5_file: h5py.File,
    crop_size: Optional[int] = None,
    stride: Optional[int] = None,
    mean=None,
    std=None,
) -> Tuple[str, int]:
    """
    Creates input/target patches from one scene folder and stores them in HDF5.

    Uses load_and_stack_full(config, folder, data_dir, mean, std).

    Writes:
      group: scene_<folder>
        - inputs:  (N, C_in, ps, ps)
        - targets: (N, C_out, ps, ps)
        - attrs: target_type, patch_size, stride, n_patches
    """
    try:
        target_type = config["target"]["type"].lower()

        # defaults from config
        if crop_size is None:
            crop_size = int(config["preprocessing"]["patch_dimension"][0])
        if stride is None:
            stride = int(config["preprocessing"].get("stride", crop_size))

        # whether to skip NaN/Inf patches
        skip_nan = bool(config.get("preprocessing", {}).get("skip_nan_patches", True))

        # Load stacks (C, H, W)
        dsm, s1, worldcover, s2_selected, indices, ind_names, _profile = load_and_stack_full(
            config, folder, data_dir, mean, std
        )

        # Inputs always: DSM + S1 + WorldCover
        # Targets depend on target.type
        if target_type == "bands":
            target_arr = s2_selected
        elif target_type == "indices":
            selected = list(config["target"]["selected_indices"])
            target_arr = _select_indices(indices, ind_names, selected)
        else:
            raise ValueError(f"Unknown target.type: {target_type}")

        # Dimensions to patch over
        H, W = target_arr.shape[1], target_arr.shape[2]

        input_patches = []
        target_patches = []

        input_type = config["preprocessing"]["input_type"]
        for top, left in _iter_patch_coords(H, W, crop_size, stride):
            
            if input_type == "all":
                x = np.concatenate(
                    [
                        dsm[:, top : top + crop_size, left : left + crop_size],
                        s1[:, top : top + crop_size, left : left + crop_size],
                        worldcover[:, top : top + crop_size, left : left + crop_size],
                    ],
                    axis=0,
                )
            if input_type == "sar":
                x = s1[:, top : top + crop_size, left : left + crop_size]
            if input_type == "wc":
                x =  worldcover[:, top : top + crop_size, left : left + crop_size]

            y = target_arr[:, top : top + crop_size, left : left + crop_size]

            if skip_nan:
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    continue

            input_patches.append(x.astype(np.float32, copy=False))
            target_patches.append(y.astype(np.float32, copy=False))

        if len(input_patches) == 0:
            print(f"[WARN] No valid patches for {folder}, skipping scene.")
            return folder, 0

        grp_name = f"scene_{folder}"
        # safer than create_group (won't crash if group exists)
        if grp_name in hdf5_file:
            del hdf5_file[grp_name]
        grp = hdf5_file.create_group(grp_name)

        X = np.stack(input_patches, axis=0)   # (N, Cin, ps, ps)
        Y = np.stack(target_patches, axis=0)  # (N, Cout, ps, ps)

        ps = crop_size
        cin = X.shape[1]
        cout = Y.shape[1]
        #grp.create_dataset("inputs", data=X, dtype=np.float32, chunks=(1, cin, ps, ps), track_times=False)
        #grp.create_dataset("targets", data=Y, dtype=np.float32, chunks=(1, cout, ps, ps), track_times=False)
        grp.create_dataset("inputs", data=X, dtype=np.float32)
        grp.create_dataset("targets", data=Y, dtype=np.float32)

        # metadata
        grp.attrs["target_type"] = target_type
        grp.attrs["patch_size"] = int(crop_size)
        grp.attrs["stride"] = int(stride)
        grp.attrs["n_patches"] = int(X.shape[0])
        grp.attrs["cin"] = int(X.shape[1])
        grp.attrs["cout"] = int(Y.shape[1])

        return folder, int(X.shape[0])

    except Exception as e:
        print(f"Error processing {folder}: {str(e)}")
        return folder, 0

