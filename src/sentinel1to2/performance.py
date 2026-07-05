from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm

from .tools.produce_outputs_from_df import produce_outputs_from_df
from .tools.compute_vegetation_indices import compute_vegetation_indices
from .tools.load_image import load_image


from .tools.compute_metrics import (
    write_metrics_header,
    write_per_channel_metrics,
    write_sam_header,
    write_scene_sam,
    #compute_metrics,
    #compute_metrics_map,
    _fmt,
    spectral_angle_mapper_map,
    ergas,
)
from .plotting.plot_comparison_rgb_composites_2d import plot_comparison_rgb_composites_2d
from .plotting.plot_comparison_histos_2d import plot_comparison_histos_2d
from .plotting.plot_s2_composites_2d import plot_s2_composites_2d
from .plotting.plot_scatter_gt_vs_inf import plot_scatter_gt_vs_inf
from .plotting.plot_abs_error import plot_abs_error
from .plotting.plot_histo_2d import plot_histo_2d
from .plotting.plot_group_metric_histograms import plot_group_metric_histograms

from .plotting.plot_sam_metrics import plot_sam_histogram, plot_sam_per_scene


def read_csv_to_list(path: Path) -> List[str]:
    rows: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Scene list CSV not found: {path}")

    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row[0].strip())
    return rows


def _get_dirs(config: dict, sample_type: str):
    if "paths" not in config or "run_dir" not in config["paths"]:
        raise KeyError("config['paths']['run_dir'] not found. Resolve paths in __main__.py first.")

    run_dir = Path(config["paths"]["run_dir"])

    if sample_type == "val":
        real_dir = Path(config["preprocessing"]["input_dir"])
    else:
        real_dir = Path(config["inference"]["input_dir"])

    # predictions live under run_dir/inference/<sample_type>/
    pred_dir = run_dir / "inference" / sample_type

    scene_list_path = run_dir / "inference" / "lists" / f"{sample_type}_scenes_inferred_list.csv"

    tables_dir = run_dir / "metrics" / "tables"
    plots_dir = run_dir / "plots" / "scenes" / sample_type

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, real_dir, pred_dir, scene_list_path, tables_dir, plots_dir


def _get_channel_names(config: dict, target_type: str) -> List[str]:
    if target_type == "indices":
        return list(config["target"]["selected_indices"])
    all_bands = config["target"]["all_bands"]
    selected_bands = config["target"]["selected_bands"]
    return [all_bands[j] for j in selected_bands]

# [WC] -----------------------------------------------------------------------
# métricas con columna única en el CSV de clases
_SINGLE_COL_METRICS = {"sam", "ergas"}


def _write_class_metrics_header(f, channel_names: List[str], metric_names: List[str]) -> None:
    cols = ["scene", "cls"]
    for m in metric_names:
        if m.lower() in ("sam", "ergas"):   # ← añadir ergas
            cols.append(m.lower())
        else:
            for ch in channel_names:
                cols.append(f"{m}__{ch}")
    f.write(",".join(cols) + "\n")


def _write_class_metrics_row(
    f,
    scene_name: str,
    cls: int,
    gt: np.ndarray,          # (C, H, W)
    pred: np.ndarray,        # (C, H, W)
    mask: np.ndarray,        # (H, W) bool
    channel_names: List[str],
    metric_names: List[str],
    target_type: str,
    min_pixels: int = 10,
    ndigits: int = 4,
) -> bool:
    n_pixels = int(mask.sum())
    if n_pixels < min_pixels:
        return False

    C = gt.shape[0]
    vals: dict[str, list[str]] = {}

    for m in metric_names:
        ml = m.lower()

        if ml == "sam":
            sam_map = spectral_angle_mapper_map(gt, pred)        # (H, W)
            vals[m] = [_fmt(float(sam_map[mask].mean()), ndigits)]

        elif ml == "ergas":
            gt_masked   = gt[:, mask]                            # (C, N)
            pred_masked = pred[:, mask]                          # (C, N)
            vals[m] = [_fmt(float(ergas(gt_masked, pred_masked)), ndigits)]

        elif ml == "ssim":
            from skimage.metrics import structural_similarity
            channel_vals = []
            for c in range(C):
                _, ssim_map = structural_similarity(
                    gt[c], pred[c], data_range=2.0, full=True
                )
                channel_vals.append(_fmt(float(ssim_map[mask].mean()), ndigits))
            vals[m] = channel_vals

        elif ml == "mae":
            vals[m] = [
                _fmt(float(np.abs(gt[c] - pred[c])[mask].mean()), ndigits)
                for c in range(C)
            ]

        elif ml == "psnr":
            vals[m] = [
                _fmt(float((10.0 * np.log10(4.0 / ((gt[c] - pred[c])**2 + 1e-12)))[mask].mean()), ndigits)
                for c in range(C)
            ]

        elif ml == "r2":
            vals[m] = [
                _fmt(float((1.0 - (gt[c] - pred[c])**2 / ((gt[c] - gt[c].mean())**2 + 1e-12))[mask].mean()), ndigits)
                for c in range(C)
            ]

        else:
            raise ValueError(f"Unknown metric '{m}'")

    # build row
    row = [scene_name, str(cls)]
    for m in metric_names:
        if m.lower() in ("sam", "ergas"):
            row.append(vals[m][0])
        else:
            row.extend(vals[m])

    f.write(",".join(row) + "\n")
    return True

def performance(config: dict, sample_type: str = "test") -> None:
    run_dir, real_dir, pred_dir, scene_list_path, tables_dir, plots_dir = _get_dirs(config, sample_type)

    target_type = config["target"]["type"]  # "bands" or "indices"
    tile_type = "scenes"

    # metrics requested
    metric_names = list(config["performance"][f"{target_type}_metric_names"])
    #sam_enabled = (target_type == "bands") and ("sam" in [m.lower() for m in metric_names])
    sam_enabled = "sam" in [m.lower() for m in metric_names]
    per_channel_metric_names = [m for m in metric_names if str(m).lower() != "sam"]
    class_plot_metrics = [m for m in per_channel_metric_names if m.lower() not in _SINGLE_COL_METRICS]

    channel_names = _get_channel_names(config, target_type)

    selected_bands = config["target"].get("selected_bands", [])
    selected_indices = config["target"].get("selected_indices", [])

    list_of_scenes = read_csv_to_list(scene_list_path)

    # Optional indices-from-bands comparison
    gt_vs_comp_file = None
    gt_vs_comp_path = None
    prefix1 = None
    indices_metric_names = None

    if target_type == "bands":
        indices_metric_names = list(config["performance"]["indices_metric_names"])
        indices_metric_names = [m for m in indices_metric_names if str(m).lower() != "sam"]
        prefix1 = f"{sample_type}_{tile_type}_{target_type}_gt_vs_comp"
        gt_vs_comp_path = tables_dir / f"{prefix1}.csv"

    prefix2 = f"{sample_type}_{tile_type}_{target_type}_gt_vs_inf"
    if target_type == "bands":
        prefix2_ind = f"{sample_type}_{tile_type}_computed_ind_gt_vs_inf"
    gt_vs_inf_path = tables_dir / f"{prefix2}.csv"

    sam_path = tables_dir / f"{prefix2}__sam.csv" if sam_enabled else None
    # [WC] ── configuración ───────────────────────────────────────────────────
    wc_enabled  = config.get("performance", {}).get("wc_class_metrics", False)
    wc_min_pix  = int(config.get("performance", {}).get("wc_min_pixels", 10))
    prefix_wc   = f"{prefix2}__per_wc_class"
    wc_out_path = tables_dir / f"{prefix_wc}.csv"
    wc_file     = None
    # [CC] ── configuración ───────────────────────────────────────────────────
    cc_enabled   = config.get("performance", {}).get("cc_class_metrics", False)
    cc_masks_dir = Path(config.get("performance", {}).get("cc_masks_dir", ""))
    cc_min_pix   = int(config.get("performance", {}).get("cc_min_pixels", 10))
    prefix_cc    = f"{prefix2}__per_cc_class"
    cc_out_path  = tables_dir / f"{prefix_cc}.csv"
    cc_file      = None
    # ─────────────────────────────────────────────────────────────────────────
    # [WC indices] ── configuración ───────────────────────────────────────────
    prefix_wc_ind   = f"{prefix2}__per_wc_class__indices"
    wc_ind_out_path = tables_dir / f"{prefix_wc_ind}.csv"
    wc_ind_file     = None
    # [CC indices] ── configuración ───────────────────────────────────────────
    prefix_cc_ind   = f"{prefix2}__per_cc_class__indices"
    cc_ind_out_path = tables_dir / f"{prefix_cc_ind}.csv"
    cc_ind_file     = None

    scenes_to_plot = int(config.get("evaluation", {}).get("scenes_to_plot", 0))
    scene_count = 0


    # [WC] ── abrir CSV por clase ─────────────────────────────────────────────
    if wc_enabled:
        wc_file = open(wc_out_path, "w")
        _write_class_metrics_header(wc_file, channel_names, per_channel_metric_names)
    # [CC] ── abrir CSV por clase ─────────────────────────────────────────────
    if cc_enabled:
        cc_file = open(cc_out_path, "w")
        _write_class_metrics_header(cc_file, channel_names, per_channel_metric_names)
    # ─────────────────────────────────────────────────────────────────────────
    sam_ind_path = None
    if target_type == "bands":
        sam_ind_path = tables_dir / f"{prefix2_ind}__sam.csv" if sam_enabled else None
        gt_vs_comp_file = open(gt_vs_comp_path, "w")
        write_metrics_header(gt_vs_comp_file, "indices", metric_names)

    # open optional SAM file
    sam_file = None
    sam_ind_file = None
    if sam_enabled  and sam_path is not None:
        sam_file = open(sam_path, "w")
        write_sam_header(sam_file)
    if sam_enabled  and (sam_ind_path is not None) and (target_type == "bands"):
        sam_ind_file = open(sam_ind_path, "w")
        write_sam_header(sam_ind_file)

    ind_channel_names_global = None   # will be set on first scene
    with open(gt_vs_inf_path, "w") as gt_vs_inf_file:
        write_metrics_header(gt_vs_inf_file, target_type, metric_names)

        for dname in tqdm(list_of_scenes, desc=f"Performance on {sample_type} sample"):
            try:
                scene, day = dname.split("_", 1)
            except ValueError:
                print(f"[SKIP] '{dname}' does not match 'scene_day' pattern")
                continue

            gt_path = real_dir / dname / f"{day}_s2.tif"
            pred_path = pred_dir / f"{sample_type}__{dname}__pred.tif"

            missing = [str(p) for p in (gt_path, pred_path) if not p.is_file()]
            if missing:
                print(f"[SKIP] Missing files for scene {dname}: {missing}")
                continue

            try:
                # -------------------------
                # Load GT / INF
                # -------------------------
                if target_type == "bands":
                    channels_gt = load_image(str(gt_path), selected_bands)/10000.0
                else:
                    s2_gt = load_image(str(gt_path), selected_bands)/10000.0 
                    ind_from_gt, ind_names_from_gt = compute_vegetation_indices(config, s2_gt)
                    sel_idx = [ind_names_from_gt.index(ind) for ind in selected_indices]
                    channels_gt = np.array([ind_from_gt[i] for i in sel_idx])

                channels_inf = load_image(str(pred_path))

                # -------------------------
                # Per-channel metrics (no SAM)
                # -------------------------
                write_per_channel_metrics(
                    target_type,
                    gt_vs_inf_file,
                    scene_name=dname,
                    gt=channels_gt,
                    pred=channels_inf,
                    channel_names=channel_names,
                    metric_names=metric_names,
                )

                # -------------------------
                # Scene-level SAM 
                # -------------------------
                if sam_enabled and sam_file is not None:
                    write_scene_sam(sam_file, scene_name=dname, gt=channels_gt, pred=channels_inf)

                # -------------------------
                # Optional: indices from bands
                # -------------------------
                if target_type == "bands" and gt_vs_comp_file is not None:
                    ind_from_gt, ind_names_from_gt = compute_vegetation_indices(config, channels_gt)
                    ind_from_inf, ind_names_from_inf = compute_vegetation_indices(config, channels_inf)
                    write_per_channel_metrics(
                        "indices",
                        gt_vs_comp_file,
                        scene_name=dname,
                        gt=ind_from_gt,
                        pred=ind_from_inf,
                        channel_names=ind_names_from_gt,
                        metric_names=metric_names,
                    )
                    # -------------------------
                    # Scene-level SAM 
                    # -------------------------
                    write_scene_sam(sam_ind_file, scene_name=dname, gt=ind_from_gt, pred=ind_from_inf)

                    # ── indices class metrics (WC and CC) ────────────────────
                    ind_from_gt_arr = np.array(ind_from_gt)   # (C_ind, H, W)

                    if ind_channel_names_global is None:
                        ind_channel_names_global = ind_names_from_gt
                        if wc_enabled:
                            wc_ind_file = open(wc_ind_out_path, "w")
                            _write_class_metrics_header(wc_ind_file, ind_channel_names_global, per_channel_metric_names)
                        if cc_enabled:
                            cc_ind_file = open(cc_ind_out_path, "w")
                            _write_class_metrics_header(cc_ind_file, ind_channel_names_global, per_channel_metric_names)

                    if wc_enabled and wc_ind_file is not None:
                        wc_scene_path = real_dir / dname / f"{day}_worldcover.tif"
                        if wc_scene_path.is_file():
                            wc_mask = load_image(str(wc_scene_path))
                            if wc_mask.ndim == 3:
                                wc_mask = wc_mask[0]
                            wc_mask = wc_mask.astype(int)
                            for cls in np.unique(wc_mask):
                                _write_class_metrics_row(
                                    f=wc_ind_file,
                                    scene_name=dname,
                                    cls=int(cls),
                                    gt=ind_from_gt_arr,
                                    pred=np.array(ind_from_inf),
                                    mask=(wc_mask == cls),
                                    target_type="indices",
                                    channel_names=ind_channel_names_global,
                                    metric_names=per_channel_metric_names,
                                    min_pixels=wc_min_pix,
                                )

                    if cc_enabled and cc_ind_file is not None:
                        cc_scene_path = cc_masks_dir / dname / f"{day}_mask.tif"
                        if cc_scene_path.is_file():
                            cc_mask = load_image(str(cc_scene_path))
                            if cc_mask.ndim == 3:
                                cc_mask = cc_mask[0]
                            cc_mask = cc_mask.astype(int)
                            for cls in np.unique(cc_mask):
                                _write_class_metrics_row(
                                    f=cc_ind_file,
                                    scene_name=dname,
                                    cls=int(cls),
                                    gt=ind_from_gt_arr,
                                    pred=np.array(ind_from_inf),
                                    mask=(cc_mask == cls),
                                    target_type="indices",
                                    channel_names=ind_channel_names_global,
                                    metric_names=per_channel_metric_names,
                                    min_pixels=cc_min_pix,
                                )


                # [WC] ── métricas por clase de world cover ───────────────────
                if wc_enabled and wc_file is not None:
                    wc_scene_path = real_dir / dname / f"{day}_worldcover.tif"
                    if not wc_scene_path.is_file():
                        print(f"[WC SKIP] Missing WC mask for scene {dname}")
                    else:
                        wc_mask = load_image(str(wc_scene_path))
                        if wc_mask.ndim == 3:
                            wc_mask = wc_mask[0]
                        wc_mask = wc_mask.astype(int)

                        for cls in np.unique(wc_mask):
                            _write_class_metrics_row(
                                f=wc_file,
                                scene_name=dname,
                                cls=int(cls),
                                gt=channels_gt,
                                pred=channels_inf,
                                mask=(wc_mask == cls),
                                target_type=target_type,
                                channel_names=channel_names,
                                metric_names=per_channel_metric_names,
                                min_pixels=wc_min_pix,
                            )
                # [CC] ── métricas por clase de climate classification ─────────
                if cc_enabled and cc_file is not None:
                    cc_scene_path = cc_masks_dir / dname / f"{day}_mask.tif"
                    if not cc_scene_path.is_file():
                        print(f"[CC SKIP] Missing CC mask for scene {dname}")
                    else:
                        cc_mask = load_image(str(cc_scene_path))
                        if cc_mask.ndim == 3:
                            cc_mask = cc_mask[0]
                        cc_mask = cc_mask.astype(int)

                        for cls in np.unique(cc_mask):
                            _write_class_metrics_row(
                                f=cc_file,
                                scene_name=dname,
                                cls=int(cls),
                                gt=channels_gt,
                                pred=channels_inf,
                                mask=(cc_mask == cls),
                                target_type=target_type,
                                channel_names=channel_names,
                                metric_names=per_channel_metric_names,
                                min_pixels=cc_min_pix,
                            )
                # ─────────────────────────────────────────────────────────────

                # -------------------------
                # Plots
                # -------------------------
                if scene_count < scenes_to_plot:
                    base_dir = plots_dir / tile_type / target_type
                    base_dir.mkdir(parents=True, exist_ok=True)

                    if target_type == "bands" and "ind_from_gt" in dir():
                        idx_dir = plots_dir / tile_type / "indices"
                        idx_dir.mkdir(parents=True, exist_ok=True)

                        plot_histo_2d(idx_dir / "histos2d", ind_from_gt, ind_names_from_gt, dname, prefix="computed_from_gt")
                        plot_histo_2d(idx_dir / "histos2d", ind_from_inf, ind_names_from_gt, dname, prefix="computed_from_inf")
                        plot_comparison_histos_2d(
                            idx_dir / "histos2d_comparison",
                            ind_from_gt,
                            ind_from_inf,
                            ind_names_from_gt,
                            dname,
                            prefix="computed_from_gt_inf",
                        )
                        plot_scatter_gt_vs_inf(
                            idx_dir / "scatter_gt_vs_inf",
                            ind_from_gt,
                            ind_from_inf,
                            ind_names_from_gt,
                            dname,
                            prefix="computed_from_gt_vs_inf",
                        )
                        plot_abs_error(
                            idx_dir / "histos_abs_error",
                            ind_from_gt,
                            ind_from_inf,
                            ind_names_from_gt,
                            dname,
                            prefix="computed_from_gt_vs_inf",
                        )

                    plot_comparison_rgb_composites_2d(
                        base_dir / "rgb_comparison",
                        channels_gt,
                        channels_inf,
                        channel_names,
                        dname,
                        prefix="gt_inf",
                    )
                    plot_s2_composites_2d(
                        base_dir / "composites",
                        channels_gt,
                        channel_names,
                        dname,
                        prefix="gt",
                    )
                    plot_s2_composites_2d(
                        base_dir / "composites",
                        channels_inf,
                        channel_names,
                        dname,
                        prefix="inf",
                    )
                    plot_histo_2d(base_dir / "histos2d", channels_gt, channel_names, dname, prefix="gt")
                    plot_histo_2d(base_dir / "histos2d", channels_inf, channel_names, dname, prefix="inf")
                    plot_comparison_histos_2d(
                        base_dir / "histos2d_comparison",
                        channels_gt,
                        channels_inf,
                        channel_names,
                        dname,
                        prefix="gt_inf",
                    )
                    plot_scatter_gt_vs_inf(
                        base_dir / "scatter_gt_vs_inf",
                        channels_gt,
                        channels_inf,
                        channel_names,
                        dname,
                        prefix="gt_vs_inf",
                    )
                    plot_abs_error(
                        base_dir / "abs_error",
                        channels_gt,
                        channels_inf,
                        channel_names,
                        dname,
                        prefix="gt_vs_inf",
                    )

                    scene_count += 1

            except Exception as e:
                print(f"[ERROR] Error for the scene {dname}: {e}")

    if gt_vs_comp_file is not None:
        gt_vs_comp_file.close()
    if sam_file is not None:
        sam_file.close()
    if (target_type == "bands") and sam_ind_file is not None:
        sam_ind_file.close()

    if wc_file is not None:
        wc_file.close()
    if cc_file is not None:
        cc_file.close()

    if wc_ind_file is not None:         
        wc_ind_file.close()
    if cc_ind_file is not None:
        cc_ind_file.close()

    # -------------------------
    # Aggregate per-channel CSVs
    # -------------------------

    if target_type == "bands" and gt_vs_comp_path is not None and gt_vs_comp_path.exists():
        gt_vs_comp_df = pd.read_csv(gt_vs_comp_path)
        if not gt_vs_comp_df.empty:
            gt_vs_comp_df_bands = gt_vs_comp_df[gt_vs_comp_df["indices"] != "all_bands"]  # ← filter
            produce_outputs_from_df(gt_vs_comp_df_bands, config, indices_metric_names, prefix1)
            plot_group_metric_histograms(
                output_dir=plots_dir / tile_type / "indices" / "metrics",
                df=gt_vs_comp_df_bands,
                group_col="indices",
                metrics=indices_metric_names,
                prefix="gt_vs_comp",
            )

    if gt_vs_inf_path.exists():
        gt_vs_inf_df = pd.read_csv(gt_vs_inf_path)
        if not gt_vs_inf_df.empty:
            gt_vs_inf_df_bands = gt_vs_inf_df[gt_vs_inf_df[target_type] != "all_bands"]
            produce_outputs_from_df(gt_vs_inf_df_bands, config, class_plot_metrics, prefix2)  # ← class_plot_metrics excludes ergas
            plot_group_metric_histograms(
                output_dir=plots_dir / tile_type / target_type / "metrics",
                df=gt_vs_inf_df_bands,
                group_col=target_type,
                metrics=class_plot_metrics,   # ← same
                prefix="gt_vs_inf",
            )
    # -------------------------
    # Plot SAM separately
    # -------------------------
    if (target_type == "bands") and sam_enabled and sam_ind_path is not None and sam_ind_path.exists():
        sam_plot_dir = plots_dir / tile_type / "sam"
        plot_sam_histogram(sam_ind_path, sam_plot_dir, prefix=prefix2_ind)
        plot_sam_per_scene(sam_ind_path, sam_plot_dir, prefix=prefix2_ind)

    if target_type == "bands" and gt_vs_comp_path is not None and gt_vs_comp_path.exists():
        gt_vs_comp_df = pd.read_csv(gt_vs_comp_path)
        if not gt_vs_comp_df.empty:
            gt_vs_comp_df_bands = gt_vs_comp_df[gt_vs_comp_df["indices"] != "all_bands"]
            indices_plot_metrics = [m for m in indices_metric_names if m.lower() not in _SINGLE_COL_METRICS]  # ← filter
            produce_outputs_from_df(gt_vs_comp_df_bands, config, indices_plot_metrics, prefix1)
            plot_group_metric_histograms(
                output_dir=plots_dir / tile_type / "indices" / "metrics",
                df=gt_vs_comp_df_bands,
                group_col="indices",
                metrics=indices_plot_metrics,
                prefix="gt_vs_comp",
            )
