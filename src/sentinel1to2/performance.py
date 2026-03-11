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


def performance(config: dict, sample_type: str = "test") -> None:
    run_dir, real_dir, pred_dir, scene_list_path, tables_dir, plots_dir = _get_dirs(config, sample_type)

    target_type = config["target"]["type"]  # "bands" or "indices"
    tile_type = "scenes"

    # metrics requested
    metric_names = list(config["performance"][f"{target_type}_metric_names"])
    sam_enabled = (target_type == "bands") and ("sam" in [m.lower() for m in metric_names])
    per_channel_metric_names = [m for m in metric_names if str(m).lower() != "sam"]

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
        prefix1 = f"{sample_type}_{tile_type}_{target_type}_gt_vs_comp"
        gt_vs_comp_path = tables_dir / f"{prefix1}.csv"

    prefix2 = f"{sample_type}_{tile_type}_{target_type}_gt_vs_inf"
    gt_vs_inf_path = tables_dir / f"{prefix2}.csv"

    sam_path = tables_dir / f"{prefix2}__sam.csv" if sam_enabled else None

    scenes_to_plot = int(config.get("evaluation", {}).get("scenes_to_plot", 0))
    scene_count = 0

    # open optional comp file
    if target_type == "bands":
        gt_vs_comp_file = open(gt_vs_comp_path, "w")
        write_metrics_header(gt_vs_comp_file, "indices", indices_metric_names)

    # open optional SAM file
    sam_file = None
    if sam_enabled and sam_path is not None:
        sam_file = open(sam_path, "w")
        write_sam_header(sam_file)

    with open(gt_vs_inf_path, "w") as gt_vs_inf_file:
        write_metrics_header(gt_vs_inf_file, target_type, per_channel_metric_names)

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
                    gt_vs_inf_file,
                    scene_name=dname,
                    gt=channels_gt,
                    pred=channels_inf,
                    channel_names=channel_names,
                    metric_names=per_channel_metric_names,
                )

                # -------------------------
                # Scene-level SAM (bands only)
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
                        gt_vs_comp_file,
                        scene_name=dname,
                        gt=ind_from_gt,
                        pred=ind_from_inf,
                        channel_names=ind_names_from_gt,
                        metric_names=indices_metric_names,
                    )

                # -------------------------
                # Plots
                # -------------------------
                if scene_count < scenes_to_plot:
                    base_dir = plots_dir / tile_type / target_type
                    base_dir.mkdir(parents=True, exist_ok=True)

                    if target_type == "bands":
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

    # -------------------------
    # Aggregate per-channel CSVs
    # -------------------------
    if target_type == "bands" and gt_vs_comp_path is not None and gt_vs_comp_path.exists():
        gt_vs_comp_df = pd.read_csv(gt_vs_comp_path)
        if not gt_vs_comp_df.empty:
            produce_outputs_from_df(gt_vs_comp_df, config, indices_metric_names, prefix1)
            plot_group_metric_histograms(
                output_dir=plots_dir / tile_type / "indices" / "metrics",
                df=gt_vs_comp_df,
                group_col="indices",
                metrics=indices_metric_names,
                prefix="gt_vs_comp",
            )

    if gt_vs_inf_path.exists():
        gt_vs_inf_df = pd.read_csv(gt_vs_inf_path)
        if not gt_vs_inf_df.empty:
            produce_outputs_from_df(gt_vs_inf_df, config, per_channel_metric_names, prefix2)
            plot_group_metric_histograms(
                output_dir=plots_dir / tile_type / target_type / "metrics",
                df=gt_vs_inf_df,
                group_col=target_type,
                metrics=per_channel_metric_names,
                prefix="gt_vs_inf",
            )

    # -------------------------
    # Plot SAM separately
    # -------------------------
    if sam_enabled and sam_path is not None and sam_path.exists():
        sam_plot_dir = plots_dir / tile_type / "sam"
        plot_sam_histogram(sam_path, sam_plot_dir, prefix=prefix2)
        plot_sam_per_scene(sam_path, sam_plot_dir, prefix=prefix2)

