from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .tools.compute_vegetation_indices import compute_vegetation_indices
from .tools.produce_outputs_from_df import produce_outputs_from_df

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


def _get_output_dirs(config: dict):
    if "paths" not in config or "run_dir" not in config["paths"]:
        raise KeyError("config['paths']['run_dir'] not found. Resolve paths in __main__.py first.")
    run_dir = Path(config["paths"]["run_dir"])
    tables_dir = run_dir / "metrics" / "tables"
    plots_dir = run_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, tables_dir, plots_dir


def _get_channel_names(config: dict, target_type: str) -> list[str]:
    if target_type == "indices":
        return list(config["target"]["selected_indices"])
    all_bands = config["target"]["all_bands"]
    selected_bands = config["target"]["selected_bands"]
    return [all_bands[j] for j in selected_bands]


def evaluate_model(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    val_loader,
    num_samples: int = 5,
    split: str = "val",
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    logger = logging.getLogger(__name__)

    _run_dir, tables_dir, plots_dir = _get_output_dirs(config)

    target_type = config["target"]["type"]

    # metric names
    bands_metric_names = list(config["performance"]["bands_metric_names"])
    indices_metric_names = list(config["performance"]["indices_metric_names"])

    if target_type == "bands":
        metric_names = bands_metric_names
        channel_names = _get_channel_names(config, "bands")
    elif target_type == "indices":
        metric_names = indices_metric_names
        channel_names = _get_channel_names(config, "indices")
    else:
        raise ValueError(f"Unknown target.type: {target_type}")

    # SAM: separate CSV only for bands
    sam_enabled = (target_type == "bands") and ("sam" in [m.lower() for m in metric_names])
    per_channel_metric_names = [m for m in metric_names if str(m).lower() != "sam"]

    model.eval()

    # tables
    prefix_inf = f"{split}_patches_{target_type}_gt_vs_inf"
    gt_vs_inf_path = tables_dir / f"{prefix_inf}.csv"

    sam_path = tables_dir / f"{prefix_inf}__sam.csv" if sam_enabled else None

    gt_vs_comp_path = None
    prefix_comp = None
    if target_type == "bands":
        prefix_comp = f"{split}_patches_indices_gt_vs_comp"
        gt_vs_comp_path = tables_dir / f"{prefix_comp}.csv"

    scenes_to_plot = int(config.get("evaluation", {}).get("scenes_to_plot", 0))

    # open optional comp file
    gt_vs_comp_file = None
    if target_type == "bands":
        gt_vs_comp_file = open(gt_vs_comp_path, "w")
        write_metrics_header(gt_vs_comp_file, "indices", indices_metric_names)

    # open SAM file (optional)
    sam_file = None
    if sam_enabled and sam_path is not None:
        sam_file = open(sam_path, "w")
        write_sam_header(sam_file)

    scene_count = 0
    processed_patches = 0

    with open(gt_vs_inf_path, "w") as gt_vs_inf_file:
        write_metrics_header(gt_vs_inf_file, target_type, per_channel_metric_names)

        with torch.no_grad():
            for inputs, targets, scenes, patch_idx in tqdm(val_loader, desc=f"Evaluating {split} batches"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)

                batch_n = inputs.size(0)
                for j in range(batch_n):
                    if processed_patches >= num_samples:
                        break

                    target_patch = targets[j].detach().cpu().squeeze().numpy()
                    output_patch = outputs[j].detach().cpu().squeeze().numpy()

                    # per-channel metrics (no SAM here)
                    write_per_channel_metrics(
                        gt_vs_inf_file,
                        scene_name=scenes[j],
                        gt=target_patch,
                        pred=output_patch,
                        channel_names=channel_names,
                        metric_names=per_channel_metric_names,
                    )

                    # SAM per patch (scene-name is patch-scene id); still useful for patch eval
                    if sam_enabled and sam_file is not None:
                        write_scene_sam(sam_file, scene_name=scenes[j], gt=target_patch, pred=output_patch)

                    # bands -> indices computed from gt vs from inf
                    if target_type == "bands" and gt_vs_comp_file is not None:
                        ind_from_gt, ind_names_from_gt = compute_vegetation_indices(config, target_patch)
                        ind_from_inf, ind_names_from_inf = compute_vegetation_indices(config, output_patch)

                        write_per_channel_metrics(
                            gt_vs_comp_file,
                            scene_name=scenes[j],
                            gt=ind_from_gt,
                            pred=ind_from_inf,
                            channel_names=ind_names_from_gt,
                            metric_names=indices_metric_names,
                        )

                    # plots
                    if scene_count < scenes_to_plot:
                        base_plot_dir = plots_dir / "patches" / split / target_type
                        if target_type == "bands":
                            idx_plot_dir = plots_dir / "patches" / split / "indices"

                            plot_comparison_histos_2d(
                                idx_plot_dir / "histos2d_comparison",
                                ind_from_gt,
                                ind_from_inf,
                                ind_names_from_gt,
                                scenes[j],
                                prefix=f"{scene_count}_computed_from_gt_inf",
                            )
                            plot_scatter_gt_vs_inf(
                                idx_plot_dir / "scatter_gt_vs_inf",
                                ind_from_gt,
                                ind_from_inf,
                                ind_names_from_gt,
                                scenes[j],
                                prefix=f"{scene_count}_computed_from_gt_vs_inf",
                            )
                            plot_abs_error(
                                idx_plot_dir / "histos_abs_error",
                                ind_from_gt,
                                ind_from_inf,
                                ind_names_from_gt,
                                scenes[j],
                                prefix=f"{scene_count}_computed_from_gt_vs_inf",
                            )
                            plot_histo_2d(
                                idx_plot_dir / "histos2d",
                                ind_from_gt,
                                ind_names_from_gt,
                                scenes[j],
                                prefix=f"{scene_count}_computed_from_gt",
                            )
                            plot_histo_2d(
                                idx_plot_dir / "histos2d",
                                ind_from_inf,
                                ind_names_from_inf,
                                scenes[j],
                                prefix=f"{scene_count}_computed_from_inf",
                            )

                        plot_scatter_gt_vs_inf(
                            base_plot_dir / "scatter_gt_vs_inf",
                            target_patch,
                            output_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_gt_vs_inf",
                        )
                        plot_comparison_histos_2d(
                            base_plot_dir / "histos2d_comparison",
                            target_patch,
                            output_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_gt_inf",
                        )
                        plot_s2_composites_2d(
                            base_plot_dir / "composites",
                            target_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_gt",
                        )
                        plot_s2_composites_2d(
                            base_plot_dir / "composites",
                            output_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_inf",
                        )
                        plot_histo_2d(
                            base_plot_dir / "histos2d",
                            target_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_gt",
                        )
                        plot_histo_2d(
                            base_plot_dir / "histos2d",
                            output_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_inf",
                        )
                        plot_comparison_rgb_composites_2d(
                            base_plot_dir / "rgb_comparison",
                            target_patch,
                            output_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_gt_inf",
                        )
                        plot_abs_error(
                            base_plot_dir / "histos_abs_error",
                            target_patch,
                            output_patch,
                            channel_names,
                            scenes[j],
                            prefix=f"{scene_count}_gt_vs_inf",
                        )

                        scene_count += 1

                    processed_patches += 1

                if processed_patches >= num_samples:
                    break

    if gt_vs_comp_file is not None:
        gt_vs_comp_file.close()
    if sam_file is not None:
        sam_file.close()

    # postprocess tables
    if target_type == "bands" and gt_vs_comp_path is not None:
        gt_vs_comp_df = pd.read_csv(gt_vs_comp_path)
        if not gt_vs_comp_df.empty:
            produce_outputs_from_df(gt_vs_comp_df, config, indices_metric_names, prefix=gt_vs_comp_path.stem)
            plot_group_metric_histograms(
                output_dir=plots_dir / "patches" / split / "indices" / "metrics",
                df=gt_vs_comp_df,
                group_col="indices",
                metrics=indices_metric_names,
                prefix="gt_vs_comp",
            )

    gt_vs_inf_df = pd.read_csv(gt_vs_inf_path)
    if not gt_vs_inf_df.empty:
        produce_outputs_from_df(gt_vs_inf_df, config, per_channel_metric_names, prefix=gt_vs_inf_path.stem)
        plot_group_metric_histograms(
            output_dir=plots_dir / "patches" / split / target_type / "metrics",
            df=gt_vs_inf_df,
            group_col=target_type,
            metrics=per_channel_metric_names,
            prefix="gt_vs_inf",
        )

    # SAM plots (separate)
    if sam_enabled and sam_path is not None and Path(sam_path).exists():
        sam_plot_dir = plots_dir / "patches" / split / "sam"
        plot_sam_histogram(Path(sam_path), sam_plot_dir, prefix=gt_vs_inf_path.stem)
        plot_sam_per_scene(Path(sam_path), sam_plot_dir, prefix=gt_vs_inf_path.stem)

    logger.info("Evaluation completed successfully.")

