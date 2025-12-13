from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .tools.compute_metrics import compute_all_metrics
from .tools.compute_vegetation_indices import compute_vegetation_indices
from .tools.produce_outputs_from_df import produce_outputs_from_df

from .plotting.plot_comparison_rgb_composites_2d import plot_comparison_rgb_composites_2d
from .plotting.plot_comparison_histos_2d import plot_comparison_histos_2d
from .plotting.plot_s2_composites_2d import plot_s2_composites_2d
from .plotting.plot_scatter_gt_vs_inf import plot_scatter_gt_vs_inf
from .plotting.plot_abs_error import plot_abs_error
from .plotting.plot_histo_2d import plot_histo_2d
from .plotting.plot_group_metric_histograms import plot_group_metric_histograms


def _get_output_dirs(config: dict):
    """
    All evaluation outputs go under run_dir:
      run_dir/metrics/tables/
      run_dir/plots/
    """
    if "paths" not in config or "run_dir" not in config["paths"]:
        raise KeyError(
            "config['paths']['run_dir'] not found. Make sure __main__.py resolves paths via resolve_paths(config)."
        )
    run_dir = Path(config["paths"]["run_dir"])
    tables_dir = run_dir / "metrics" / "tables"
    plots_dir = run_dir / "plots"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, tables_dir, plots_dir


def _get_channel_names(config: dict, target_type: str) -> list[str]:
    if target_type == "indices":
        return list(config["target"]["selected_indices"])

    # bands
    band_names = [
        "b1", "blue", "green", "red", "b5", "rededge", "b7", "nir", "b8a", "b9", "b10", "swir", "b12"
    ]
    selected_bands = config["target"].get("selected_bands", [])
    return [band_names[j] for j in selected_bands]


def evaluate_model(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    val_loader,
    num_samples: int = 5,
    split: str = "val",  # "val" or "test" (if you pass a test_loader later)
):
    """
    Evaluates patch-wise outputs from a loader and writes:
      - per-patch metrics tables (csv)
      - aggregated outputs from df
      - plots for first N scenes

    Outputs are written under:
      run_dir/metrics/tables/
      run_dir/plots/patches/<split>/...
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    logger = logging.getLogger(__name__)

    run_dir, tables_dir, plots_dir = _get_output_dirs(config)

    target_type = config["target"]["type"]
    bands_metric_names = config["performance"]["bands_metric_names"]
    indices_metric_names = config["performance"]["indices_metric_names"]

    if target_type == "bands":
        metric_names = bands_metric_names
        channel_names = _get_channel_names(config, "bands")
    elif target_type == "indices":
        metric_names = indices_metric_names
        channel_names = _get_channel_names(config, "indices")
    else:
        raise ValueError(f"Unknown target.type: {target_type}")

    model.eval()

    # ------------- Table paths -------------
    # For bands: we also compute indices-from-gt vs indices-from-inf (gt_vs_comp)
    gt_vs_comp_path = None
    if target_type == "bands":
        prefix_comp = f"{split}_patches_indices_gt_vs_comp"
        gt_vs_comp_path = tables_dir / f"{prefix_comp}.csv"

    prefix_inf = f"{split}_patches_{target_type}_gt_vs_inf"
    gt_vs_inf_path = tables_dir / f"{prefix_inf}.csv"

    scenes_to_plot = int(config.get("evaluation", {}).get("scenes_to_plot", 0))

    logger.info(f"Writing evaluation tables to: {tables_dir}")
    logger.info(f"Writing evaluation plots  to: {plots_dir}")

    # ------------- Evaluate -------------
    scene_count = 0
    processed_patches = 0

    # Use context managers to ensure files close properly
    if target_type == "bands":
        gt_vs_comp_file = open(gt_vs_comp_path, "w")
        gt_vs_comp_file.write(",".join(["scene", "indices"] + indices_metric_names) + "\n")
    else:
        gt_vs_comp_file = None

    with open(gt_vs_inf_path, "w") as gt_vs_inf_file:
        gt_vs_inf_file.write(",".join(["scene", target_type] + metric_names) + "\n")

        with torch.no_grad():
            for inputs, targets, scenes, patch_idx in tqdm(val_loader, desc=f"Evaluating {split} batches"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)

                # iterate over batch samples
                batch_n = inputs.size(0)
                for j in range(batch_n):
                    # stop after num_samples patches total
                    if processed_patches >= num_samples:
                        break

                    target_patch = targets[j].detach().cpu().squeeze().numpy()
                    output_patch = outputs[j].detach().cpu().squeeze().numpy()

                    # bands -> compute indices from gt and inf, compare
                    if target_type == "bands":
                        ind_from_gt, ind_names_from_gt = compute_vegetation_indices(config, target_patch)
                        ind_from_inf, ind_names_from_inf = compute_vegetation_indices(config, output_patch)
                        compute_all_metrics(
                            gt_vs_comp_file,
                            scenes[j],
                            ind_from_gt,
                            ind_from_inf,
                            ind_names_from_gt,
                            indices_metric_names,
                        )

                    # always: gt vs inf on target channels
                    compute_all_metrics(
                        gt_vs_inf_file,
                        scenes[j],
                        target_patch,
                        output_patch,
                        channel_names,
                        metric_names,
                    )

                    # plots only for first scenes_to_plot patches
                    if scene_count < scenes_to_plot:
                        # where to store plots
                        base_plot_dir = plots_dir / "patches" / split / target_type

                        # extra plots for indices computed from bands
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

                        # main target plots (bands or indices)
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

    # close optional file
    if gt_vs_comp_file is not None:
        gt_vs_comp_file.close()

    # ------------- Postprocess tables into outputs + plots -------------
    if target_type == "bands" and gt_vs_comp_path is not None:
        gt_vs_comp_df = pd.read_csv(gt_vs_comp_path)
        logger.info(f"Loaded: {gt_vs_comp_path} (rows={len(gt_vs_comp_df)})")
        produce_outputs_from_df(gt_vs_comp_df, config, indices_metric_names, prefix=gt_vs_comp_path.stem)

        plot_group_metric_histograms(
            output_dir=plots_dir / "patches" / split / "indices" / "metrics",
            df=gt_vs_comp_df,
            group_col="indices",
            metrics=indices_metric_names,
            prefix="gt_vs_comp",
        )

    gt_vs_inf_df = pd.read_csv(gt_vs_inf_path)
    logger.info(f"Loaded: {gt_vs_inf_path} (rows={len(gt_vs_inf_df)})")
    produce_outputs_from_df(gt_vs_inf_df, config, metric_names, prefix=gt_vs_inf_path.stem)

    plot_group_metric_histograms(
        output_dir=plots_dir / "patches" / split / target_type / "metrics",
        df=gt_vs_inf_df,
        group_col=target_type,
        metrics=metric_names,
        prefix="gt_vs_inf",
    )

    logger.info("Evaluation completed successfully.")

