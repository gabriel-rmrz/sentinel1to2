import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .tools.produce_outputs_from_df import produce_outputs_from_df
from .tools.compute_vegetation_indices import compute_vegetation_indices
from .tools.load_image import load_image
from .tools.compute_metrics import compute_all_metrics
from .plotting.plot_comparison_rgb_composites_2d import (
    plot_comparison_rgb_composites_2d,
)
from .plotting.plot_comparison_histos_2d import plot_comparison_histos_2d
from .plotting.plot_s2_composites_2d import plot_s2_composites_2d
from .plotting.plot_scatter_gt_vs_inf import plot_scatter_gt_vs_inf
from .plotting.plot_abs_error import plot_abs_error
from .plotting.plot_histo_2d import plot_histo_2d
from .plotting.plot_group_metric_histograms import plot_group_metric_histograms


def read_csv_to_list(path: Path) -> List[str]:
    """Read a one-column CSV and return the first column as a list of strings."""
    rows: List[str] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row[0])
    return rows


def _resolve_data_dirs(
    config: dict,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Resolve all directories used in performance().

    """
    job_dir = Path(config["job"]["dir"])
    job_data_dir = job_dir / "data"
    job_outputs_dir = job_dir / "outputs"
    job_tables_dir = job_outputs_dir / "tables"
    job_plots_dir = job_outputs_dir / "plots"
    job_lists_dir = job_data_dir / "lists"

    # If caller did not pass explicit dirs, fall back to config["inference"]
    real_dir = Path(config["inference"]["input_dir"])

    pred_dir = Path(config["inference"]["output_dir"])

    job_tables_dir.mkdir(parents=True, exist_ok=True)
    job_plots_dir.mkdir(parents=True, exist_ok=True)

    return real_dir, pred_dir, job_data_dir, job_tables_dir, job_plots_dir, job_lists_dir


def performance(
    config: dict,
    sample_type: str = "test",
) -> None:
    """
    Evaluate model performance on full Sentinel-2 scenes.

    This function:
      * Loads GT Sentinel-2 scenes and predicted scenes.
      * Computes per-band metrics (always).
      * Optionally computes vegetation indices from S2 bands (only if
        target_type == "bands") and metrics on those indices.
      * Writes summary CSVs and produces plots.

    Parameters
    ----------
    config : dict
        Global configuration dictionary (parsed from YAML).
    real_dir : Path, optional
        Directory with *ground-truth* scenes. If None, taken from
        config["inference"]["input_dir"] (relative to job data dir if not
        absolute).
    pred_dir : Path, optional
        Directory with predicted scenes. If None, taken from
        config["inference"]["output_dir"] (relative to job data dir if not
        absolute).
    sample_type : str
        Dataset split name ("train", "val", "test", ...). Used to select
        the scene list CSV and prediction filename prefix.
    """
    (
        real_dir,
        pred_dir,
        job_data_dir,
        job_tables_dir,
        job_plots_dir,
        job_lists_dir,
    ) = _resolve_data_dirs(config)

    target_type = config["target"]["type"]  # "bands" or e.g. "indices"
    tile_type = "scenes"

    # Metric names depend on what the network is predicting
    metric_names = config["performance"][f"{target_type}_metric_names"]

    # Bands configuration (always used for GT S2 loading; indices are derived
    # only if target_type == "bands")
    all_bands = config["target"]["all_bands"]
    selected_bands = config["target"]["selected_bands"]
    channel_names = [all_bands[j] for j in selected_bands]

    # List of scenes to evaluate (e.g. "test_scenes_inferred_list.csv")
    scene_list_path = job_lists_dir / f"{sample_type}_scenes_inferred_list.csv"
    list_of_scenes = read_csv_to_list(scene_list_path)

    # ------------------------------------------------------------------
    # Output CSVs
    #   1) GT indices vs indices from INF (only if target_type == "bands")
    #   2) GT bands vs predicted bands (always)
    # ------------------------------------------------------------------
    gt_vs_comp_file = None
    indices_metric_names = None
    if target_type == "bands":
        indices_metric_names = config["performance"]["indices_metric_names"]
        prefix1 = f"{sample_type}_{tile_type}_{target_type}_gt_vs_comp"
        table1_path = job_tables_dir / f"{prefix1}.csv"
        gt_vs_comp_file = open(table1_path, "w")
        gt_vs_comp_file.write(",".join(["scene", "indices", *indices_metric_names]) + "\n")
    else:
        # When the target is vegetation indices directly, we do not recompute them
        # from S2 bands here; only GT-vs-INF metrics on the target itself.
        prefix1 = None
        table1_path = None

    prefix2 = f"{sample_type}_{tile_type}_{target_type}_gt_vs_inf"
    table2_path = job_tables_dir / f"{prefix2}.csv"
    gt_vs_inf_file = open(table2_path, "w")
    gt_vs_inf_file.write(",".join(["scene", target_type] +  metric_names) + "\n")

    # ------------------------------------------------------------------
    # Main loop over scenes
    # ------------------------------------------------------------------
    scene_count = 0
    scenes_to_plot = config["evaluation"]["scenes_to_plot"]

    for dname in tqdm(list_of_scenes, desc=f"Performance on {sample_type} sample"):
        # Scene folder is expected to be "{scene}_{day}"
        try:
            scene, day = dname.split("_")
        except ValueError:
            print(f"[SKIP] '{dname}' does not match 'scene_day' pattern")
            continue

        gt_path = real_dir / dname / f"{day}_s2.tif"
        pred_path = job_data_dir / pred_dir / f"{sample_type}_{dname}_pred.tif"

        missing_files = [str(p) for p in (gt_path, pred_path) if not p.is_file()]
        if missing_files:
            print(f"[SKIP] Missing files for scene {dname}: {missing_files}")
            continue

        try:
            # ------------------------------------------------------------------
            # Load GT and prediction
            # ------------------------------------------------------------------
            # Sentinel-2 GT bands: select subset and rescale to [0, 1]
            channels_gt = load_image(str(gt_path), selected_bands)
            channels_gt = channels_gt / 10000.0

            # Predicted S2 bands / targets: stored already in training scale
            channels_inf = load_image(str(pred_path))

            # ------------------------------------------------------------------
            # Compute per-target metrics (bands or indices, depending on target_type)
            # ------------------------------------------------------------------
            compute_all_metrics(
                gt_vs_inf_file,
                dname,
                channels_gt,
                channels_inf,
                channel_names,
                metric_names,
            )

            # ------------------------------------------------------------------
            # Optional: compute vegetation indices metrics only if target is bands
            # ------------------------------------------------------------------
            if target_type == "bands" and gt_vs_comp_file is not None:
                ind_from_gt, ind_names_from_gt = compute_vegetation_indices(
                    config, channels_gt
                )
                ind_from_inf, ind_names_from_inf = compute_vegetation_indices(
                    config, channels_inf
                )

                compute_all_metrics(
                    gt_vs_comp_file,
                    dname,
                    ind_from_gt,
                    ind_from_inf,
                    ind_names_from_gt,
                    metric_names,
                )

            # ------------------------------------------------------------------
            # Scene-level plots (only for a subset of scenes)
            # ------------------------------------------------------------------
            if scene_count < scenes_to_plot:
                # If we have bands as target, also plot index-based diagnostics
                if target_type == "bands":
                    job_plots_comp_ind_dir = job_plots_dir / f"{tile_type}/{sample_type}"

                    plot_histo_2d(
                        f"{job_plots_comp_ind_dir}/indices/histos2d",
                        ind_from_gt,
                        ind_names_from_gt,
                        dname,
                        prefix="computed_from_gt",
                    )
                    plot_histo_2d(
                        f"{job_plots_comp_ind_dir}/indices/histos2d",
                        ind_from_inf,
                        ind_names_from_inf,
                        dname,
                        prefix="computed_from_inf",
                    )
                    plot_comparison_histos_2d(
                        f"{job_plots_comp_ind_dir}/indices/histos2d_comparison",
                        ind_from_gt,
                        ind_from_inf,
                        ind_names_from_gt,
                        dname,
                        prefix="computed_from_gt_inf",
                    )
                    plot_scatter_gt_vs_inf(
                        f"{job_plots_comp_ind_dir}/indices/scatter_gt_vs_inf",
                        ind_from_gt,
                        ind_from_inf,
                        ind_names_from_gt,
                        dname,
                        prefix="computed_from_gt_vs_inf",
                    )
                    plot_abs_error(
                        f"{job_plots_comp_ind_dir}/indices/histos_abs_error",
                        ind_from_gt,
                        ind_from_inf,
                        ind_names_from_gt,
                        dname,
                        prefix="computed_from_gt_vs_inf",
                    )

                # Bands / RGB composites + histograms (always)
                plot_comparison_rgb_composites_2d(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos2d_comparison",
                    channels_gt,
                    channels_inf,
                    channel_names,
                    dname,
                    prefix="gt_inf",
                )

                plot_s2_composites_2d(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos2d",
                    channels_gt,
                    channel_names,
                    dname,
                    prefix="gt",
                )
                plot_s2_composites_2d(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos2d",
                    channels_inf,
                    channel_names,
                    dname,
                    prefix="inf",
                )

                plot_histo_2d(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos2d",
                    channels_gt,
                    channel_names,
                    dname,
                    prefix="gt",
                )
                plot_histo_2d(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos2d",
                    channels_inf,
                    channel_names,
                    dname,
                    prefix="inf",
                )

                plot_comparison_rgb_composites_2d(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos2d_comparison",
                    channels_gt,
                    channels_inf,
                    channel_names,
                    dname,
                    prefix="gt_inf",
                )
                plot_comparison_histos_2d(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos2d_comparison",
                    channels_gt,
                    channels_inf,
                    channel_names,
                    dname,
                    prefix="gt_inf",
                )

                plot_scatter_gt_vs_inf(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/scatter_gt_vs_inf",
                    channels_gt,
                    channels_inf,
                    channel_names,
                    dname,
                    prefix="gt_vs_inf",
                )

                plot_abs_error(
                    job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/histos_abs_error",
                    channels_gt,
                    channels_inf,
                    channel_names,
                    dname,
                    prefix="gt_vs_inf",
                )

                scene_count += 1

        except Exception as e:  # broad catch so one bad scene doesn't kill everything
            print(f"[ERROR] Error for the scene {dname}: {e}")

    # ------------------------------------------------------------------
    # Close files
    # ------------------------------------------------------------------
    gt_vs_inf_file.close()
    if gt_vs_comp_file is not None:
        gt_vs_comp_file.close()

    # ------------------------------------------------------------------
    # Aggregate CSVs → DataFrames, then produce summary tables + plots
    # ------------------------------------------------------------------
    if target_type == "bands" and table1_path is not None:
        print(table1_path)
        gt_vs_comp_df = pd.read_csv(table1_path)
        print(gt_vs_comp_df)

        if not gt_vs_comp_df.empty:
            produce_outputs_from_df(gt_vs_comp_df, config, indices_metric_names, prefix1)
            plot_group_metric_histograms(
                output_dir=Path(job_plots_dir / f"{tile_type}/{sample_type}/indices/metrics"),
                df=gt_vs_comp_df,
                group_col="indices",
                metrics=indices_metric_names,
                prefix="gt_vs_comp",
            )
        else:
            print(f"[WARN] Indices metrics table {table1_path} is empty; skipping summary plots.")

    print(table2_path)
    gt_vs_inf_df = pd.read_csv(table2_path)
    print(gt_vs_inf_df)

    if not gt_vs_inf_df.empty:
        produce_outputs_from_df(gt_vs_inf_df, config, metric_names, prefix2)
        plot_group_metric_histograms(
            output_dir=Path(job_plots_dir / f"{tile_type}/{sample_type}/{target_type}/metrics"),
            df=gt_vs_inf_df,
            group_col=target_type,
            metrics=metric_names,
            prefix="gt_vs_inf",
        )
    else:
        print(f"[WARN] GT vs INF metrics table {table2_path} is empty; skipping summary plots.")

