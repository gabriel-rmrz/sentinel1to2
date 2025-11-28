import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
#from ..tools.stretch_2d import stretch_2d

# Reuse your existing color mapping
METRIC_COLORS = {
    "mae":  "#f4a62a",
    "psnr": "#1f77b4",
    "ssim": "#2ca02c",
    "r2":   "#9467bd",
}
def plot_group_metric_histograms(
    output_dir: Path,
    df: pd.DataFrame,
    group_col: str,                 # "band" or "veg_index"
    metrics=("mae", "psnr", "ssim", "r2"),
    prefix: str = "metrics",
):
    """
    Generic plotter for metrics grouped by a column (band or veg_index).

    Produces:
      1) Histograms: one per (group, metric), values across scenes.
      2) Violin plots: one per metric, groups side by side.

    Parameters
    ----------
    output_dir : Path
        Directory where PNGs are saved.
    df : pd.DataFrame
        DataFrame with at least columns: ["scene", group_col] + metric columns.
    group_col : str
        Name of the column used to group entries (e.g. "band" or "veg_index").
    metrics : iterable of str
        Metric column names to plot (e.g. ["mae","psnr","ssim","r2"]).
    prefix : str
        Prefix for output filenames.
    """
    output_dir = Path(output_dir)
    output_dir_violin = output_dir / 'violin'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_violin.mkdir(parents=True, exist_ok=True)

    required = {"scene", group_col, *metrics}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    groups = sorted(df[group_col].unique())

    # ------------------------------------------------------------------
    # 1) Histograms per (group, metric)
    # ------------------------------------------------------------------
    for g in groups:
        df_g = df[df[group_col] == g]

        for metric in metrics:
            vals = df_g[metric].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            # Stats
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            n_val = int(vals.size)

            stats_text = (
                f"N:    {n_val}\n"
                f"Mean: {mean_val:.4f}\n"
                f"Std:  {std_val:.4f}"
            )

            color = METRIC_COLORS.get(metric, "#f4a62a")

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(
                vals,
                bins=20,
                color=color,
                edgecolor="black",
                alpha=0.85,
            )

            ax.set_xlabel(metric.upper(), fontsize=11)
            ax.set_ylabel("Frequency (scenes)", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

            ax.text(
                0.97, 0.97, stats_text,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.3",
                    alpha=0.85,
                ),
            )

            title_label = f"{group_col}={g}".upper()
            fig.suptitle(
                f"{title_label} – {metric.upper()} across scenes",
                fontsize=14,
                y=1.02,
            )

            plt.tight_layout()

            fname = output_dir / f"{prefix}_{group_col}_{g}_{metric}_hist.png"
            fig.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close(fig)

    # ------------------------------------------------------------------
    # 2) Violin plots per metric: groups side by side
    # ------------------------------------------------------------------
    for metric in metrics:
        data_per_group = []
        labels = []

        for g in groups:
            vals = df.loc[df[group_col] == g, metric].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data_per_group.append(vals)
            labels.append(str(g))

        if len(data_per_group) == 0:
            continue

        color = METRIC_COLORS.get(metric, "#f4a62a")

        fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(labels)), 5))

        vp = ax.violinplot(
            data_per_group,
            showmeans=True,
            showextrema=True,
            showmedians=False,
        )

        # Style violins
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.6)

        # Means / extrema color
        for partname in ('cbars', 'cmins', 'cmaxs', 'cmeans'):
            if partname in vp:
                vp[partname].set_edgecolor("black")
                vp[partname].set_linewidth(1.0)

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        fig.suptitle(
            f"{metric.upper()} across {group_col.replace('_', ' ')}s",
            fontsize=14,
            y=1.02,
        )

        plt.tight_layout()

        fname = output_dir_violin / f"{prefix}_{group_col}_{metric}_violin.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
