import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_abs_error(output_dir: Path,
                   indices_gt: np.ndarray,   # (C, H, W)
                   indices_inf: np.ndarray,  # (C, H, W)
                   names,
                   scene,
                   prefix: str):
    """
    Plot absolute error histograms for each index/band:
        | GT - INF |
    Adds statistics box and consistent styling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = indices_gt.shape[0]
    assert indices_inf.shape[0] == C
    assert len(names) == C

    for i in range(C):
        raw_name = names[i]

        # ----------------------------------------------------------
        # Compute absolute error and stats
        # ----------------------------------------------------------
        abs_err = np.abs(indices_gt[i] - indices_inf[i]).flatten()
        mean_val = np.mean(abs_err)
        std_val = np.std(abs_err)

        stats_text = (
            f"Mean: {mean_val:.4f}\n"
            f"Std:  {std_val:.4f}"
        )

        # ----------------------------------------------------------
        # Plot histogram
        # ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(
            abs_err,
            bins=30,
            color="#f4a62a",        # warm orange
            edgecolor="black",
            alpha=0.85
        )

        ax.set_xlabel("Absolute Error", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        # ----------------------------------------------------------
        # Stats box (top-right)
        # ----------------------------------------------------------
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.3",
                alpha=0.85
            )
        )

        # ----------------------------------------------------------
        # Global title
        # ----------------------------------------------------------
        fig.suptitle(
            f"{scene} – {raw_name.upper()}  |  Absolute Error Histogram",
            fontsize=14,
            y=1.02
        )

        plt.tight_layout()

        # ----------------------------------------------------------
        # Save
        # ----------------------------------------------------------
        fname = output_dir / f"{prefix}_{scene}_{raw_name}_abs_error.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
