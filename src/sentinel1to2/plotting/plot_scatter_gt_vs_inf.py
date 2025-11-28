import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_scatter_gt_vs_inf(output_dir: Path,
                           indices_gt: np.ndarray,   # (C, H, W)
                           indices_inf: np.ndarray,  # (C, H, W)
                           names,
                           scene,
                           prefix: str):
    """
    Plot GT vs INF scatter plot for each index.
    Uses 'lower-right' soft-density style + red x=y line.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = indices_gt.shape[0]
    assert indices_inf.shape[0] == C
    assert len(names) == C

    for i in range(C):
        raw_name = names[i]

        # Flatten
        x = indices_gt[i].astype(np.float32).flatten()
        y = indices_inf[i].astype(np.float32).flatten()

        # Clean NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter cloud (lower-right style: soft blue, tiny dots, slight density)
        ax.scatter(
            x, y,
            s=2,
            alpha=0.15,
            color="#1f77b4"  # soft blue
        )

        # Red x=y reference line
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi],
                color="red", linewidth=2, linestyle="-",
                label="x = y")

        # Labels and cosmetics
        ax.set_xlabel("GT Value", fontsize=11)
        ax.set_ylabel("Predicted Value", fontsize=11)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.grid(alpha=0.25, linestyle="--")

        # Global title
        fig.suptitle(
            f"{scene} – {raw_name.upper()}  |  Prediction vs GT",
            fontsize=14,
            y=1.02
        )

        plt.tight_layout()

        # Save
        fname = output_dir / f"{prefix}_{scene}_{raw_name}_scatter.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
