from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_sam_histogram(
    sam_csv: Path,
    out_dir: Path,
    prefix: str = "sam",
    bins: int = 50,
) -> None:
    """
    Plot a histogram of SAM values from a CSV with columns: scene,sam
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sam_csv)

    if df.empty or "sam" not in df.columns:
        return

    vals = df["sam"].dropna().values
    if len(vals) == 0:
        return

    plt.figure()
    plt.hist(vals, bins=bins)
    plt.xlabel("SAM (radians)")
    plt.ylabel("Count")
    plt.title("SAM distribution")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_hist.png")
    plt.close()


def plot_sam_per_scene(
    sam_csv: Path,
    out_dir: Path,
    prefix: str = "sam",
    max_scenes: int = 200,
) -> None:
    """
    Plot SAM values per scene as a simple scatter (or line) plot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sam_csv)

    if df.empty or "sam" not in df.columns or "scene" not in df.columns:
        return

    df = df.dropna(subset=["sam"]).copy()
    if df.empty:
        return

    # keep only first max_scenes for readability
    df = df.head(max_scenes)

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(df)), df["sam"].values, marker="o", linestyle="None")
    plt.xlabel("Scene index (truncated)")
    plt.ylabel("SAM (radians)")
    plt.title("SAM per scene")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_per_scene.png")
    plt.close()

