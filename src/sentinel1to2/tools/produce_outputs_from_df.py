from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _get_run_metrics_dir(config: dict) -> Path:
    run_dir = Path(config["paths"]["run_dir"])
    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _detect_group_col(df: pd.DataFrame, config: dict) -> str:
    """
    Decide which column groups the metrics:
      - prefers explicit columns in df: 'bands' or 'indices'
      - otherwise falls back to config['target']['type'] if present in df
    """
    cols = set(df.columns)

    if "indices" in cols:
        return "indices"
    if "bands" in cols:
        return "bands"

    # fallback: sometimes the group column is literally the target_type
    target_type = str(config.get("target", {}).get("type", "")).lower()
    if target_type and target_type in cols:
        return target_type

    raise ValueError(
        "Could not detect group column. Expected one of: 'bands', 'indices', or config['target']['type'] present in df."
    )


def summarize_metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric_names: Iterable[str],
) -> pd.DataFrame:
    """
    Returns a tidy DF with columns:
      group | metric | mean | std
    """
    metric_names = list(metric_names)

    missing = [m for m in metric_names if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns in df: {missing}")

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in df columns: {list(df.columns)}")

    rows = []
    for metric in metric_names:
        for group_value, sub in df.groupby(group_col):
            rows.append(
                {
                    "group": group_value,
                    "metric": metric,
                    "mean": float(sub[metric].mean()),
                    "std": float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0,
                }
            )

    return pd.DataFrame(rows)


def _format_mean_std_table(
    summary_df: pd.DataFrame,
    metric_names: list[str],
) -> pd.DataFrame:
    """
    Convert tidy summary into a wide table where each cell is "mean ± std".
    Index: metric
    Columns: group values
    """
    if summary_df.empty:
        return pd.DataFrame()

    # pivot mean and std separately
    wide_mean = summary_df.pivot(index="metric", columns="group", values="mean")
    wide_std = summary_df.pivot(index="metric", columns="group", values="std")

    # enforce metric row order
    wide_mean = wide_mean.reindex(metric_names)
    wide_std = wide_std.reindex(metric_names)

    # "mean ± std"
    combined = pd.DataFrame(index=wide_mean.index)
    for col in wide_mean.columns:
        combined[col] = wide_mean[col].map(lambda x: f"{x:.3f}") + " ± " + wide_std[col].map(lambda x: f"{x:.3f}")

    combined.index.name = ""
    return combined


def save_summary_csv(summary_df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)


def save_summary_latex(
    summary_df: pd.DataFrame,
    metric_names: list[str],
    out_tex: Path,
    caption: str = "Results summary",
    label: str = "tab:results",
) -> None:
    """
    Writes a LaTeX table where entries are "mean ± std".
    """
    combined = _format_mean_std_table(summary_df, metric_names)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    if combined.empty:
        out_tex.write_text("% Empty summary table\n")
        return

    # nice vertical borders
    column_format = "|l|" + "|".join(["c"] * len(combined.columns)) + "|"

    latex = combined.to_latex(
        index=True,
        escape=False,  # keep ±
        column_format=column_format,
        caption=caption,
        label=label,
    )

    # replace booktabs rules with \hline for the vertical bars to show
    latex = (
        latex.replace(r"\toprule", r"\hline")
        .replace(r"\midrule", r"\hline")
        .replace(r"\bottomrule", r"\hline")
    )

    out_tex.write_text(latex)


def produce_outputs_from_df(
    df: pd.DataFrame,
    config: dict,
    metric_names: list[str],
    prefix: str,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main entry point used by evaluate/performance scripts.

    Writes:
      - run_dir/metrics/{prefix}_means.csv
      - run_dir/metrics/{prefix}_means.tex

    Returns
    -------
    summary_df : pd.DataFrame
        Tidy summary table with columns: group | metric | mean | std
    """
    logging.getLogger(__name__).info(f"Producing summary outputs for prefix='{prefix}'")

    if df is None or df.empty:
        logging.getLogger(__name__).warning("Input dataframe is empty; skipping outputs.")
        return pd.DataFrame(columns=["group", "metric", "mean", "std"])

    if group_col is None:
        group_col = _detect_group_col(df, config)

    out_dir = _get_run_metrics_dir(config)
    out_csv = out_dir / f"{prefix}_means.csv"
    out_tex = out_dir / f"{prefix}_means.tex"

    summary_df = summarize_metrics_by_group(df, group_col=group_col, metric_names=metric_names)

    save_summary_csv(summary_df, out_csv)
    save_summary_latex(
        summary_df,
        metric_names=metric_names,
        out_tex=out_tex,
        caption=f"Results summary ({prefix})",
        label=f"tab:{prefix}",
    )

    return summary_df

