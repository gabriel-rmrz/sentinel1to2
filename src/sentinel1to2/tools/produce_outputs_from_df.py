import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

    
def produce_outputs_from_df(df, config, metric_names, prefix):
  print(df.keys())
  if "indices" in df.keys():
    hist_vars = df["indices"].unique()
    var_type = "indices"
  elif "bands" in df.keys():
    hist_vars = df["bands"].unique()
    var_type = "bands"

  means_df = pd.DataFrame(columns=["hist_vars","metric","mean", "std"])
  for mn in metric_names:
    for hv in hist_vars:
      '''
      fig, ax = plt.subplots()
      df[df[var_type] ==hv].hist(mn, ax=ax)
      fig.savefig(f"plots/metrics/histos/{prefix}_{hv}_{mn}.png")
      plt.close(fig)
      '''
      means_df.loc[-1] = [mn, hv , df[df[var_type] ==hv][mn].mean(), df[df[var_type] ==hv][mn].std()]
      means_df.index = means_df.index + 1
      means_df = means_df.sort_index()
  means_df.to_csv(f"tables/{prefix}_means.csv", index=False)
  save_df_to_latex(means_df, config, prefix)

def save_df_to_latex(df, config, prefix):
    job_dir = Path(config['job']['dir'])
    # df columns: hist_vars | metric | mean | std
    wide = df.pivot(index="metric", columns="hist_vars", values=["mean", "std"])
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

    order = ["mae", "psnr", "ssim", "r2"]
    cols = pd.MultiIndex.from_product([order, ["mean", "std"]])
    wide = wide.reindex(columns=cols)

    # Combine mean ± std into single cells (use the full string key, not col[0]!)
    combined = pd.DataFrame(
        {
            hv: wide[(hv, "mean")].map(lambda m: f"{m:.3f}") +
                " ± " +
                wide[(hv, "std")].map(lambda s: f"{s:.3f}")
            for hv in order
        },
        index=wide.index
    )

    combined.index.name = ""

    # Vertical bars: one per column + outer borders
    column_format = "|l|" + "|".join(["c"] * len(combined.columns)) + "|"

    latex = combined.to_latex(
        index=True,
        multicolumn=True,
        multicolumn_format="c",
        escape=False,  # keep ±
        column_format=column_format,
        caption="Results summary",
        label="tab:results",
    )

    # Draw lines if booktabs inserted them
    latex = (latex.replace(r"\toprule", r"\hline")
                  .replace(r"\midrule", r"\hline")
                  .replace(r"\bottomrule", r"\hline"))

    Path(job_dir / f"outputs/tables/{prefix}_means.tex").write_text(latex)

'''
def save_df_to_latex(df, prefix):
  # Assuming df has columns: hist_vars | metric | mean | std
  wide = df.pivot(index="metric", columns="hist_vars", values=["mean", "std"])
  wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

  order = ["mae", "psnr", "ssim", "r2"]
  cols = pd.MultiIndex.from_product([order, ["mean", "std"]])
  wide = wide.reindex(columns=cols)

  # Optional: hide index name
  wide.index.name = ""

  # --- Here’s where the vertical lines magic happens ---
  column_format = "|l|rr|rr|rr|rr|"

  latex = wide.to_latex(
      index=True,
      multicolumn=True,
      multicolumn_format="c",
      float_format=lambda x: f"{x:.3f}",
      column_format=column_format,
      caption="Results summary",
      label="tab:results",
  )
  # assume `latex` is the string from wide.to_latex(...)
  # 1) vertical bars around each 2-col group in the header
  latex = latex.replace(r"\multicolumn{2}{c}{", r"\multicolumn{2}{|c|}{")

  # 2) use a tabular preamble with bars between groups
  latex = latex.replace(r"\begin{tabular}{l", r"\begin{tabular}{|l|rr|rr|rr|rr|}")

  # 3) if you used booktabs, swap to \hline so bars are drawn
  latex = (latex.replace(r"\toprule", r"\hline")
              .replace(r"\midrule", r"\hline")
              .replace(r"\bottomrule", r"\hline"))

  Path(f"tables/{prefix}_means.tex").write_text(latex)

'''
