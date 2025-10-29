import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# === Analisi per classi ESA ===
def plot_r2_by_class(r2_map_path, esa_path):
    with rasterio.open(r2_map_path) as src_r2, rasterio.open(esa_path) as src_esa:
        r2 = src_r2.read(1)
        esa = src_esa.read(1, out_shape=r2.shape, resampling=rasterio.enums.Resampling.nearest)

    mask = np.isfinite(r2)
    r2 = r2[mask]
    esa = esa[mask]
    r2[r2 < -1] = -1

    classes = np.unique(esa)
    stats = []
    plt.figure(figsize=(10,6))
    data = []
    labels = []
    for cls in classes:
        vals = r2[esa == cls]
        if len(vals) > 30:
            data.append(vals)
            labels.append(str(cls))
            stats.append((cls, len(vals), np.nanmean(vals), np.nanstd(vals)))

    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("ESA WorldCover class")
    plt.ylabel("R²")
    plt.title("Distribuzione R² per classe ESA")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "r2_by_class_boxplot.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"📊 Boxplot salvato: {boxplot_path}")

    csv_path = os.path.join(output_dir, "r2_by_class.csv")
    np.savetxt(csv_path, stats, delimiter=",", fmt=["%d", "%d", "%.4f", "%.4f"],
               header="ESA_Class,Count,Mean_R2,Std_R2", comments='')
    print(f"📄 CSV statistiche per classe: {csv_path}")
