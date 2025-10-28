import numpy as np
import rasterio
# === Caricamento immagine ===
def load_image(path, bands=None, scale=None):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
        if bands is not None:
            img = img[bands]
        if scale:
            img = img / scale
    return img  
