import numpy as np
import rasterio

def load_predicted_ndvi(pred_path):
  with rasterio.open(pred_path) as src:
    pred_ndvi = src.read().astype(np.float32)
    #pred_ndvi = (pred_ndvi[6]-pred_ndvi[2])/(pred_ndvi[6] + pred_ndvi[2])
    #pred_ndvi = np.clip(pred_ndvi, -1, 1)
    #pred_ndvi = pred_ndvi[KKK]
    print(pred_ndvi.shape)
  return pred_ndvi
