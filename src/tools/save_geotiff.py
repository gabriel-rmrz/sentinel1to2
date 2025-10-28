import rasterio
import numpy as np


def save_geotiff(output_array, profile, output_path):
  if output_array.ndim == 2:
    output_array = output_array[np.newaxis, ...]
  count = output_array.shape[0]

  profile.update(dtype=rasterio.float32, count=count)

  with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(output_array.astype(np.float32))
