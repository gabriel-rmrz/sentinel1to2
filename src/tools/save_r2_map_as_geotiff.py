import rasterio

def save_r2_map_as_geotiff(scene, r2_map, ref_path):
  with rasterio.open(ref_path) as ref:
    meta = ref.meta.copy()
  meta.update({
    "count": 1,
    "dtype": "float32"
  })
  out_path = os.path.join(output_dir, f"{scene}_r2_map.tif")
  with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(r2_map.astype(np.float32), 1)
  print(f"🗺️ Salvato GeoTIFF R²: {out_path}")
  return out_path
