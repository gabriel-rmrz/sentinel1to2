import os 
from tools.compute_sam import compute_sam
from tools.compute_ndvi_from_s2 import compute_ndvi_from_s2
from tools.load_predicted_ndvi import load_predicted_ndvi  
from tools.compare_ndvi import compare_ndvi

def process_scenes(data_dir, pred_dir):
  results = []
  for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
      continue

    second_part = folder.split("_")[1]
    s2_path = os.path.join(folder_path, f"{second_part}_s2.tif")
    #s2_path = os.path.join(folder_path, f"{folder}_s2.tif")
    pred_path = os.path.join(pred_dir, f"{folder}_pred.tif")

    print(s2_path)
    print(pred_path)

    if not os.path.exists(s2_path) or not os.path.exists(pred_path):
      print(f"[SKIP] Mancante: {folder}")
      continue
    try:
      ndvi_true, _ = compute_ndvi_from_s2(s2_path)
      ndvi_pred = load_predicted_ndvi(pred_path)
        
      # Eventuale resize (solo se dimensioni non corrispondono)
      if ndvi_true.shape != ndvi_pred.shape:
          print(f"[WARN] Dimensioni diverse per {folder}, saltato")
          continue

      sam = compute_sam(ndvi_true, ndvi_pred)
      #results.append((folder,sam))
      mae, ssim_val, psnr_val = compare_ndvi(ndvi_true, ndvi_pred)
      print(f"{folder} — MAE: {mae:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}, SAM:  {sam:.2f}")
      results.append((folder, mae, ssim_val, psnr_val,sam))

    except Exception as e:
      print(f"[ERROR] Errore in {folder}: {e}")

  return results
  
