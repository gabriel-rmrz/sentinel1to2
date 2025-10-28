import os 
import re
# === Parser per reali ===
def get_real_scenes(real_dir):
  scene_dict = {}
  pattern = re.compile(r"(\d+)_(\d+)")
  for folder in os.listdir(real_dir):
    match = pattern.match(folder)
    if match:
      scene, day = match.groups()
      full_path = os.path.join(real_dir, folder, f"{day}_s2.tif")
      if os.path.exists(full_path):
        scene_dict.setdefault(scene, []).append(int(day))
  return {k: sorted(v) for k, v in scene_dict.items() if len(v) >= 2}
