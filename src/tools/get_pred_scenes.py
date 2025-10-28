import os
import re

# === Parser per predizioni ===
def get_pred_scenes(pred_dir):
  scene_dict = {}
  pattern = re.compile(r"(\d+)_(\d+)_pred\.tif")
  for fname in os.listdir(pred_dir):
    match = pattern.match(fname)
    if match:
      scene, day = match.groups()
      scene_dict.setdefault(scene, []).append(int(day))
  return {k: sorted(v) for k, v in scene_dict.items() if len(v) >= 2}
