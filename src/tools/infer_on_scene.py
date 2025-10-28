import torch
import numpy as np
from tqdm import tqdm 
from tools.pad_image import pad_image

def infer_on_scene(model, input_stack, device, patch_size=128, stride=32):
  model.eval()
  input_stack, orig_h, orig_w = pad_image(input_stack, patch_size, stride)
  c, h, w = input_stack.shape

  # Calcolo numero canali output C
  with torch.no_grad():
    dummy_input = torch.from_numpy(np.zeros((1, c, patch_size, patch_size), dtype=np.float32)).to(device)
    C = model(dummy_input).shape[1]

  output_map = np.zeros((C, h, w), dtype=np.float32)
  count_map = np.zeros((C, h, w), dtype=np.float32)

  with torch.no_grad():
    for top in tqdm(range(0, h - patch_size + 1, stride), desc="Infer patch rows"):
      for left in range(0, w - patch_size + 1, stride):
        patch = input_stack[:, top:top+patch_size, left:left+patch_size]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

        pred = model(patch_tensor).cpu().squeeze(0).numpy()  # (C, H, W)
        output_map[:, top:top+patch_size, left:left+patch_size] += pred
        count_map[:, top:top+patch_size, left:left+patch_size] += 1

  count_map[count_map == 0] = 1
  output_map /= count_map

  return output_map[:, :orig_h, :orig_w]  # taglio padding
