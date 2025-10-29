import os
import torch
import numpy as np
import segmentation_models_pytorch as smp

from .tools.infer_on_scene import infer_on_scene
from .tools.save_geotiff import save_geotiff
from .tools.load_and_stack_full import load_and_stack_full

def inference(scene_folder, model_path, data_dir, output_dir, device='cuda'):
    # === Normalizzazione ===
    norm_params = np.load("normalization_params.npz")
    MEAN = norm_params["mean"]
    STD = norm_params["std"]
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = smp.Unet(encoder_name="efficientnet-b0", in_channels=4, classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Carica stack input
    dsm, s1, wc, _ind, profile = load_and_stack_full(scene_folder, data_dir, MEAN, STD)
    input_stack = np.concatenate([dsm,s1,wc], axis=0)
    print(f"Input shape: {input_stack.shape}")

    # Inference
    output = infer_on_scene(model, input_stack, device)

    # Salva TIFF
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scene_folder}_pred.tif")
    save_geotiff(output, profile, out_path)
    print(f"✅ Output salvato in: {out_path}")
