import segmentation_models_pytorch as smp
import random
import torch

from torch.utils.data import DataLoader, random_split
from train_model import train_model
from evaluate_model import evaluate_model
from tools.scene_split_dataset import scene_split_dataset
from models.losses.CombinedLoss import CombinedLoss
from batch_run_inference import batch_run_inference
from process_scenes import process_scenes
from performance import performance

def main():
  do_training = False
  do_evaluation = False
  do_inference = False
  do_performance = True
  # Configurazioni
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  batch_size = 4
  learning_rate = 1e-4
  n_workers = 4
  
  # Caricamento dataset
  train_dataset = scene_split_dataset("data/train_dataset_S2.h5")
  val_dataset = scene_split_dataset("data/val_dataset_S2.h5")
  
  # DataLoaders
  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=True
  )
  
  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=True
  )

  model = smp.Unet(
    encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7                 # meno blocchi dell’encoder
    encoder_weights='imagenet',             # o None
    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=9,                      # model output channels (number of classes in your dataset)
  #    decoder_channels=(128, 64, 32),
  #    encoder_depth=3
  )

  #model = UNet(in_channels=4, init_features=64, out_channels=1).to(device)
  #decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None,
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  if do_training:
    print(device)
    print( sum(p.numel() for p in model.parameters() if p.requires_grad) )
    #model = UNet(in_channels=4, init_features=32, out_channels=1).to(device)
    #model =  EffUNet(in_channels=6, classes=1)
    #model = CustomUNet(in_channels=6, init_features=32, depth=5, dropout_rate=0.05, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = nn.L1Loss() # nn.MSELoss()  # Per regressione
    criterion = CombinedLoss(alpha=1.0, beta=2, gamma=0.1)
    
    # Addestramento
    train_losses, val_losses = train_model(
      model,
      device,
      train_loader,
      val_loader,
      criterion,
      optimizer,
      epochs=50,
      patience=10
    )
  if do_evaluation:
    if not do_training:
      model.load_state_dict(torch.load("best_model.pth"))
    evaluate_model(model, device, val_loader, num_samples= 100000)
  if do_inference:
    batch_run_inference(
      model_path="best_model.pth",
      data_dir="data/inference/",
      output_dir="data/output_combined/",
      device="cpu"
    )
  if do_performance:
    data_dir = "data/inference"
    pred_dir = "data/output_combined"
    results = process_scenes(data_dir, pred_dir)
    performance()

if __name__=='__main__':
  main()
