import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

def train_model(model, device, config,  train_loader, val_loader, criterion, optimizer, epochs=100, patience=5):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
  )
  job_dir = Path(config["job"]["dir"])
  job_data_dir = job_dir / "data"
  best_val_loss = np.inf
  no_improve = 0
  train_losses = []
  val_losses = []
  
  for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0
    
    # Training
    for inputs, targets, _scene, _patch_idx in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
      
      epoch_train_loss += loss.item() * inputs.size(0)
    
    # Validation
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
      for inputs, targets, _scene, _patch_idx in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        epoch_val_loss += loss.item() * inputs.size(0)
    
    # Calculate metrics
    train_loss = epoch_train_loss / len(train_loader.dataset)
    val_loss = epoch_val_loss / len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      no_improve = 0
      torch.save(model.state_dict(), job_data_dir / config["training"]["model_output"])
    else:
      no_improve += 1
        
    logging.info(f'Epoch {epoch+1}/{epochs}')
    logging.info(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    if no_improve >= patience:
      logging.info(f'Early stopping at epoch {epoch+1}')
      return train_losses, val_losses
  
  return train_losses, val_losses
