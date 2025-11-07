import os
import numpy as np
from tqdm import tqdm
import h5py

from sklearn.model_selection import train_test_split
from .tools.process_scene import process_scene

def prepare_input_data():
  #data_dir = "/lustrehome/garamire/share/agri2intesa/s1_to_s2/train/"
  data_dir = "/Users/garamire/Work/AgriIntesa/sentinel1to2/src/sentinel1to2/data/train/"
  all_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
  
  # Split scena
  train_folders, val_folders = train_test_split(all_folders, test_size=0.2, random_state=42)

  """
  # Step 1: crea HDF5 train temporaneo
  train_hdf5_path = 'train_dataset_temp.h5'
  with h5py.File(train_hdf5_path, 'w') as hf:
      metadata_grp = hf.create_group("metadata")
      metadata_grp.create_dataset("scene_list", data=np.array(train_folders, dtype='S'))

      for folder in tqdm(train_folders, desc="Processing training scenes"):
          process_scene(folder, data_dir, hf)

  # Step 2: calcolo mean/std
  mean, std = compute_hdf5_mean_std(train_hdf5_path)
  print("Mean:", mean)
  print("Std:", std)

  
  
  # Salva i parametri per futuro uso
  np.savez("normalization_params.npz", mean=mean, std=std)
  """
  
  params = np.load("normalization_params.npz")
  mean = params["mean"] 
  std =  params["std"]

  
  # Step 3: crea HDF5 train definitivo normalizzato
  with h5py.File('data/train_dataset_S2.h5', 'w') as hf:
    metadata_grp = hf.create_group("metadata")
    metadata_grp.create_dataset("scene_list", data=np.array(train_folders, dtype='S'))
    for folder in tqdm(train_folders, desc="Writing normalized training scenes"):
      process_scene(folder, data_dir, hf, mean=mean, std=std)

  # Step 4: crea HDF5 val usando stessi parametri
  with h5py.File('data/val_dataset_S2.h5', 'w') as hf:
    metadata_grp = hf.create_group("metadata")
    metadata_grp.create_dataset("scene_list", data=np.array(val_folders, dtype='S'))
    for folder in tqdm(val_folders, desc="Writing normalized validation scenes"):
      process_scene(folder, data_dir, hf, mean=mean, std=std)

