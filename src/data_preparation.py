import numpy as np
import os
import h5py
from tqdm import tqdm
from tools.process_scene import process_scene
from tools.compute_hdf5_mean_std import compute_hdf5_mean_std
from sklearn.model_selection import train_test_split

#TODO: Add this to the pipeline
#TODO: set max_number_of_files, ouput_file name, as an input parameter (maybe using a config)
#TODO: Fix the computation of the mean and std

def main():
  do_params = True
  data_dir = "/Users/garamire/Work/AgriIntesa/sentinel1to2/data/train/"
  all_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
  
  # Split scena
  train_folders, val_folders = train_test_split(all_folders, test_size=0.2, random_state=42)

  '''
  if do_params:
    # Step 1: crea HDF5 train temporaneo
    train_hdf5_path = 'data/train_dataset_temp.h5'
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
  '''
  
  params = np.load("normalization_params.npz")
  mean = params["mean"] 
  std =  params["std"]
  print(f"Mean: {mean}")
  print(f"Std: {std}")
  exit()

  
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

  
  # Cleanup
  
  if do_params:
    os.remove(train_hdf5_path)

if __name__=='__main__':
  main()
