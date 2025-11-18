#import os
import logging
import csv
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
import h5py

from sklearn.model_selection import train_test_split
from .tools.process_scene import process_scene
from .tools.compute_hdf5_mean_std import compute_hdf5_mean_std

def write_list_to_csv(path, list_out):
  with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([[x] for x in list_out])
  return

def prepare_input_data(config):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
  )

  logging.info("Preparation of the input samples")
  params = config['preprocessing']
  data_dir = Path(params['input_dir'])
  job_dir = Path(config['job']['dir'])
  job_data_dir = job_dir/'data'

  train_tmp_hdf5_path = job_data_dir / 'train_dataset_temp.h5'
  train_hdf5_path = job_data_dir / 'train_dataset_S2.h5'
  val_hdf5_path = job_data_dir / 'val_dataset_S2.h5'

  if job_dir.exists():
    answer = input(f"The job folder '{job_dir}' already exists. Do you want to delete it? [yes/N]: ").strip().lower()

    if answer == "yes":
      shutil.rmtree(job_dir)
      logging.info(f"Directory {job_dir} have been recreated.")
    else:
      logging.info(f"The preparation of the samples has been aborted.")
      return

  job_data_dir.mkdir(parents=True, exist_ok=False)
  all_scenes = sorted([f.name for f in Path(data_dir).iterdir() if f.is_dir()])
  sample_size = params['sample_size']
  if sample_size == 0 or sample_size > len(all_scenes):
    sample_scenes = all_scenes
    sample_size = len(all_scenes)
  else:
    sample_scenes = all_scenes[:sample_size]

  logging.info(f"Sampling {sample_size} out of {len(all_scenes)} scenes available in {data_dir}")
  
  # Split scena
  train_folders, val_folders = train_test_split(sample_scenes, test_size=0.2, random_state=42)
  write_list_to_csv(job_data_dir / 'training_scene_list.csv', train_folders)
  write_list_to_csv(job_data_dir / 'validation_scene_list.csv', val_folders)
  
  if params['do_norm_params']:
    # Step 1: crea HDF5 train temporaneo
    with h5py.File(train_tmp_hdf5_path, 'w') as hf:
        metadata_grp = hf.create_group("metadata")
        metadata_grp.create_dataset("scene_list", data=np.array(train_folders, dtype='S'))
        for folder in tqdm(train_folders, desc="Processing training scenes"):
            process_scene(folder, data_dir, hf)

    # Step 2: calcolo mean/std
    mean, std = compute_hdf5_mean_std(train_tmp_hdf5_path)
    print("Mean:", mean)
    print("Std:", std)
    # Salva i parametri per futuro uso
    np.savez(job_data_dir / "normalization_params.npz", mean=mean, std=std)
  else:
    params = np.load("normalization_params.npz")
    mean = params["mean"] 
    std =  params["std"]
    np.savez(job_data_dir/"normalization_params.npz", mean=mean, std=std)
  
  # Step 3: crea HDF5 train definitivo normalizzato
  with h5py.File(train_hdf5_path, 'w') as hf:
    metadata_grp = hf.create_group("metadata")
    metadata_grp.create_dataset("scene_list", data=np.array(train_folders, dtype='S'))
    for folder in tqdm(train_folders, desc="Writing normalized training scenes"):
      process_scene(folder, data_dir, hf, mean=mean, std=std)

  # Step 4: crea HDF5 val usando stessi parametri
  with h5py.File(val_hdf5_path, 'w') as hf:
    metadata_grp = hf.create_group("metadata")
    metadata_grp.create_dataset("scene_list", data=np.array(val_folders, dtype='S'))
    for folder in tqdm(val_folders, desc="Writing normalized validation scenes"):
      process_scene(folder, data_dir, hf, mean=mean, std=std)
