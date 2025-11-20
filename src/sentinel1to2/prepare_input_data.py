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

  logger = logging.getLogger(__name__)
  logger.info("Preparation of the input samples")
  params = config['preprocessing']
  data_dir = Path(params['input_dir'])
  job_dir = Path(config['job']['dir'])
  job_data_dir = job_dir/'data'
  job_lists_dir = job_data_dir / 'lists'

  train_tmp_hdf5_path = job_data_dir / config["training"]["data"]["train_dataset"] 
  train_hdf5_path = job_data_dir / config["training"]["data"]["train_dataset"] 
  val_hdf5_path = job_data_dir / config["training"]["data"]["val_dataset"] 

  if job_dir.exists():
    answer = input(f"The job folder '{job_dir}' already exists. Do you want to delete it? [yes/N]: ").strip().lower()

    if answer == "yes":
      shutil.rmtree(job_dir)
      job_data_dir.mkdir(parents=True, exist_ok=False)
      job_lists_dir.mkdir()
      logging.info(f"Directory {job_dir} have been recreated.")
    else:
      logging.info(f"The preparation of the samples has been aborted.")
      return

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
  logging.info(f"Saving scene lists.")
  
  write_list_to_csv(job_lists_dir / 'training_scene_list.csv', train_folders)
  write_list_to_csv(job_lists_dir / 'validation_scene_list.csv', val_folders)
  
  norm_params_out_path = job_data_dir / config["preprocessing"]["norm_params_file"] #"normalization_params.npz"
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
    logging.info(f"Saving normalization parameters into {norm_params_out_path}.")
    np.savez(norm_params_out_path, mean=mean, std=std)
  else:
    params = np.load(params['precal_norm_params_path'])
    mean = params["mean"] 
    std =  params["std"]
    logging.info(f"Saving normalization parameters into {norm_params_out_path}.")
    np.savez(norm_params_out_path, mean=mean, std=std)
  
  # Step 3: crea HDF5 train definitivo normalizzato
  logging.info(f"Producing training dataset {train_hdf5_path}")
  with h5py.File(train_hdf5_path, 'w') as hf:
    metadata_grp = hf.create_group("metadata")
    metadata_grp.create_dataset("scene_list", data=np.array(train_folders, dtype='S'))
    for folder in tqdm(train_folders, desc="Writing normalized training scenes"):
      process_scene(folder, data_dir, hf, mean=mean, std=std)

  # Step 4: crea HDF5 val usando stessi parametri
  logging.info(f"Producing validation dataset {val_hdf5_path}")
  with h5py.File(val_hdf5_path, 'w') as hf:
    metadata_grp = hf.create_group("metadata")
    metadata_grp.create_dataset("scene_list", data=np.array(val_folders, dtype='S'))
    for folder in tqdm(val_folders, desc="Writing normalized validation scenes"):
      process_scene(folder, data_dir, hf, mean=mean, std=std)

  logging.info(f"Preparation of the samples perfomed successfully.")
