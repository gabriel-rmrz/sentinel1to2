def inspect_hdf5_io_shapes(hdf5_path):
    """
    Ispeziona il file HDF5 per determinare automaticamente
    il numero di input e output canali del modello.

    Restituisce:
    - num_inputs: numero di canali input
    - num_outputs: numero di canali target
    - input_shape: shape (H, W) di un input patch
    - output_shape: shape (H, W) di un target patch
    """
    with h5py.File(hdf5_path, "r") as f:
      for key in f.keys():
        if key.startswith("scene_"):
          input_shape = f[f"{key}/inputs"].shape[1:]  # (C, H, W)
          output_shape = f[f"{key}/targets"].shape[1:]  # (C, H, W)
          num_inputs = input_shape[0]
          num_outputs = output_shape[0]
          return num_inputs, num_outputs, input_shape[1:], output_shape[1:]
      raise ValueError("Nessuna scena trovata nel file HDF5.")

