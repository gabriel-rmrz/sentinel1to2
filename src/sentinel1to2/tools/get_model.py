import segmentation_models_pytorch as smp
def get_model(config):
  if config["target"]["type"] == "bands":
    classes = len(config["target"]["selected_bands"])
  elif config["target"]["type"] == "indices":
    classes = len(config["target"]["selected_indices"])

  m_name = config["model"]["name"]
  if m_name == "SMP_UNet":
     model = smp.Unet(**config["model"]["parameters"], classes=classes )
  return model
