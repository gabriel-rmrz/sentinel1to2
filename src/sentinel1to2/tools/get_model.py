import segmentation_models_pytorch as smp
def get_model(config):
  m_name = config["model"]["name"]
  if m_name == "SMP_UNet":
     model = smp.Unet(**config["model"]["parameters"])
  return model
