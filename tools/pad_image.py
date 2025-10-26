import numpy as np
def pad_image(img, patch_size, stride):
  c, h, w = img.shape
  pad_h = (stride - h % stride) % stride
  pad_w = (stride - w % stride) % stride
  padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
  return padded, h, w
