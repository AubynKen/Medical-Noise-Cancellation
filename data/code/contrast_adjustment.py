"""
Contrast adjustment
Author: Pinglei He

The noisy images and the base images do not have the same color contrast because the normalization step during the
conversion from raw format to png format is done after image normalization. Here we create base images with the same
contrast as the noisy images, so that the base images and noisy images have the same mean.
"""

import numpy as np
from PIL import Image
import os

base_path = "../dataset/validation_set"
files = os.listdir(base_path)

for file in files:
    if not file.endswith(".png"):
        continue
    im = Image.open(os.path.join(base_path, file))
    im = np.array(im, dtype="float")
    im = im / 3 + 15802.056997617085
    im = Image.fromarray(im.astype(np.uint16))
    im.save(f"../dataset/validation_set_contrast_adjusted/{file}")
