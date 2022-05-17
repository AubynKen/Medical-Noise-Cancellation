"""
Creation of noisy images.
Author: Pinglei He
"""


import os
import numpy as np
from PIL import Image

base_std = 1398.96558145
path = "../dataset/validation_set_contrast_adjusted"
target = "../dataset/validation_set_noisy"
clean_images = os.listdir(path)

for image in clean_images:
    im = Image.open(os.path.join(path, image))
    im = np.array(im, dtype="float")
    im += np.random.normal(loc=0, scale=4 * base_std, size=im.shape)
    im = Image.fromarray(im.astype("uint16"))
    im.save(os.path.join(target, image))