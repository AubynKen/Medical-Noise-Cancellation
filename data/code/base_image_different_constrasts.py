# this code was used to generate base images (ie. images used for data-augmentation) with different contrast levels.

import numpy as np
from PIL import Image
import os

base_path = "../dataset/base_img_orig_contrast"
files = os.listdir(base_path)

for file in files:
    if not file.endswith(".jpg"):
        continue
    im = Image.open(os.path.join(base_path, file))

    # high contrast ie. 1.0 contrast
    arr = np.array(im, dtype="float")
    arr = arr * (2 ** 16 - 1) / np.max(arr)
    im = Image.fromarray(arr.astype(np.uint16))
    im.save(f"../dataset/base_img_different_contrast/100_contrast_{file}")

    # mid contrast ie. 0.5 contrast
    arr_mid = 0.5 * arr + 0.5 * (2 ** 16 - 1)
    im = Image.fromarray(arr_mid.astype(np.uint16))
    im.save(f"../dataset/base_img_different_contrast/050_contrast_{file}")

    # low contrast ie. 0.25 contrast
    arr_low = 0.25 * arr + 0.75 * (2 ** 16 - 1)
    im = Image.fromarray(arr_low.astype(np.uint16))
    im.save(f"../dataset/base_img_different_contrast/025_contrast_{file}")

    # mid_low contrast ie 0.375 contrast
    arr_mid_low = 0.375 * arr + 0.625 * (2 ** 16 - 1)
    im = Image.fromarray(arr_mid_low.astype(np.uint16))
    im.save(f"../dataset/base_img_different_contrast/037_contrast_{file}")


