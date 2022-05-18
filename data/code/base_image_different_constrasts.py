import numpy as np
from PIL import Image
import os

base_path = "../dataset/more_base_png"
files = os.listdir(base_path)

for file in files:
    if not file.endswith(".png"):
        continue
    im = Image.open(os.path.join(base_path, file))

    # high contrast ie. 1.0 contrast
    arr = np.array(im, dtype="float")
    im = Image.fromarray(arr.astype(np.uint16))
    im.save(f"../dataset/validation_set_contrast_adjusted/100_contrast_{file}")

    # mid contrast ie. 0.5 contrast
    arr_mid = 0.5 * arr + 0.5 * (2 ** 16 - 1)
    im = Image.fromarray(arr_mid.astype(np.uint16))
    im.save(f"../dataset/validation_set_contrast_adjusted/050_contrast_{file}")

    # low contrast ie. 0.25 contrast
    arr_low = 0.25 * arr + 0.75 * (2 ** 16 - 1)
    im = Image.fromarray(arr_low.astype(np.uint16))
    im.save(f"../dataset/validation_set_contrast_adjusted/025_contrast_{file}")

    # mid_low contrast ie 0.375 contrast
    arr_mid_low = 0.375 * arr + 0.625 * (2 ** 16 - 1)
    im = Image.fromarray(arr_mid_low.astype(np.uint16))
    im.save(f"../dataset/validation_set_contrast_adjusted/037_contrast_{file}")


