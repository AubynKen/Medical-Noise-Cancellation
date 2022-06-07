# author: Pinglei He
# code used for normalizing images and saving them into 16 bit grayscale images in png

from PIL import Image
import os
import numpy as np

path = "../dataset/validation_set_clean"
for file in os.listdir(path):
    if not file.endswith(".png"):
        continue
    im = Image.open(os.path.join(path, file))
    arr = np.array(im, dtype="float")
    arr = (arr * (2 ** 16 - 1) / np.max(arr)).astype("uint16")
    im = Image.fromarray(arr)
    print(im.mode)
    im.save(os.path.join("../dataset/validation_set_bis", file))
    im = Image.open(os.path.join("../dataset/validation_set_bis", file))
    print(im.mode)