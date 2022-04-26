import os
import numpy as np
from PIL import Image

clean_images = os.listdir("data/dataset/base_png_adjusted")

for image in clean_images:
    im = Image.open("data/dataset/base_png_adjusted/" + image)
    im = np.array(im, dtype="float")
    im += np.random.normal(loc=0, scale=4 * 1398.96558145, size=im.shape)
    im = Image.fromarray(im.astype("uint16"))
    im.save("data/dataset/noisy_png/" + image)