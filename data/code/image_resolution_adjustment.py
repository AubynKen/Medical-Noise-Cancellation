"""
Creation of better quality base image 1 to 6
Author: Pinglei He

The original images were obtained from 8 bit images. (The original images can be found at folder
data/dataset/base_png_8bit). Therefore, the images 1, 2, 3 have very bad quality and we can visualize the
quantification of their pixels. This script is used to use the image 7, 8, 9, converted into 16 bit images, to
recreate images 1 to 6 with better quality.

The unadjusted images are kept in the folder data/dataset/base_png_8bit.
"""

import numpy as np
from PIL import Image
import os

max_pixel = 2 ** 16 - 1  # 16 bit

im7 = Image.open("../dataset/base_png/base7.png")
im7 = np.array(im7, dtype="float")
im4 = (im7 + max_pixel) / 2 # image 4 is supposed to be
im1 = (im4 + max_pixel) / 2
im4 = Image.fromarray(im4.astype("uint16"))
im1 = Image.fromarray(im1.astype("uint16"))
im4.save("./data/dataset/base_png/base4.png")
im1.save("./data/dataset/base_png/base1.png")


im8 = Image.open("../dataset/base_png/base8.png")
im8 = np.array(im8, dtype="float")
im5 = (im8 + max_pixel) / 2
im2 = (im5 + max_pixel) / 2
im5 = Image.fromarray(im5.astype("uint16"))
im2 = Image.fromarray(im2.astype("uint16"))
im5.save("./data/dataset/base_png/base5.png")
im2.save("./data/dataset/base_png/base2.png")

im9 = Image.open("../dataset/base_png/base9.png")
im9 = np.array(im9, dtype="float")
im6 = (im9 + max_pixel) / 2
im3 = (im6 + max_pixel) / 2
im6 = Image.fromarray(im6.astype("uint16"))
im3 = Image.fromarray(im3.astype("uint16"))
im6.save("./data/dataset/base_png/base6.png")
im3.save("./data/dataset/base_png/base3.png")


