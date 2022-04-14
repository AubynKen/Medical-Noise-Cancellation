
import numpy as np
from PIL import Image
import skimage.io as sk


def get_tif(path):
    return sk.imread(path, plugin="tifffile")*1.


def noise_gaussin(im, mean=0, std=4):
    return im + np.random.normal(mean, std, size=im.shape)
